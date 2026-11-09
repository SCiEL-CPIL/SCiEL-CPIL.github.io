"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Number of edges crossing the 0/1 partition."""
    # cut = sum over v of (neighbors of v in opposite side)
    # For 0/1 labels: opposite = 1 - labels
    # Each edge counted twice (once from each endpoint), so divide by 2.
    return int(np.sum(adj @ (1 - labels)) // 2)


def _tabu_search(adj, labels, tabu_len, iters, deadline):
    """Cerny-style tabu search. Returns best labels found."""
    n = len(adj)
    labels = labels.copy()
    best_labels = labels.copy()
    best_cut = _cut_size(adj, labels)

    # current cut for quick delta computation
    cur_cut = best_cut
    tabu_until = np.zeros(n, dtype=np.int64)
    tabu_counter = 0
    rng = np.random.default_rng()

    for it in range(iters):
        if time.time() >= deadline:
            break
        # Pick a random vertex to consider flipping
        v = rng.integers(0, n)
        # Gain of flipping v:
        # flipping v changes cut by: (neighbors in opposite) - (neighbors in same)
        # neighbors in opposite = adj[v] @ (1 - labels)
        # neighbors in same = adj[v] @ labels
        opp = int(adj[v] @ (1 - labels))
        same = int(adj[v] @ labels)
        gain = opp - same
        # tabu check with aspiration
        is_tabu = tabu_counter < tabu_until[v]
        if gain > 0 or (not is_tabu and gain > 0) or (not is_tabu and gain >= 0 and rng.random() < 0.5):
            # accept: improving move, or non-tabu worsening move (Cerny acceptance)
            pass
        else:
            # worsening tabu move -> skip
            continue

        # perform flip
        labels[v] = 1 - labels[v]
        tabu_until[v] = tabu_counter + tabu_len
        tabu_counter += 1
        cur_cut += gain
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = labels.copy()

    return best_labels, best_cut


def _spectral_init(adj, rng):
    """Initialize labels using the leading eigenvector of (D - A)."""
    n = len(adj)
    D = np.diag(adj.sum(axis=1))
    L = D - adj
    # Use power iteration for leading eigenvector (sparse-friendly enough for n=100)
    x = rng.standard_normal(n)
    x = x / (np.linalg.norm(x) + 1e-12)
    for _ in range(50):
        y = L @ x
        nrm = np.linalg.norm(y)
        if nrm < 1e-12:
            break
        x = y / nrm
    # threshold at median
    med = np.median(x)
    labels = (x > med).astype(np.int64)
    return labels


def _greedy_flip(adj, labels, max_passes=5):
    """Standard greedy 1-flip local search."""
    n = len(adj)
    labels = labels.copy()
    for _ in range(max_passes):
        improved = False
        for v in range(n):
            opp = int(adj[v] @ (1 - labels))
            same = int(adj[v] @ labels)
            if opp > same:
                labels[v] = 1 - labels[v]
                improved = True
        if not improved:
            break
    return labels


def partition(adjacency):
    adj = np.asarray(adjacency, dtype=np.int64)
    n = len(adj)
    deadline = time.time() + 110.0  # safety margin before 120s

    rng = np.random.default_rng(0)

    best_labels = None
    best_cut = -1

    # Multiple restarts with different strategies
    n_restarts = 0
    while time.time() < deadline - 5.0:
        n_restarts += 1
        # Alternate between spectral init and random init
        if n_restarts % 3 == 0:
            labels = rng.integers(0, 2, size=n)
        elif n_restarts % 3 == 1:
            labels = _spectral_init(adj, rng)
        else:
            # alternating start
            labels = np.arange(n) % 2

        # Greedy improve first
        labels = _greedy_flip(adj, labels)
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

        # Tabu search from this starting point
        tabu_labels, tabu_cut = _tabu_search(adj, labels, tabu_len=10, iters=2000, deadline=deadline)
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels.copy()

        # Final greedy polish
        polished = _greedy_flip(adj, best_labels)
        p_cut = _cut_size(adj, polished)
        if p_cut > best_cut:
            best_cut = p_cut
            best_labels = polished

    return best_labels
# EVOLVE-BLOCK-END
