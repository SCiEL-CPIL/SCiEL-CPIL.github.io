"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

DEADLINE_S = 110.0  # safety margin under the 120 s / 150 s kill limit


def _cut_size(adj, labels):
    # labels: 0/1 int array. cut = # edges with endpoints on opposite sides.
    # Using 0/1: contribution per directed edge v->u is labels[v]*(1-labels[u]).
    # Sum over all v,u then halve (each undirected edge counted once per direction
    # only when crossing, so actually each crossing edge counted exactly once in
    # the directed sum? Let's verify: crossing edge (v in A, u in B):
    #   term v->u: labels[v]*(1-labels[u]) = 1*1 = 1
    #   term u->v: labels[u]*(1-labels[v]) = 0*0 = 0
    # So each crossing edge contributes exactly 1. Good, no halving needed.
    return int(np.sum(labels * (1 - adj.dot(labels))))


def _spectral_init(adj, n, rng):
    # Build the SDP-style matrix: maximize x^T A x s.t. x_i^2=1.
    # Use the Fiedler (2nd eigenvector) of L = D - A as the main axis,
    # plus a random axis to break symmetry and add a 2nd spectral direction.
    D = np.asarray(adj, dtype=float).sum(axis=1)
    L = np.diag(D) - adj
    try:
        w, V = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return rng.integers(0, 2, size=n)
    # Fiedler vector (smallest non-trivial) -> index 1
    fiedler = V[:, 1]
    # Second smallest for extra structure
    second = V[:, 2] if n >= 3 else np.zeros(n)
    # Random axis
    rand_ax = rng.standard_normal(n)
    # Combine into 3D embedding
    emb = np.column_stack([fiedler, second, rand_ax])
    # Random hyperplane in 3D
    h = rng.standard_normal(3)
    proj = emb @ h
    # Threshold at median to keep partition balanced
    thr = np.median(proj)
    return (proj >= thr).astype(np.int8)


def _greedy_flip(adj, labels, n, rng):
    # Passes of single-flip improvement until no improvement.
    improved = True
    passes = 0
    while improved and passes < 50:
        improved = False
        passes += 1
        order = rng.permutation(n)
        for v in order:
            lv = labels[v]
            same = int(adj[v].dot(labels))            # neighbors same side as v
            other = int(adj[v].sum() - same)          # neighbors opposite side
            # Flipping v: gain = other - same
            if other > same:
                labels[v] = 1 - lv
                improved = True
    return labels


def _two_swap(adj, labels, n, rng, max_rounds=8):
    # Try random pairs of vertices, flip both if it improves.
    improved_any = False
    for _ in range(max_rounds):
        improved = False
        for _ in range(n):  # n random pair trials per round
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            li, lj = labels[i], labels[j]
            if li == lj:
                continue
            # Compute delta of flipping both i and j.
            # Use incremental: current cut contribution involving i and j.
            # Simpler: compute full delta via neighbor counts.
            # gain for flipping i alone = other_i - same_i
            same_i = int(adj[i].dot(labels))
            other_i = int(adj[i].sum() - same_i)
            same_j = int(adj[j].dot(labels))
            other_j = int(adj[j].sum() - same_j)
            # When flipping both, the i-j edge interaction changes.
            # Approximate combined gain: (other_i - same_i) + (other_j - same_j)
            # plus correction for the edge (i,j) if it exists and crosses.
            # We'll just do a direct delta by temporarily evaluating.
            gain = (other_i - same_i) + (other_j - same_j)
            # If edge (i,j) exists and currently crosses, after flipping both
            # (which keeps them on opposite sides) it still crosses -> no change.
            # If they were same side (not here) it'd matter. So fine.
            if gain > 0:
                labels[i] = 1 - li
                labels[j] = 1 - lj
                improved = True
                improved_any = True
        if not improved:
            break
    return labels, improved_any


def partition(adjacency):
    start = time.time()
    adj = np.asarray(adjacency)
    if adj.ndim != 2:
        raise ValueError("adjacency must be a 2D matrix")
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=np.int8)
    # Symmetrize and ensure bool/int
    A = ((adj + adj.T) > 0).astype(np.int8)

    rng = np.random.default_rng(12345)

    best_labels = None
    best_cut = -1

    def deadline():
        return (time.time() - start) >= DEADLINE_S

    # --- Phase 1: spectral + random initializations, each improved ---
    init_count = 0
    while not deadline():
        if init_count % 5 == 0:
            init = _spectral_init(A, n, rng)
        else:
            init = rng.integers(0, 2, size=n).astype(np.int8)
        init_count += 1

        labels = init.copy()
        _greedy_flip(A, labels, n, rng)
        labels, _ = _two_swap(A, labels, n, rng, max_rounds=6)
        cut = _cut_size(A, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

        # Stop early if we've hit a strong cut (e.g., near half of edges)
        total_edges = int(A.sum()) // 2
        if best_cut >= 0.5 * total_edges:
            # keep a little more searching but cap it
            if init_count >= 40:
                break

    # --- Phase 2: perturb best and re-improve while time remains ---
    while best_labels is not None and not deadline():
        cand = best_labels.copy()
        # random perturbation: flip a handful of vertices
        k = rng.integers(1, max(2, n // 10) + 1)
        flip_idx = rng.choice(n, size=k, replace=False)
        cand[flip_idx] = 1 - cand[flip_idx]
        _greedy_flip(A, cand, n, rng)
        cand, _ = _two_swap(A, cand, n, rng, max_rounds=6)
        cut = _cut_size(A, cand)
        if cut > best_cut:
            best_cut = cut
            best_labels = cand.copy()

    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2

    return best_labels
# EVOLVE-BLOCK-END
