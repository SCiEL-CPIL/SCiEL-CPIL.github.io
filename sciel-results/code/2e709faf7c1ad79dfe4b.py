"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    # Number of edges crossing the partition (each edge counted once).
    return int(np.sum(adj[labels == 0, labels == 1]))


def _local_search(adj, deg, labels, deadline):
    """Vectorized single-flip local search. Returns (labels, cut)."""
    n = len(labels)
    labels = labels.copy()
    # gain[v] = change in cut if v is flipped = 2*cross(v) - deg(v)
    cross = adj @ (1 - labels)
    gain = 2 * cross - deg
    while True:
        if time.time() > deadline:
            break
        pos = gain > 0
        if not pos.any():
            break
        labels[pos] = 1 - labels[pos]
        # Update gain for flipped vertices and their neighbors exactly.
        for v in np.flatnonzero(pos):
            nb = np.flatnonzero(adj[v])
            if len(nb) == 0:
                continue
            cross_nb = adj[nb] @ (1 - labels)
            gain[nb] = 2 * cross_nb - deg[nb]
            gain[v] = 2 * (adj[v] @ (1 - labels)) - deg[v]
    cut = _cut_size(adj, labels)
    return labels, cut


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    deadline = time.time() + 140.0  # safety margin before 150s kill

    deg = adj.sum(axis=1).astype(np.int64)

    # --- Spectral (Fiedler) initialization ---
    labels = None
    try:
        L = np.diag(deg.astype(np.float64)) - adj.astype(np.float64)
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector = eigenvector of the 2nd smallest eigenvalue.
        fiedler = eigvecs[:, 1]
        labels = (fiedler >= fiedler.mean()).astype(np.int8)
        # Ensure both sides are non-empty.
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.arange(n) % 2
    except Exception:
        labels = np.arange(n) % 2

    best_labels, best_cut = _local_search(adj, deg, labels, deadline)

    # --- Iterated Local Search with random perturbations ---
    # Number of flips to perturb scales with graph density so it stays
    # meaningful for sparse and denser 100-node graphs alike.
    m = int(np.sum(deg)) // 2  # number of edges
    base_k = max(2, int(round(0.04 * n + 0.0015 * m)))

    rng = np.random.default_rng(12345)
    while time.time() < deadline:
        # Perturb the current best solution.
        cand = best_labels.copy()
        k = int(rng.integers(max(1, base_k - 2), base_k + 3))
        flip_idx = rng.choice(n, size=k, replace=False)
        cand[flip_idx] = 1 - cand[flip_idx]

        new_labels, new_cut = _local_search(adj, deg, cand, deadline)
        if new_cut > best_cut:
            best_cut = new_cut
            best_labels = new_labels
            # Occasionally kick harder to escape a long plateau.
            if time.time() < deadline:
                k2 = int(rng.integers(base_k, base_k + 6))
                flip_idx2 = rng.choice(n, size=k2, replace=False)
                best_labels[flip_idx2] = 1 - best_labels[flip_idx2]
                best_labels, best_cut = _local_search(adj, deg, best_labels, deadline)

    return best_labels
# EVOLVE-BLOCK-END
