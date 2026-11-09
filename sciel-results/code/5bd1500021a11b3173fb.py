"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Solve Maximum Cut using spectral initialization (top eigenvector of the
    signed Laplacian) followed by an incremental-delta greedy 1-flip polish.

    adjacency: bool matrix (n x n)
    returns: 0/1 label per vertex (numpy array of int)
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)

    adj = np.array(adjacency, dtype=np.int8)

    start_time = time.time()
    deadline = start_time + 135.0  # safety margin under 150s

    # Build degree vector and the signed Laplacian L = D - A.
    # For a 0/1 labeling x, x^T L x = 2 * (cut size), so the top eigenvector
    # of L gives a principled global starting partition.
    deg = adj.sum(axis=1).astype(np.float64)
    L = np.diag(deg) - adj.astype(np.float64)

    # Spectral initialization via the top eigenvector.
    try:
        # eigh is more stable for symmetric matrices; take the last eigenvector.
        vals, vecs = np.linalg.eigh(L)
        v = vecs[:, -1]
        # Threshold at the mean to balance partition sizes.
        labels = (v > np.mean(v)).astype(np.int8)
    except Exception:
        labels = (np.arange(n) % 2).astype(np.int8)

    # Incremental-delta greedy 1-flip polish.
    # Maintain cut = number of crossing edges.
    # For vertex v, let same = # neighbors in same side, diff = # neighbors in
    # other side. Flipping v changes cut by (same - diff):
    #   - its 'diff' edges become non-crossing (lose diff)
    #   - its 'same' edges become crossing (gain same)
    # So delta = same - diff. Flip if delta > 0.

    def compute_cut(labels):
        # 0.5 * sum_{i,j} adj[i,j] * (labels[i]-labels[j])^2
        diff = (labels[:, None] - labels[None, :]) ** 2
        return int(np.sum(adj * diff) // 2)

    cut = compute_cut(labels)
    best_cut = cut
    best_labels = labels.copy()

    # Precompute neighbor lists for fast same/diff counts.
    # Use row sums weighted by label agreement.
    # same(v) = sum_j adj[v,j] * (labels[j] == labels[v])
    # diff(v) = sum_j adj[v,j] * (labels[j] != labels[v])
    # We can compute these via matrix-vector products with the label vector.

    def greedy_polish(labels, cut, max_passes=50):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline - 5.0:
                break
            improved = False
            passes += 1
            # Compute same/diff for all vertices at once.
            # same_vec[v] = sum_j adj[v,j] * (labels[j] == labels[v])
            # Let same_label = (labels == labels[v]) for each v; this is a matrix.
            # Efficiently: 
            #   match = (labels[:, None] == labels[None, :])  # n x n bool
            #   same_vec = adj @ match (but match depends on v's label)
            # Simpler: 
            #   same_vec[v] = sum_j adj[v,j] where labels[j]==labels[v]
            #   = (adj * (labels == labels[v])) summed over j
            # We can do: 
            #   same_vec = np.array([np.sum(adj[v] * (labels == labels[v])) for v in range(n)])
            # But that's O(n^2) per pass. For n=100, that's fine.
            # Vectorized:
            match = (labels[:, None] == labels[None, :])  # n x n
            same_vec = (adj.astype(np.int32) * match.astype(np.int32)).sum(axis=1)
            diff_vec = deg.astype(np.int32) - same_vec
            delta = same_vec - diff_vec  # = 2*same_vec - deg
            # Flip vertices with positive delta, one at a time in descending order.
            order = np.argsort(-delta)
            for v in order:
                if time.time() > deadline - 5.0:
                    break
                d = delta[v]
                if d > 0:
                    # Flip vertex v
                    labels[v] = 1 - labels[v]
                    cut += int(d)
                    improved = True
                    # Recompute delta for all affected vertices (neighbors of v and v itself).
                    # For efficiency with n=100, just recompute the whole thing.
                    match = (labels[:, None] == labels[None, :])
                    same_vec = (adj.astype(np.int32) * match.astype(np.int32)).sum(axis=1)
                    diff_vec = deg.astype(np.int32) - same_vec
                    delta = same_vec - diff_vec
        return labels, cut

    labels, cut = greedy_polish(labels, cut, max_passes=100)

    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()

    # If we still have time, do a few more passes with different spectral seeds
    # to escape any local optima.
    if time.time() < deadline - 20.0:
        for seed in range(3):
            if time.time() > deadline - 10.0:
                break
            # Perturb the spectral solution: flip a small random subset.
            labels2 = best_labels.copy()
            num_flips = max(1, n // 10)
            flip_indices = np.random.choice(n, size=num_flips, replace=False)
            labels2[flip_indices] = 1 - labels2[flip_indices]
            cut2 = compute_cut(labels2)
            labels2, cut2 = greedy_polish(labels2, cut2, max_passes=50)
            if cut2 > best_cut:
                best_cut = cut2
                best_labels = labels2.copy()

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
