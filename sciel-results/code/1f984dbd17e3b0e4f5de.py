# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Wall-clock budget with a safety margin (evaluator kills at 150 s).
_DEADLINE = 120.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def _cut_size(adj, labels):
    """Number of undirected edges crossing the 0/1 partition."""
    s = 2 * labels.astype(np.float64) - 1          # +/-1
    # cut = 0.5 * (sum(A) - s^T A s)
    return 0.5 * (adj.sum() - s @ (adj @ s))


def _spectral_init(adj):
    """Fiedler-vector (second eigenvector of the Laplacian) partition."""
    n = adj.shape[0]
    A = adj.astype(np.float64)
    deg = A.sum(axis=1)
    L = sp.csr_matrix(np.diag(deg) - A)
    # Constant vector is the zero eigenvector; find the next-smallest eigenpair.
    try:
        vals, vecs = spla.eigsh(L, k=2, which='SM')
        # eigsh returns ascending; index 0 is ~0 (constant), index 1 is Fiedler.
        v = vecs[:, 1]
    except Exception:
        # Fallback: random unit vector.
        v = np.random.randn(n)
        v /= (np.linalg.norm(v) + 1e-12)
    labels = (v >= 0).astype(np.int64)
    # Avoid a degenerate all-one partition.
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.arange(n) % 2
    return labels


def _greedy_refine(adj, labels, max_passes=200):
    """Vectorized best-improvement greedy pass; stop when no flip helps."""
    n = adj.shape[0]
    best = _cut_size(adj, labels)
    s = 2 * labels.astype(np.float64) - 1

    for _ in range(max_passes):
        if _remaining() < 2.0:
            break
        # For vertex v currently +1: flipping to -1 changes cut by
        #   delta_v = 2 * (sum_{u: s[u]=-1} A[v,u] - sum_{u: s[u]=+1} A[v,u])
        # In matrix form with the current s (before any flip):
        #   s^T A s = 2*sum_{same} - 2*sum_{diff}
        #   A.sum()  = 2*sum_{same} + 2*sum_{diff}
        #   => 2*sum_diff - 2*sum_same = A.sum() - s^T A s
        #   => delta_v = A.sum() - s^T A s  (per-vertex, using current s)
        delta = adj.sum() - s @ (adj @ s)
        # A flip is beneficial iff delta_v > 0.
        flip = delta > 0
        if not flip.any():
            break
        s[flip] = -s[flip]
        labels[flip] = 1 - labels[flip]
        new_cut = _cut_size(adj, labels)
        if new_cut <= best:
            # Should not happen with strict improvement; keep best state.
            break
        best = new_cut
    return labels, best


def _random_init(n, rng):
    p = 0.5
    labels = (rng.random(n) < p).astype(np.int64)
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.arange(n) % 2
    return labels


def partition(adjacency):
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)

    rng = np.random.default_rng(0)

    # Diverse initializations: spectral + several random restarts.
    inits = []
    inits.append(_spectral_init(adj))
    for _ in range(8):
        inits.append(_random_init(n, rng))

    best_labels = None
    best_cut = -1.0

    for init in inits:
        if _remaining() < 3.0:
            break
        labels, cut = _greedy_refine(adj, init)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Final high-precision polish on the best solution (tighter, more passes).
    if best_labels is not None and _remaining() > 2.0:
        polished, polished_cut = _greedy_refine(adj, best_labels, max_passes=500)
        if polished_cut >= best_cut:
            best_labels = polished
            best_cut = polished_cut

    if best_labels is None:
        best_labels = np.arange(n) % 2

    # Safety margin: return the best found well before the wall-clock limit.
    if _remaining() < 1.0:
        return best_labels.astype(np.int64)
    return best_labels.astype(np.int64)
# EVOLVE-BLOCK-END
