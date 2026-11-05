"""Rewritten program: spectral + degree + spread priority for greedy MIS."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _principal_eigenvector(adj, max_iter=200, tol=1e-9, time_budget=25.0):
    """Unit-norm principal eigenvector of a symmetric 0/1 adjacency matrix."""
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0)
    A = np.asarray(adj, dtype=np.float64)
    # Try scipy sparse eigensolver first (fast, robust for ER graphs).
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        As = csr_matrix(A)
        if n >= 3:
            vals, vecs = eigsh(As, k=1, which="LA", return_eigenvectors=True)
            v = vecs[:, 0]
        else:
            vals, vecs = np.linalg.eigh(A)
            v = vecs[:, -1]
        v = np.asarray(v, dtype=np.float64).ravel()
        if np.all(np.isfinite(v)) and np.linalg.norm(v) > 1e-12:
            return v / np.linalg.norm(v)
    except Exception:
        pass
    # Fallback: power iteration with Rayleigh quotient convergence.
    v = np.ones(n, dtype=np.float64)
    v /= np.linalg.norm(v)
    start = time.time()
    prev = 0.0
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        w = A @ v
        nw = np.linalg.norm(w)
        if nw < 1e-15:
            break
        w /= nw
        lam = float(v @ w)
        v = w
        if abs(lam - prev) < tol:
            prev = lam
            break
        prev = lam
    return v


def _normalize(x):
    """Robust z-score; returns zeros for degenerate input."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    std = x.std()
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def priority(adjacency):
    start = time.time()
    adj = np.asarray(adjacency)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        return np.zeros(adj.shape[0] if adj.ndim else 0, dtype=np.float64)
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Symmetrize to a clean 0/1 matrix.
    A = ((adj.astype(np.float64) + adj.T.astype(np.float64)) > 0.5).astype(np.float64)
    np.fill_diagonal(A, 0.0)

    deg = A.sum(axis=1)                       # degree of each vertex

    # Spectral centrality: high = hub. Penalize hubs (peripheral picks spread out).
    v = _principal_eigenvector(A)
    v = v - v.mean()                          # center so sign reflects relative centrality

    # Spread bonus: vertices far from high-degree neighbors are easier to pack.
    # d_j = sum over neighbors i of (deg_i / (deg_i + 1))  -> high for peripheral vertices.
    d = A @ (deg / (deg + 1.0))

    z_deg = _normalize(deg)
    z_v = _normalize(v)
    z_d = _normalize(d)

    # Higher score = picked earlier. Low degree + low centrality + high spread.
    score = -1.0 * z_deg - 0.5 * z_v + 0.5 * z_d

    # Safety margin: ensure we return well before the 120s wall-clock limit.
    if time.time() - start > 100.0:
        return -deg.astype(np.float64)

    return score.astype(np.float64)
# EVOLVE-BLOCK-END
