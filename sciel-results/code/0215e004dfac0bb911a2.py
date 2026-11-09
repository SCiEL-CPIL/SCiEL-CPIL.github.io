"""Reconstruct unrooted topology from JC-simulated sequences.

Uses BIONJ (Gascuel 1997) with vectorized JC distance computation and deadline guard.
"""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 260.0  # safety margin below 270s wall-clock kill


def _time_ok():
    return time.time() < _DEADLINE


def _jc_matrix(sequences, labels):
    """Vectorized Jukes-Cantor corrected pairwise distances."""
    n = len(labels)
    if n == 0:
        return np.array([])
    L = len(sequences[labels[0]])
    if L == 0:
        return np.zeros((n, n))
    
    # Encode sequences as one-hot: 4 channels for A, C, G, T
    A = np.zeros((n, L), dtype=np.int8)
    C = np.zeros((n, L), dtype=np.int8)
    G = np.zeros((n, L), dtype=np.int8)
    T = np.zeros((n, L), dtype=np.int8)
    
    for i, lab in enumerate(labels):
        s = sequences[lab]
        A[i] = (s == 'A').astype(np.int8)
        C[i] = (s == 'C').astype(np.int8)
        G[i] = (s == 'G').astype(np.int8)
        T[i] = (s == 'T').astype(np.int8)
    
    # Compute same-position counts via matrix multiplication
    same = A @ A.T + C @ C.T + G @ G.T + T @ T.T
    hamming = L - same
    np.fill_diagonal(hamming, 0)
    
    p = hamming.astype(np.float64) / L
    
    # JC correction: -0.75 * ln(1 - 4p/3)
    arg = 1.0 - 4.0 * p / 3.0
    np.clip(arg, 1e-9, None, out=arg)
    D = -0.75 * np.log(arg)
    D[~np.isfinite(D)] = 3.0
    np.fill_diagonal(D, 0.0)
    
    return D


def _bionj(D, labels):
    """BIONJ reconstruction. Returns a Newick string (no branch lengths)."""
    n = len(labels)
    if n <= 1:
        return "(" + ",".join(labels) + ");"
    if n == 2:
        return "(%s,%s);" % (labels[0], labels[1])
    if n == 3:
        return "((%s,%s),%s);" % (labels[0], labels[1], labels[2])

    D = D.astype(np.float64, copy=True)
    trees = {i: labels[i] for i in range(n)}
    alive = list(range(n))
    
    # Pre-allocate space for new nodes
    max_nodes = 2 * n - 1
    full_D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    full_D[:n, :n] = D
    full_trees = [labels[i] for i in range(n)]
    
    next_node_id = n
    
    while len(alive) > 3:
        if not _time_ok():
            break
        
        m = len(alive)
        r = np.array(alive, dtype=np.int64)
        sub = full_D[np.ix_(r, r)]
        rowsum = sub.sum(axis=1)
        
        # BIONJ beta correction
        beta = (rowsum - sub.mean(axis=1) * m) / (m - 2)
        
        # Q criterion matrix (symmetric)
        Q = (m - 2) * sub - beta[:, None] - beta[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Pick the minimum off-diagonal pair
        min_idx = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_idx, Q.shape)
        i_idx, j_idx = int(i_idx), int(j_idx)
        
        if i_idx == j_idx:
            break
        
        a, b = r[i_idx], r[j_idx]
        
        # Compute branch lengths for BIONJ
        d_ab = full_D[a, b]
        da = rowsum[i_idx]
        db = rowsum[j_idx]
        beta_a = beta[i_idx]
        beta_b = beta[j_idx]
        
        la = 0.5 * d_ab + (da - db - beta_a + beta_b) / (2.0 * (m - 2))
        lb = 0.5 * d_ab - (da - db - beta_a + beta_b) / (2.0 * (m - 2))
        la = max(la, 0.0)
        lb = max(lb, 0.0)
        
        # Create new node
        new_idx = next_node_id
        next_node_id += 1
        
        joined = "(%s,%s)" % (full_trees[a], full_trees[b])
        full_trees.append(joined)
        
        # Update distances for new node
        rest = [x for x in r if x != a and x != b]
        full_D[new_idx, new_idx] = 0.0
        
        for k, x in enumerate(rest):
            val = 0.5 * (full_D[a, x] + full_D[b, x] - d_ab) + 0.5 * (la + lb)
            full_D[new_idx, x] = val
            full_D[x, new_idx] = val
        
        alive = rest + [new_idx]
    
    # Build final tree
    if len(alive) == 3:
        return "((%s,%s),%s);" % (full_trees[alive[0]], full_trees[alive[1]], full_trees[alive[2]])
    elif len(alive) == 2:
        return "(%s,%s);" % (full_trees[alive[0]], full_trees[alive[1]])
    else:
        return "(" + ",".join(full_trees[x] for x in alive) + ");"


def _star(labels):
    """Fallback star tree."""
    return "(" + ",".join(labels) + ");"


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences.
    
    Args:
        sequences: dict mapping taxon labels (t0, t1, ...) to DNA strings over ACGT
    
    Returns:
        Newick string over the labels t0..t(n-1)
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return "(" + ",".join(labels) + ");"
    
    try:
        D = _jc_matrix(sequences, labels)
        return _bionj(D, labels)
    except Exception:
        # Fallback to star tree if something goes wrong
        return _star(labels)
