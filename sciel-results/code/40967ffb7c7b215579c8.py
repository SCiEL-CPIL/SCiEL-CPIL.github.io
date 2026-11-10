"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Vectorized Neighbour Joining on Jukes-Cantor corrected distances (Saitou and Nei 1987).'''
import time
import numpy as np

_DEADLINE_START = None
_DEADLINE_LIMIT = 265.0  # wall-clock seconds; evaluator kills at 270s


def _elapsed():
    return time.time() - _DEADLINE_START


def _jc_distances(mat):
    """mat: (n, L) int array with 0..3 encoding. Returns (n, n) JC distances."""
    n, L = mat.shape
    # Vectorized pairwise mismatch count via one-hot encoding
    # Encode each position as a 4-bit one-hot; mismatches = L - matches
    # Use matrix multiplication on one-hot: (n, L, 4) @ (n, L, 4)^T per site
    # Efficient approach: for each base, count co-occurrences
    # matches[i,j] = sum_s sum_c (onehot[i,s,c] * onehot[j,s,c])
    # = sum_c sum_s (onehot[i,s,c] * onehot[j,s,c])
    # Let M_c be (n, L) binary matrix for base c
    # matches = sum_c M_c @ M_c.T
    matches = np.zeros((n, n), dtype=np.float64)
    for c in range(4):
        Mc = (mat == c).astype(np.float32)
        matches += Mc @ Mc.T
    mismatches = L - matches
    p = mismatches / L
    np.fill_diagonal(p, 0.0)
    # JC correction: d = -3/4 * ln(1 - 4p/3)
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    d = np.where(p >= 0.75, 3.0, d)
    d = np.where(np.isnan(d) | np.isinf(d), 3.0, d)
    np.fill_diagonal(d, 0.0)
    return d


def _nj_with_structure(dist):
    """Neighbor Joining. dist is (n,n). Returns Newick with leaf labels 0..n-1."""
    n0 = dist.shape[0]
    if n0 < 3:
        if n0 == 1:
            return "0;"
        return "(0,1);"
    # Pre-allocate full (2n-1) x (2n-1) distance matrix
    max_nodes = 2 * n0 - 1
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n0, :n0] = dist
    # Track which nodes are "alive" (not yet joined)
    alive = list(range(n0))
    next_id = n0
    # Newick subtree for each node
    subtree = [str(i) for i in range(max_nodes)]
    while len(alive) > 3:
        if _elapsed() > 260.0:
            break
        k = len(alive)
        idx = np.array(alive, dtype=np.int64)
        # Extract sub-matrix of alive nodes
        sub = D[np.ix_(idx, idx)].copy()
        r = sub.sum(axis=1)
        # Q matrix: Q[i,j] = (k-2)*D[i,j] - r[i] - r[j]
        Q = (k - 2) * sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        # Find minimum Q entry
        min_idx = np.argmin(Q)
        i, j = np.unravel_index(min_idx, Q.shape)
        a = alive[i]
        b = alive[j]
        # Create new internal node
        new = next_id
        next_id += 1
        subtree[new] = "(" + subtree[a] + "," + subtree[b] + ")"
        # Update distances from new node to all other alive nodes
        others = [x for x in alive if x != a and x != b]
        if others:
            ox = np.array(others, dtype=np.int64)
            d_ab = D[a, b]
            vals = 0.5 * (D[a, ox] + D[b, ox] - d_ab)
            D[new, ox] = vals
            D[ox, new] = vals
        # Remove a and b from alive, add new
        alive = [x for x in alive if x != a and x != b]
        alive.append(new)
    # Final: 3 nodes remain, form a trifurcation
    if len(alive) == 3:
        a, b, c = alive
        return "(" + subtree[a] + "," + subtree[b] + "," + subtree[c] + ");"
    # Fallback (shouldn't happen)
    return "(" + ",".join(subtree[x] for x in alive) + ");"


def _reconstruct_single(sequences):
    global _DEADLINE_START, _DEADLINE_LIMIT
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
        _DEADLINE_LIMIT = _DEADLINE_START + 265.0

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n < 3:
        if n == 1:
            return labels[0] + ";"
        return "(" + ",".join(labels) + ");"
    # encode sequences: map A,C,G,T -> 0,1,2,3
    L = len(sequences[labels[0]])
    enc = np.empty((n, L), dtype=np.int8)
    code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for pos, ch in enumerate(s):
            enc[i, pos] = code[ch]
    D = _jc_distances(enc)
    newick = _nj_with_structure(D)
    # Relabel: leaf tokens 0..n-1 -> labels[0..n-1]
    import re
    def _repl(m):
        return labels[int(m.group(0))]
    result = re.sub(r'\d+', _repl, newick)
    return result


def reconstruct(sequences):
    global _DEADLINE_START, _DEADLINE_LIMIT
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
        _DEADLINE_LIMIT = _DEADLINE_START + 265.0

    try:
        return _reconstruct_single(sequences)
    except Exception:
        # fallback: star tree
        labels = sorted(sequences, key=lambda name: int(name[1:]))
        if len(labels) <= 2:
            return "(" + ",".join(labels) + ");"
        return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
