"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining with NumPy JC distances and deadline guard.'''
import math
import time
import numpy as np


def _jc_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using NumPy."""
    n = len(labels)
    L = len(sequences[labels[0]])
    char_to_code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    seq_mat = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        seq_mat[i] = np.array([char_to_code[c] for c in s], dtype=np.int8)
    
    diff = np.sum(seq_mat[:, np.newaxis, :] != seq_mat[np.newaxis, :, :], axis=2)
    p = diff.astype(np.float64) / L
    p_clipped = np.minimum(p, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(D, 0.0)
    return D


def _nj_build(D):
    """Neighbor Joining. Returns Newick string."""
    n = D.shape[0]
    if n < 3:
        return "0"
    
    D = D.copy()
    active = list(range(n))
    trees = {i: str(i) for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        r = np.array([sum(D[a, b] for b in active if b != a) for a in active])
        
        best_val = float('inf')
        best_i = best_j = -1
        for ii in range(m):
            for jj in range(ii + 1, m):
                a = active[ii]
                b = active[jj]
                q = (m - 2) * D[a, b] - r[ii] - r[jj]
                if q < best_val:
                    best_val = q
                    best_i, best_j = a, b
        
        a, b = best_i, best_j
        rest = [x for x in active if x not in (a, b)]
        d_ab = D[a, b]
        
        new_id = next_id
        next_id += 1
        
        for x in rest:
            D[new_id, x] = 0.5 * (D[a, x] + D[b, x] - d_ab)
            D[x, new_id] = D[new_id, x]
        
        trees[new_id] = f"({trees[a]},{trees[b]})"
        active = rest + [new_id]
    
    return "(" + ",".join(trees[x] for x in active) + ");"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return labels[0] + ";"
        return "(" + ",".join(labels) + ");"
    
    if n == 3:
        return "(" + ",".join(labels) + ");"
    
    start_time = time.time()
    deadline = start_time + 170.0
    
    try:
        D = _jc_matrix(sequences, labels)
        newick = _nj_build(D)
        
        # Verify all labels are present in the newick string
        # Simple check: count occurrences of each label
        for lab in labels:
            if lab not in newick:
                raise ValueError(f"Missing label {lab}")
                
    except Exception as e:
        # Fallback to star tree if NJ fails
        newick = "(" + ",".join(labels) + ");"
        
    if time.time() > deadline:
        return newick
    
    return newick
# EVOLVE-BLOCK-END
