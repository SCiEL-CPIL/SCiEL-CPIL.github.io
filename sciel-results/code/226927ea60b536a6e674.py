"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]
_DEADLINE = 140.0

def _strassen_222(U, V, W, i_0, j_0, k_0, m, p, n, col):
    """Add Strassen 2x2x2 decomposition starting at column col. Returns new col."""
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[(i_0+0)*m + (j_0+0), col] = 1
    U[(i_0+1)*m + (j_0+1), col] = 1
    V[(j_0+0)*p + (k_0+0), col] = 1
    V[(j_0+1)*p + (k_0+1), col] = 1
    W[(k_0+0)*n + (i_0+0), col] = 1
    W[(k_0+1)*n + (i_0+1), col] = 1
    col += 1
    
    # Term 2: M2 = (A10+A11)B00
    U[(i_0+1)*m + (j_0+0), col] = 1
    U[(i_0+1)*m + (j_0+1), col] = 1
    V[(j_0+0)*p + (k_0+0), col] = 1
    W[(k_0+0)*n + (i_0+1), col] = 1
    W[(k_0+1)*n + (i_0+1), col] = -1
    col += 1
    
    # Term 3: M3 = A00(B01-B11)
    U[(i_0+0)*m + (j_0+0), col] = 1
    V[(j_0+0)*p + (k_0+1), col] = 1
    V[(j_0+1)*p + (k_0+1), col] = -1
    W[(k_0+1)*n + (i_0+0), col] = 1
    W[(k_0+1)*n + (i_0+1), col] = 1
    col += 1
    
    # Term 4: M4 = A11(B10-B00)
    U[(i_0+1)*m + (j_0+1), col] = 1
    V[(j_0+1)*p + (k_0+0), col] = 1
    V[(j_0+0)*p + (k_0+0), col] = -1
    W[(k_0+0)*n + (i_0+0), col] = 1
    W[(k_0+0)*n + (i_0+1), col] = 1
    col += 1
    
    # Term 5: M5 = (A00+A01)B11
    U[(i_0+0)*m + (j_0+0), col] = 1
    U[(i_0+0)*m + (j_0+1), col] = 1
    V[(j_0+1)*p + (k_0+1), col] = 1
    W[(k_0+1)*n + (i_0+0), col] = 1
    W[(k_0+0)*n + (i_0+0), col] = -1
    col += 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[(i_0+1)*m + (j_0+0), col] = 1
    U[(i_0+0)*m + (j_0+0), col] = -1
    V[(j_0+0)*p + (k_0+0), col] = 1
    V[(j_0+0)*p + (k_0+1), col] = 1
    W[(k_0+1)*n + (i_0+1), col] = 1
    col += 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[(i_0+0)*m + (j_0+1), col] = 1
    U[(i_0+1)*m + (j_0+1), col] = -1
    V[(j_0+1)*p + (k_0+0), col] = 1
    V[(j_0+1)*p + (k_0+1), col] = 1
    W[(k_0+0)*n + (i_0+0), col] = 1
    col += 1
    
    return col

def _schoolbook_block(U, V, W, i_0, j_0, k_0, n_b, m_b, p_b, m, p, n, col):
    """Add schoolbook decomposition for n_b x m_b x p_b block."""
    for u in range(n_b):
        for v in range(m_b):
            for w in range(p_b):
                U[(i_0+u)*m + (j_0+v), col] = 1
                V[(j_0+v)*p + (k_0+w), col] = 1
                W[(k_0+w)*n + (i_0+u), col] = 1
                col += 1
    return col

def _compute_rank_for_decomposition(n, m, p, i_splits, j_splits, k_splits):
    """Compute the rank for a given block decomposition."""
    i_offsets = [0]
    for s in i_splits[:-1]:
        i_offsets.append(i_offsets[-1] + s)
    j_offsets = [0]
    for s in j_splits[:-1]:
        j_offsets.append(j_offsets[-1] + s)
    k_offsets = [0]
    for s in k_splits[:-1]:
        k_offsets.append(k_offsets[-1] + s)
    
    rank = 0
    for i_idx, i_0 in enumerate(i_offsets):
        n_b = i_splits[i_idx]
        for j_idx, j_0 in enumerate(j_offsets):
            m_b = j_splits[j_idx]
            for k_idx, k_0 in enumerate(k_offsets):
                p_b = k_splits[k_idx]
                if n_b == 2 and m_b == 2 and p_b == 2:
                    rank += 7
                else:
                    rank += n_b * m_b * p_b
    return rank

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try multiple block decompositions and pick the best
    # Each decomposition is (i_splits, j_splits, k_splits)
    decompositions = []
    
    # Original: [2,2], [2,2,1], [2,2,2]
    decompositions.append(([2,2], [2,2,1], [2,2,2]))
    
    # Try [2,2], [1,2,2], [2,2,2]
    decompositions.append(([2,2], [1,2,2], [2,2,2]))
    
    # Try [2,2], [2,1,2], [2,2,2]
    decompositions.append(([2,2], [2,1,2], [2,2,2]))
    
    # Try [1,3], [2,2,1], [2,2,2]
    decompositions.append(([1,3], [2,2,1], [2,2,2]))
    
    # Try [3,1], [2,2,1], [2,2,2]
    decompositions.append(([3,1], [2,2,1], [2,2,2]))
    
    # Try [2,2], [2,2,1], [1,2,3]
    decompositions.append(([2,2], [2,2,1], [1,2,3]))
    
    # Try [2,2], [2,2,1], [3,2,1]
    decompositions.append(([2,2], [2,2,1], [3,2,1]))
    
    # Try [2,2], [1,1,3], [2,2,2]
    decompositions.append(([2,2], [1,1,3], [2,2,2]))
    
    # Try [2,2], [3,1,1], [2,2,2]
    decompositions.append(([2,2], [3,1,1], [2,2,2]))
    
    # Try [1,2,1], [2,2,1], [2,2,2]
    decompositions.append(([1,2,1], [2,2,1], [2,2,2]))
    
    # Try [2,1,1], [2,2,1], [2,2,2]
    decompositions.append(([2,1,1], [2,2,1], [2,2,2]))
    
    # Try [1,1,2], [2,2,1], [2,2,2]
    decompositions.append(([1,1,2], [2,2,1], [2,2,2]))
    
    # Try [2,2], [2,2,1], [2,1,3]
    decompositions.append(([2,2], [2,2,1], [2,1,3]))
    
    # Try [2,2], [2,2,1], [1,3,2]
    decompositions.append(([2,2], [2,2,1], [1,3,2]))
    
    # Find the best decomposition
    best_rank = float('inf')
    best_decomp = decompositions[0]
    for decomp in decompositions:
        r = _compute_rank_for_decomposition(n, m, p, decomp[0], decomp[1], decomp[2])
        if r < best_rank:
            best_rank = r
            best_decomp = decomp
    
    i_splits, j_splits, k_splits = best_decomp
    
    # Build offsets
    i_offsets = [0]
    for s in i_splits[:-1]:
        i_offsets.append(i_offsets[-1] + s)
    j_offsets = [0]
    for s in j_splits[:-1]:
        j_offsets.append(j_offsets[-1] + s)
    k_offsets = [0]
    for s in k_splits[:-1]:
        k_offsets.append(k_offsets[-1] + s)
    
    # Allocate with safe upper bound
    max_rank = 120
    U = np.zeros((n * m, max_rank), dtype=int)
    V = np.zeros((m * p, max_rank), dtype=int)
    W = np.zeros((p * n, max_rank), dtype=int)
    
    col = 0
    
    for i_idx, i_0 in enumerate(i_offsets):
        n_b = i_splits[i_idx]
        for j_idx, j_0 in enumerate(j_offsets):
            m_b = j_splits[j_idx]
            for k_idx, k_0 in enumerate(k_offsets):
                p_b = k_splits[k_idx]
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    col = _strassen_222(U, V, W, i_0, j_0, k_0, m, p, n, col)
                else:
                    col = _schoolbook_block(U, V, W, i_0, j_0, k_0, n_b, m_b, p_b, m, p, n, col)
    
    # Trim to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
