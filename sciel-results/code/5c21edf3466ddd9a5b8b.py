"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that U,V,W form a valid tensor decomposition of matmul(n,m,p)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    
    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True

def _construct_schoolbook(n=4, m=4, p=5):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return the 7 rank-1 tensors for 2x2x2 matrix multiplication.
    Each tensor is (u, v, w) where u is 4-dim, v is 4-dim, w is 4-dim.
    Indices: u[i*m+j], v[j*p+k], w[k*n+i] with n=m=p=2.
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(([1,0, 0,1], [1,0, 0,1], [1,0, 0,1]))
    # M2 = (A10+A11)B00
    terms.append(([0,0, 1,1], [1,0, 0,0], [0,1, 0,0]))
    # M3 = A00(B10-B11)
    terms.append(([1,0, 0,0], [0,0, 1,-1], [0,0, 1,0]))
    # M4 = A11(B00-B01)
    terms.append(([0,0, 0,1], [1,-1, 0,0], [0,0, 0,1]))
    # M5 = (A00+A01)B11
    terms.append(([1,1, 0,0], [0,0, 0,1], [0,0, 1,0]))
    # M6 = (A10-A00)(B00+B01)
    terms.append(([-1,0, 1,0], [1,1, 0,0], [0,1, 0,0]))
    # M7 = (A01-A11)(B10+B11)
    terms.append(([0,1, 0,-1], [0,0, 1,1], [0,0, 0,1]))
    return terms

def _embed_strassen_block(i0, j0, k0, n=4, m=4, p=5):
    """
    Embed a 2x2x2 Strassen block at position (i0, j0, k0).
    Returns (U_block, V_block, W_block) as 2D arrays of shape (n*m, 7), (m*p, 7), (p*n, 7).
    """
    terms = _strassen_2x2x2()
    
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    for r, (u, v, w) in enumerate(terms):
        for di in range(2):
            for dj in range(2):
                i = i0 + di
                j = j0 + dj
                if i < n and j < m:
                    U[i*m + j, r] = u[di*2 + dj]
        for dj in range(2):
            for dk in range(2):
                j = j0 + dj
                k = k0 + dk
                if j < m and k < p:
                    V[j*p + k, r] = v[dj*2 + dk]
        for dk in range(2):
            for di in range(2):
                k = k0 + dk
                i = i0 + di
                if k < p and i < n:
                    W[k*n + i, r] = w[dk*2 + di]
    
    return U, V, W

def _blocks_overlap(pos1, pos2):
    """Check if two 2x2x2 blocks at given positions overlap."""
    i1, j1, k1 = pos1
    i2, j2, k2 = pos2
    # Blocks overlap if their index ranges intersect in all 3 dimensions
    if abs(i1 - i2) < 2 and abs(j1 - j2) < 2 and abs(k1 - k2) < 2:
        return True
    return False

def _build_decomposition(block_positions, n=4, m=4, p=5):
    """
    Build a decomposition from a set of Strassen blocks + schoolbook fill.
    Returns (result_dict, rank) or (None, None) if invalid.
    """
    covered = set()
    for (i0, j0, k0) in block_positions:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i, j, k = i0 + di, j0 + dj, k0 + dk
                    if i < n and j < m and k < p:
                        covered.add((i, j, k))
    
    num_blocks = len(block_positions)
    strassen_rank = num_blocks * 7
    schoolbook_rank = n * m * p - len(covered)
    total_rank = strassen_rank + schoolbook_rank
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for (i0, j0, k0) in block_positions:
        U_blk, V_blk, W_blk = _embed_strassen_block(i0, j0, k0, n, m, p)
        U[:, col:col+7] = U_blk
        V[:, col:col+7] = V_blk
        W[:, col:col+7] = W_blk
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if _validate(result["U"], result["V"], result["W"], n, m, p):
        return result, total_rank
    return None, None

def _find_best_strassen_combo(n=4, m=4, p=5, deadline=None):
    """
    Search for the best combination of non-overlapping 2x2x2 Strassen blocks.
    Returns (best_result, best_rank).
    """
    all_positions = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_positions.append((i0, j0, k0))
    
    best_result = None
    best_rank = n * m * p
    
    # Pre-check: which pairs are non-overlapping
    non_overlap_pairs = []
    for idx1 in range(len(all_positions)):
        for idx2 in range(idx1 + 1, len(all_positions)):
            if not _blocks_overlap(all_positions[idx1], all_positions[idx2]):
                non_overlap_pairs.append((idx1, idx2))
    
    # Try all single blocks (should all be valid, but verify)
    for pos in all_positions:
        if deadline is not None and time.time() > deadline - 5:
            break
        result, rank = _build_decomposition([pos], n, m, p)
        if result is not None and rank < best_rank:
            best_rank = rank
            best_result = result
    
    # Try all non-overlapping pairs
    for idx1, idx2 in non_overlap_pairs:
        if deadline is not None and time.time() > deadline - 5:
            break
        result, rank = _build_decomposition([all_positions[idx1], all_positions[idx2]], n, m, p)
        if result is not None and rank < best_rank:
            best_rank = rank
            best_result = result
    
    # Try triples: only if we still have time
    if deadline is None or time.time() < deadline - 10:
        for idx1 in range(len(all_positions)):
            for idx2 in range(idx1 + 1, len(all_positions)):
                if _blocks_overlap(all_positions[idx1], all_positions[idx2]):
                    continue
                for idx3 in range(idx2 + 1, len(all_positions)):
                    if _blocks_overlap(all_positions[idx1], all_positions[idx3]) or \
                       _blocks_overlap(all_positions[idx2], all_positions[idx3]):
                        continue
                    if deadline is not None and time.time() > deadline - 5:
                        break
                    result, rank = _build_decomposition(
                        [all_positions[idx1], all_positions[idx2], all_positions[idx3]], n, m, p)
                    if result is not None and rank < best_rank:
                        best_rank = rank
                        best_result = result
    
    if best_result is None:
        best_result = _construct_schoolbook(n, m, p)
        best_rank = n * m * p
    
    return best_result, best_rank

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 80
    
    best_result = _construct_schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    try:
        strassen_result, strassen_rank = _find_best_strassen_combo(n, m, p, deadline)
        if strassen_rank < best_rank:
            best_result = strassen_result
            best_rank = strassen_rank
    except Exception:
        pass
    
    return best_result

# EVOLVE-BLOCK-END
