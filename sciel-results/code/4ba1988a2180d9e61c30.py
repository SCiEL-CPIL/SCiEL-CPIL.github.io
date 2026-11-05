"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

import time
from itertools import product

# EVOLVE-BLOCK-START
def _partitions_of_size(n, max_block=2):
    """Generate all partitions of n into parts of size 1 or max_block."""
    if n == 0:
        yield []
        return
    if n >= 1:
        for rest in _partitions_of_size(n - 1, max_block):
            yield [1] + rest
    if n >= max_block:
        for rest in _partitions_of_size(n - max_block, max_block):
            yield [max_block] + rest

def _block_rank(a, b, c):
    """Minimum rank for multiplying a×b by b×c matrix, for a,b,c in {1,2}."""
    # Rank table for small block sizes
    rank_table = {
        (1,1,1): 1,
        (1,1,2): 2,
        (1,2,1): 2,
        (2,1,1): 2,
        (1,2,2): 4,
        (2,1,2): 4,
        (2,2,1): 4,
        (2,2,2): 7,
    }
    return rank_table.get((a,b,c), a*b*c)

def _construct_block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """Construct the tensor decomposition using the given block partitions."""
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Map from block index to actual indices
    n_offsets = [0]
    for size in n_parts[:-1]:
        n_offsets.append(n_offsets[-1] + size)
    m_offsets = [0]
    for size in m_parts[:-1]:
        m_offsets.append(m_offsets[-1] + size)
    p_offsets = [0]
    for size in p_parts[:-1]:
        p_offsets.append(p_offsets[-1] + size)
    
    for ni_idx, ni_size in enumerate(n_parts):
        for mj_idx, mj_size in enumerate(m_parts):
            for pk_idx, pk_size in enumerate(p_parts):
                r = _block_rank(ni_size, mj_size, pk_size)
                base_rank = rank
                rank += r
                
                # For each block, we need to add the decomposition
                # For schoolbook (1,1,1): single term
                # For Strassen (2,2,2): 7 terms
                # For mixed: use appropriate decomposition
                
                if ni_size == 1 and mj_size == 1 and pk_size == 1:
                    # Single term
                    i = n_offsets[ni_idx]
                    j = m_offsets[mj_idx]
                    k = p_offsets[pk_idx]
                    U_rows.append([i * m + j, base_rank, 1])
                    V_rows.append([j * p + k, base_rank, 1])
                    W_rows.append([k * n + i, base_rank, 1])
                
                elif ni_size == 2 and mj_size == 2 and pk_size == 2:
                    # Strassen: 7 terms
                    i0 = n_offsets[ni_idx]
                    j0 = m_offsets[mj_idx]
                    k0 = p_offsets[pk_idx]
                    
                    # Strassen formulas for 2x2 x 2x2 -> 2x2
                    # M1 = (A00 + A11)(B00 + B11)
                    # M2 = (A10 + A11)B00
                    # M3 = A00(B10 - B11)
                    # M4 = A11(B01 - B00)
                    # M5 = (A00 + A10)B11
                    # M6 = (A01 - A00)(B00 + B10)
                    # M7 = (A11 - A10)(B01 + B11)
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    
                    # Term 1: M1 = (A00+A11)(B00+B11)
                    # U: A00 and A11 get +1, V: B00 and B11 get +1
                    # W: C00 gets +1, C11 gets +1
                    col = base_rank
                    U_rows.append([(i0+0)*m + (j0+0), col, 1])
                    U_rows.append([(i0+1)*m + (j0+1), col, 1])
                    V_rows.append([(j0+0)*p + (k0+0), col, 1])
                    V_rows.append([(j0+1)*p + (k0+1), col, 1])
                    W_rows.append([(k0+0)*n + (i0+0), col, 1])
                    W_rows.append([(k0+1)*n + (i0+1), col, 1])
                    
                    # Term 2: M2 = (A10+A11)B00
                    col = base_rank + 1
                    U_rows.append([(i0+1)*m + (j0+0), col, 1])
                    U_rows.append([(i0+1)*m + (j0+1), col, 1])
                    V_rows.append([(j0+0)*p + (k0+0), col, 1])
                    W_rows.append([(k0+0)*n + (i0+1), col, 1])
                    
                    # Term 3: M3 = A00(B10-B11)
                    col = base_rank + 2
                    U_rows.append([(i0+0)*m + (j0+0), col, 1])
                    V_rows.append([(j0+1)*p + (k0+0), col, 1])
                    V_rows.append([(j0+1)*p + (k0+1), col, -1])
                    W_rows.append([(k0+1)*n + (i0+0), col, 1])
                    
                    # Term 4: M4 = A11(B01-B00)
                    col = base_rank + 3
                    U_rows.append([(i0+1)*m + (j0+1), col, 1])
                    V_rows.append([(j0+0)*p + (k0+1), col, 1])
                    V_rows.append([(j0+0)*p + (k0+0), col, -1])
                    W_rows.append([(k0+0)*n + (i0+0), col, 1])
                    W_rows.append([(k0+1)*n + (i0+1), col, 1])
                    
                    # Term 5: M5 = (A00+A10)B11
                    col = base_rank + 4
                    U_rows.append([(i0+0)*m + (j0+0), col, 1])
                    U_rows.append([(i0+1)*m + (j0+0), col, 1])
                    V_rows.append([(j0+1)*p + (k0+1), col, 1])
                    W_rows.append([(k0+0)*n + (i0+0), col, -1])
                    W_rows.append([(k0+1)*n + (i0+1), col, 1])
                    
                    # Term 6: M6 = (A01-A00)(B00+B10)
                    col = base_rank + 5
                    U_rows.append([(i0+0)*m + (j0+1), col, 1])
                    U_rows.append([(i0+0)*m + (j0+0), col, -1])
                    V_rows.append([(j0+0)*p + (k0+0), col, 1])
                    V_rows.append([(j0+1)*p + (k0+0), col, 1])
                    W_rows.append([(k0+1)*n + (i0+1), col, 1])
                    
                    # Term 7: M7 = (A11-A10)(B01+B11)
                    col = base_rank + 6
                    U_rows.append([(i0+1)*m + (j0+1), col, 1])
                    U_rows.append([(i0+1)*m + (j0+0), col, -1])
                    V_rows.append([(j0+0)*p + (k0+1), col, 1])
                    V_rows.append([(j0+1)*p + (k0+1), col, 1])
                    W_rows.append([(k0+1)*n + (i0+0), col, 1])
                    W_rows.append([(k0+0)*n + (i0+1), col, 1])
                
                elif ni_size == 1 and mj_size == 2 and pk_size == 2:
                    # 1x2 x 2x2 -> 1x2, rank 4
                    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
                    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
                    i0 = n_offsets[ni_idx]
                    j0 = m_offsets[mj_idx]
                    k0 = p_offsets[pk_idx]
                    
                    # Term 1: A[0,0]*B[0,0] -> C[0,0]
                    col = base_rank
                    U_rows.append([i0*m + j0, col, 1])
                    V_rows.append([j0*p + k0, col, 1])
                    W_rows.append([k0*n + i0, col, 1])
                    
                    # Term 2: A[0,1]*B[1,0] -> C[0,0]
                    col = base_rank + 1
                    U_rows.append([i0*m + (j0+1), col, 1])
                    V_rows.append([(j0+1)*p + k0, col, 1])
                    W_rows.append([k0*n + i0, col, 1])
                    
                    # Term 3: A[0,0]*B[0,1] -> C[0,1]
                    col = base_rank + 2
                    U_rows.append([i0*m + j0, col, 1])
                    V_rows.append([j0*p + (k0+1), col, 1])
                    W_rows.append([(k0+1)*n + i0, col, 1])
                    
                    # Term 4: A[0,1]*B[1,1] -> C[0,1]
                    col = base_rank + 3
                    U_rows.append([i0*m + (j0+1), col, 1])
                    V_rows.append([(j0+1)*p + (k0+1), col, 1])
                    W_rows.append([(k0+1)*n + i0, col, 1])
                
                elif ni_size == 2 and mj_size == 1 and pk_size == 2:
                    # 2x1 x 1x2 -> 2x2, rank 4
                    # C[i,0] = A[i,0]*B[0,0]
                    # C[i,1] = A[i,0]*B[0,1]
                    i0 = n_offsets[ni_idx]
                    j0 = m_offsets[mj_idx]
                    k0 = p_offsets[pk_idx]
                    
                    # Term 1: A[0,0]*B[0,0] -> C[0,0]
                    col = base_rank
                    U_rows.append([(i0+0)*m + j0, col, 1])
                    V_rows.append([j0*p + k0, col, 1])
                    W_rows.append([k0*n + i0, col, 1])
                    
                    # Term 2: A[1,0]*B[0,0] -> C[1,0]
                    col = base_rank + 1
                    U_rows.append([(i0+1)*m + j0, col, 1])
                    V_rows.append([j0*p + k0, col, 1])
                    W_rows.append([k0*n + (i0+1), col, 1])
                    
                    # Term 3: A[0,0]*B[0,1] -> C[0,1]
                    col = base_rank + 2
                    U_rows.append([(i0+0)*m + j0, col, 1])
                    V_rows.append([j0*p + (k0+1), col, 1])
                    W_rows.append([(k0+1)*n + i0, col, 1])
                    
                    # Term 4: A[1,0]*B[0,1] -> C[1,1]
                    col = base_rank + 3
                    U_rows.append([(i0+1)*m + j0, col, 1])
                    V_rows.append([j0*p + (k0+1), col, 1])
                    W_rows.append([(k0+1)*n + (i0+1), col, 1])
                
                elif ni_size == 2 and mj_size == 2 and pk_size == 1:
                    # 2x2 x 2x1 -> 2x1, rank 4
                    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
                    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
                    i0 = n_offsets[ni_idx]
                    j0 = m_offsets[mj_idx]
                    k0 = p_offsets[pk_idx]
                    
                    # Term 1: A[0,0]*B[0,0] -> C[0,0]
                    col = base_rank
                    U_rows.append([(i0+0)*m + j0, col, 1])
                    V_rows.append([j0*p + k0, col, 1])
                    W_rows.append([k0*n + i0, col, 1])
                    
                    # Term 2: A[0,1]*B[1,0] -> C[0,0]
                    col = base_rank + 1
                    U_rows.append([(i0+0)*m + (j0+1), col, 1])
                    V_rows.append([(j0+1)*p + k0, col, 1])
                    W_rows.append([k0*n + i0, col, 1])
                    
                    # Term 3: A[1,0]*B[0,0] -> C[1,0]
                    col = base_rank + 2
                    U_rows.append([(i0+1)*m + j0, col, 1])
                    V_rows.append([j0*p + k0, col, 1])
                    W_rows.append([k0*n + (i0+1), col, 1])
                    
                    # Term 4: A[1,1]*B[1,0] -> C[1,0]
                    col = base_rank + 3
                    U_rows.append([(i0+1)*m + (j0+1), col, 1])
                    V_rows.append([(j0+1)*p + k0, col, 1])
                    W_rows.append([k0*n + (i0+1), col, 1])
                
                elif ni_size == 1 and mj_size == 1 and pk_size == 2:
                    # 1x1 x 1x2 -> 1x2, rank 2
                    i0 = n_offsets[ni_idx]
                    j0 = m_offsets[mj_idx]
                    k0 = p_offsets[pk_idx]
                    
                    col = base_rank
                    U_rows.append([i0*m + j0, col, 1])
                    V_rows.append([j0*p + k0, col, 1])
                    W_rows.append([k0*n + i0, col, 1])
                    
                    col = base_rank + 1
                    U_rows.append([i0*m + j0, col, 1])
                    V_rows.append([j0*p + (k0+1), col, 1])
                    W_rows.append([(k0+1)*n + i0, col, 1])
                
                elif ni_size == 1 and mj_size == 2 and pk_size == 1:
                    # 1x2 x 2x1 -> 1x1, rank 2
                    i0 = n_offsets[ni_idx]
                    j0 = m_offsets[mj_idx]
                    k0 = p_offsets[pk_idx]
                    
                    col = base_rank
                    U_rows.append([i0*m + j0, col, 1])
                    V_rows.append([j0*p + k0, col, 1])
                    W_rows.append([k0*n + i0, col, 1])
                    
                    col = base_rank + 1
                    U_rows.append([i0*m + (j0+1), col, 1])
                    V_rows.append([(j0+1)*p + k0, col, 1])
                    W_rows.append([k0*n + i0, col, 1])
                
                elif ni_size == 2 and mj_size == 1 and pk_size == 1:
                    # 2x1 x 1x1 -> 2x1, rank 2
                    i0 = n_offsets[ni_idx]
                    j0 = m_offsets[mj_idx]
                    k0 = p_offsets[pk_idx]
                    
                    col = base_rank
                    U_rows.append([(i0+0)*m + j0, col, 1])
                    V_rows.append([j0*p + k0, col, 1])
                    W_rows.append([k0*n + i0, col, 1])
                    
                    col = base_rank + 1
                    U_rows.append([(i0+1)*m + j0, col, 1])
                    V_rows.append([j0*p + k0, col, 1])
                    W_rows.append([k0*n + (i0+1), col, 1])
    
    # Build the matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for row, col, val in U_rows:
        U[row][col] = val
    for row, col, val in V_rows:
        V[row][col] = val
    for row, col, val in W_rows:
        W[row][col] = val
    
    return {"U": U, "V": V, "W": W}, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_rank = n * m * p  # Schoolbook
    best_result = None
    
    # Generate all partitions
    n_parts_list = list(_partitions_of_size(n))
    m_parts_list = list(_partitions_of_size(m))
    p_parts_list = list(_partitions_of_size(p))
    
    # Try all combinations
    for n_parts in n_parts_list:
        for m_parts in m_parts_list:
            for p_parts in p_parts_list:
                if time.time() > deadline:
                    break
                
                # Calculate total rank
                total_rank = 0
                for ni, mj, pk in product(n_parts, m_parts, p_parts):
                    total_rank += _block_rank(ni, mj, pk)
                
                if total_rank < best_rank:
                    result, actual_rank = _construct_block_decomposition(n, m, p, n_parts, m_parts, p_parts)
                    if actual_rank == total_rank:
                        best_rank = total_rank
                        best_result = result
    
    if best_result is None:
        # Fallback to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
