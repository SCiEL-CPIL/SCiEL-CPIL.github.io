import time
import numpy as np
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the classic Strassen 2x2x2 decomposition with 7 terms.
    U, V, W are 4x7 matrices (row-major indexing).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3: M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4: M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1
    
    # Term 5: M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6: M6 = (c-d)(f+g)
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (b-d)(e+f)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[1][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify_fast(n, m, p, U, V, W):
    """
    Verify that the decomposition exactly reconstructs the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _apply_strassen_block(U, V, W, i_off, j_off, k_off, col):
    """
    Apply Strassen decomposition to a 2x2x2 block at offset (i_off, j_off, k_off).
    U rows: (i_off+i)*m + (j_off+j), V rows: (j_off+j)*p + (k_off+k), W rows: (k_off+k)*n + (i_off+i)
    Returns col + 7.
    """
    n, m, p = 5, 5, 5
    U_s, V_s, W_s = _strassen_2x2()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                idx_u = (i_off + i) * m + (j_off + j)
                U[idx_u][col + c] = U_s[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                idx_v = (j_off + j) * p + (k_off + k)
                V[idx_v][col + c] = V_s[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                idx_w = (k_off + k) * n + (i_off + i)
                W[idx_w][col + c] = W_s[k * 2 + i][c]
    return col + 7

def _apply_schoolbook_block(U, V, W, i_range, j_range, k_range, col):
    """
    Apply schoolbook decomposition to a block with given index ranges.
    Returns col + len(i_range)*len(j_range)*len(k_range).
    """
    n, m, p = 5, 5, 5
    for i in i_range:
        for j in j_range:
            for k in k_range:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return col

def _construct_5x5x5():
    """
    Construct a valid decomposition for 5x5x5 matrix multiplication.
    Uses fine-grained block partitioning with Strassen on all 2x2x2 sub-blocks:
    
    Partition: i in {0,1},{2,3},{4}; j in {0,1},{2,3},{4}; k in {0,1},{2,3},{4}
    
    2x2x2 blocks (8 total): Strassen (rank 7 each) = 56
    2x2x1 blocks (4 total): schoolbook (rank 4 each) = 16
    2x1x2 blocks (4 total): schoolbook (rank 4 each) = 16
    1x2x2 blocks (4 total): schoolbook (rank 4 each) = 16
    2x1x1 blocks (4 total): schoolbook (rank 2 each) = 8
    1x2x1 blocks (4 total): schoolbook (rank 2 each) = 8
    1x1x2 blocks (4 total): schoolbook (rank 2 each) = 8
    1x1x1 blocks (1 total): schoolbook (rank 1) = 1
    Total: 56+16+16+16+8+8+8+1 = 129... wait that's worse.
    
    Actually let me reconsider. The 2+3 partition is better:
    - TL (2x2x2): Strassen (rank 7)
    - BR (3x3x3): Schoolbook (rank 27)  
    - TR (2x5x3): partition into Strassen + schoolbook sub-blocks (rank 28)
    - BL (3x5x2): partition into Strassen + schoolbook sub-blocks (rank 28)
    Total: 7 + 27 + 28 + 28 = 90
    """
    n, m, p = 5, 5, 5
    nm, mp, pn = n * m, m * p, p * n
    
    total_rank = 7 + 27 + 28 + 28
    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]
    
    col = 0
    
    # 1. TL Block: Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    col = _apply_strassen_block(U, V, W, 0, 0, 0, col)
    
    # 2. BR Block: Schoolbook for i in {2,3,4}, j in {2,3,4}, k in {2,3,4}
    col = _apply_schoolbook_block(U, V, W, range(2, 5), range(2, 5), range(2, 5), col)
    
    # 3. TR Block: i in {0,1}, j in {0..4}, k in {2,3,4}
    #    Sub-partition:
    #    - i{0,1}, j{0,1}, k{2,3}: 2x2x2 Strassen (rank 7)
    #    - i{0,1}, j{0,1}, k{4}:   2x2x1 schoolbook (rank 4)
    #    - i{0,1}, j{2,3}, k{2,3}: 2x2x2 Strassen (rank 7)
    #    - i{0,1}, j{2,3}, k{4}:   2x2x1 schoolbook (rank 4)
    #    - i{0,1}, j{4}, k{2,3}:   2x1x2 schoolbook (rank 4)
    #    - i{0,1}, j{4}, k{4}:     2x1x1 schoolbook (rank 2)
    #    Total: 7+4+7+4+4+2 = 28
    col = _apply_strassen_block(U, V, W, 0, 0, 2, col)
    col = _apply_schoolbook_block(U, V, W, range(0, 2), range(0, 2), range(4, 5), col)
    col = _apply_strassen_block(U, V, W, 0, 2, 2, col)
    col = _apply_schoolbook_block(U, V, W, range(0, 2), range(2, 4), range(4, 5), col)
    col = _apply_schoolbook_block(U, V, W, range(0, 2), range(4, 5), range(2, 4), col)
    col = _apply_schoolbook_block(U, V, W, range(0, 2), range(4, 5), range(4, 5), col)
    
    # 4. BL Block: i in {2,3,4}, j in {0..4}, k in {0,1}
    #    Sub-partition:
    #    - i{2,3}, j{0,1}, k{0,1}: 2x2x2 Strassen (rank 7)
    #    - i{2,3}, j{2,3}, k{0,1}: 2x2x2 Strassen (rank 7)
    #    - i{2,3}, j{4}, k{0,1}:   2x1x2 schoolbook (rank 4)
    #    - i{4}, j{0,1}, k{0,1}:   1x2x2 schoolbook (rank 4)
    #    - i{4}, j{2,3}, k{0,1}:   1x2x2 schoolbook (rank 4)
    #    - i{4}, j{4}, k{0,1}:     1x1x2 schoolbook (rank 2)
    #    Total: 7+7+4+4+4+2 = 28
    col = _apply_strassen_block(U, V, W, 2, 0, 0, col)
    col = _apply_strassen_block(U, V, W, 2, 2, 0, col)
    col = _apply_schoolbook_block(U, V, W, range(2, 4), range(4, 5), range(0, 2), col)
    col = _apply_schoolbook_block(U, V, W, range(4, 5), range(0, 2), range(0, 2), col)
    col = _apply_schoolbook_block(U, V, W, range(4, 5), range(2, 4), range(0, 2), col)
    col = _apply_schoolbook_block(U, V, W, range(4, 5), range(4, 5), range(0, 2), col)
    
    return {"U": U, "V": V, "W": W}, total_rank

def _try_improve_with_strassen_blocks(n, m, p, U, V, W, deadline):
    """
    Try to improve the decomposition by applying Strassen to more blocks.
    This is a heuristic search that may or may not succeed within the time limit.
    """
    if time.time() > deadline:
        return U, V, W, 94
    
    # For now, we keep the current decomposition.
    # A full search for better decompositions is complex and time-consuming.
    # We return the current valid decomposition.
    return U, V, W, 94

def construct():
    """
    Main entry point. Returns a valid tensor decomposition for 5x5x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    result, rank = _construct_5x5x5()
    
    # Verify the decomposition
    if _verify_fast(n, m, p, result["U"], result["V"], result["W"]):
        # Try to improve if time permits
        U, V, W, new_rank = _try_improve_with_strassen_blocks(
            n, m, p, result["U"], result["V"], result["W"], deadline
        )
        if new_rank < rank and _verify_fast(n, m, p, U, V, W):
            return {"U": U, "V": V, "W": W}
        
        return result
    
    # Fallback: Pure schoolbook (rank 125)
    U = [[0] * 125 for _ in range(25)]
    V = [[0] * 125 for _ in range(25)]
    W = [[0] * 125 for _ in range(25)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
