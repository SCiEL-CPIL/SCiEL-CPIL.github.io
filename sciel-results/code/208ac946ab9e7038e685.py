"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-50 decomposition for 2x4 * 4x7 matrix multiplication.
    
    Strategy: Split the 4 middle dimension into 2+2.
    For each 2x2 * 2x7 product, split the 7 columns into 2+2+2+1.
    Use Strassen (rank 7) for each 2x2*2x2 block, schoolbook (rank 4) for 2x2*2x1.
    Total: 2 * (7+7+7+4) = 50.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0] * rank for _ in range(p * n)]  # 14 x 50
    
    col = 0
    
    # A is 2x4: A[i][j] where i in {0,1}, j in {0,1,2,3}
    # B is 4x7: B[j][k] where j in {0,1,2,3}, k in {0,...,6}
    # C is 2x7: C[i][k] = sum_j A[i][j] * B[j][k]
    # W index: W[k*n + i] = W[k*2 + i]
    
    # Split middle dimension j into two halves: j in {0,1} (top) and j in {2,3} (bottom)
    for half in range(2):
        j_offset = half * 2  # 0 or 2
        
        # Split the 7 columns of B_half into groups: 2, 2, 2, 1
        col_groups = [(0, 2), (2, 4), (4, 6), (6, 7)]
        
        for k_start, k_end in col_groups:
            k_width = k_end - k_start
            
            if k_width == 2:
                # 2x2 * 2x2: Strassen, 7 multiplications
                strassen_terms = [
                    # (a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11)
                    (1, 0, -1, 0, 0, 0, 1, 1, 1, 0, 0, 1),      # M1
                    (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0),       # M2
                    (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 0),      # M3
                    (0, 0, 0, 1, 1, -1, 0, 0, 1, 0, 1, 0),      # M4
                    (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),      # M5
                    (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),      # M6
                    (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 1),      # M7
                ]
                
                for (a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11) in strassen_terms:
                    c_idx = col
                    col += 1
                    
                    # U: A[i][j_offset + a]
                    if a00 != 0:
                        U[0 * 4 + (j_offset + 0)][c_idx] += a00
                    if a01 != 0:
                        U[0 * 4 + (j_offset + 1)][c_idx] += a01
                    if a10 != 0:
                        U[1 * 4 + (j_offset + 0)][c_idx] += a10
                    if a11 != 0:
                        U[1 * 4 + (j_offset + 1)][c_idx] += a11
                    
                    # V: B[j_offset + b][k_start + c]
                    if b00 != 0:
                        V[(j_offset + 0) * 7 + (k_start + 0)][c_idx] += b00
                    if b01 != 0:
                        V[(j_offset + 0) * 7 + (k_start + 1)][c_idx] += b01
                    if b10 != 0:
                        V[(j_offset + 1) * 7 + (k_start + 0)][c_idx] += b10
                    if b11 != 0:
                        V[(j_offset + 1) * 7 + (k_start + 1)][c_idx] += b11
                    
                    # W: C[i][k_start + c] -> W[(k_start+c)*2 + i]
                    if c00 != 0:
                        W[(k_start + 0) * 2 + 0][c_idx] += c00
                    if c01 != 0:
                        W[(k_start + 1) * 2 + 0][c_idx] += c01
                    if c10 != 0:
                        W[(k_start + 0) * 2 + 1][c_idx] += c10
                    if c11 != 0:
                        W[(k_start + 1) * 2 + 1][c_idx] += c11
            
            elif k_width == 1:
                # 2x2 * 2x1: schoolbook, 4 multiplications
                # Each term: U has one A element, V has one B element, W has one C element
                # Term (i, b): U[A[i][j_offset+b]] = 1, V[B[j_offset+b][k_start]] = 1, W[C[i][k_start]] = 1
                for i in range(2):
                    for b_idx in range(2):
                        c_idx = col
                        col += 1
                        
                        # U: A[i][j_offset + b_idx]
                        U[i * 4 + (j_offset + b_idx)][c_idx] = 1
                        
                        # V: B[j_offset + b_idx][k_start]
                        V[(j_offset + b_idx) * 7 + k_start][c_idx] = 1
                        
                        # W: C[i][k_start] -> W[k_start*2 + i]
                        W[k_start * 2 + i][c_idx] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
