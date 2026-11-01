import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x8 matrix.
    
    The multiplication is C = A @ B where A is 2x2, B is 2x8, C is 2x8.
    We use Strassen's algorithm on each 2x2 block of B (4 blocks),
    giving rank 7*4 = 28, which is minimal.
    
    Returns:
        dict with keys 'U', 'V', 'W' - lists of lists representing
        factor matrices of shapes (4, 28), (16, 28), (16, 28)
    """
    n, m, p = 2, 2, 8
    num_blocks = p // m  # 4 blocks of size 2x2
    rank_per_block = 7   # Strassen rank for 2x2x2
    rank = num_blocks * rank_per_block  # 28
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 16 x 28
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B21 - B22)
    # P4 = A22(B11 - B12)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    #
    # Result:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # U coefficients: for each Strassen product, the linear combination of A entries
    # A is indexed as A[i,j] with i in {0,1}, j in {0,1}
    # U index: i*m + j = i*2 + j
    strassen_U = [
        [(0, 1), (3, 1)],       # P1: A11 + A22 -> indices 0, 3
        [(2, 1), (3, 1)],       # P2: A21 + A22 -> indices 2, 3
        [(0, 1)],               # P3: A11 -> index 0
        [(3, 1)],               # P4: A22 -> index 3
        [(0, 1), (1, 1)],       # P5: A11 + A12 -> indices 0, 1
        [(2, 1), (0, -1)],      # P6: A21 - A11 -> indices 2, 0
        [(1, 1), (3, -1)]       # P7: A12 - A22 -> indices 1, 3
    ]
    
    # V coefficients: for each Strassen product, the linear combination of B entries
    # B is indexed as B[j,k] with j in {0,1}, k in {0,1} (local)
    # V index: j*p + k_global where k_global = block_start + k_local
    strassen_V = [
        [(0, 0, 1), (1, 1, 1)],       # P1: B11 + B22
        [(0, 0, 1)],                  # P2: B11
        [(1, 0, 1), (1, 1, -1)],      # P3: B21 - B22
        [(0, 0, 1), (0, 1, -1)],      # P4: B11 - B12
        [(1, 1, 1)],                  # P5: B22
        [(0, 0, 1), (0, 1, 1)],       # P6: B11 + B12
        [(1, 0, 1), (1, 1, 1)]        # P7: B21 + B22
    ]
    
    # W coefficients: for each Strassen product, contribution to C entries
    # C is indexed as C[i,k] with i in {0,1}, k in {0,1} (local)
    # W index: k_global*n + i where k_global = block_start + k_local
    strassen_W = [
        [(0, 0, 1), (1, 1, 1)],       # P1: C11 += 1, C22 += 1
        [(1, 0, 1), (1, 1, -1)],      # P2: C21 += 1, C22 += -1
        [(0, 1, 1), (1, 1, 1)],       # P3: C12 += 1, C22 += 1
        [(0, 0, 1), (1, 0, 1)],       # P4: C11 += 1, C21 += 1
        [(0, 0, -1), (0, 1, 1)],      # P5: C11 += -1, C12 += 1
        [(1, 1, 1)],                  # P6: C22 += 1
        [(0, 0, 1)]                   # P7: C11 += 1
    ]
    
    for t in range(num_blocks):
        block_start = t * m  # start column index in B and C
        for s in range(rank_per_block):
            r = t * rank_per_block + s  # global rank index
            
            # Fill U
            for u_idx, coeff in strassen_U[s]:
                U[u_idx][r] = coeff
            
            # Fill V
            for b_row, b_col_local, coeff in strassen_V[s]:
                b_col_global = block_start + b_col_local
                v_idx = b_row * p + b_col_global
                V[v_idx][r] = coeff
            
            # Fill W
            for c_row, c_col_local, coeff in strassen_W[s]:
                c_col_global = block_start + c_col_local
                w_idx = c_col_global * n + c_row
                W[w_idx][r] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
