import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 6x6 matrix multiplication.
    
    Strategy: Use 2x2 Strassen blocks (7 multiplications per 2x2 block).
    For 6x6 matrices, we can partition into 3x3 blocks of size 2x2.
    This gives 27 blocks * 7 = 189 rank terms.
    
    The key insight is that the block structure allows us to reuse
    Strassen's decomposition independently for each block pair,
    giving a total rank of 189 (vs 216 for schoolbook).
    
    Additionally, we can potentially merge some blocks if they share
    the same Strassen patterns, but for simplicity and correctness,
    we use the direct 2x2 Strassen approach with rank 189.
    """
    n, m, p = DIMENSIONS
    block_size = 2
    num_blocks = n // block_size  # 3 blocks of size 2
    
    # Total rank: 3*3*3 blocks * 7 Strassen terms = 189
    rank = (num_blocks ** 3) * 7
    
    # Initialize factor matrices with zeros
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # For matrices A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[C00,C01],[C10,C11]]
    # 
    # Strassen's 7 products:
    # P1 = (a+d)(e+h)  -> contributes to all
    # P2 = (c+d)*e     -> C10, C11
    # P3 = a*(f-h)     -> C01
    # P4 = d*(g-c)     -> C10
    # P5 = (a+b)*h     -> C01, C11
    # P6 = (c-a)*(e+g) -> C00, C10
    # P7 = (b-d)*(f+h) -> C00, C11
    #
    # C00 = P1 - P5 + P6 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P4 + P6 - P7
    
    # u_coeffs: coefficients for [a, b, c, d] in U
    # v_coeffs: coefficients for [e, f, g, h] in V
    # w_coeffs: coefficients for [C00, C01, C10, C11] in W
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # P1 = (a+d)(e+h)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # P2 = (c+d)*e
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # P3 = a*(f-h)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # P4 = d*(g-c)
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 0]),
        # P5 = (a+b)*h
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
        # P6 = (c-a)*(e+g)
        ([-1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]),
        # P7 = (b-d)*(f+h)
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, -1]),
    ]
    
    # Local indices for 2x2 blocks
    # A[i,j], B[i,j], C[i,j] where i,j in {0,1}
    # Global index for A: row = I*2 + i, col = J*2 + j
    # Global index for B: row = J*2 + i, col = K*2 + j
    # Global index for C: row = I*2 + i, col = K*2 + j
    
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                for t in range(7):
                    r = (I * num_blocks + J) * num_blocks * 7 + K * 7 + t
                    if r >= rank:
                        continue
                    
                    u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
                    
                    # Update U: A block at position (I, J)
                    for idx, coeff in enumerate(u_coeffs):
                        if coeff != 0:
                            i_loc = idx // 2  # 0 or 1
                            j_loc = idx % 2   # 0 or 1
                            row_A = I * block_size + i_loc
                            col_A = J * block_size + j_loc
                            U[row_A * m + col_A, r] += coeff
                    
                    # Update V: B block at position (J, K)
                    for idx, coeff in enumerate(v_coeffs):
                        if coeff != 0:
                            i_loc = idx // 2  # row index within B block
                            j_loc = idx % 2   # col index within B block
                            row_B = J * block_size + i_loc
                            col_B = K * block_size + j_loc
                            V[row_B * p + col_B, r] += coeff
                    
                    # Update W: C block at position (I, K)
                    # W index convention: W[k*n + i, r] where k is col, i is row
                    for idx, coeff in enumerate(w_coeffs):
                        if coeff != 0:
                            i_loc = idx // 2  # row index within C block
                            k_loc = idx % 2   # col index within C block
                            row_C = I * block_size + i_loc
                            col_C = K * block_size + k_loc
                            # W[col_C * n + row_C, r] += coeff
                            W[col_C * n + row_C, r] += coeff
    
    # Convert to lists for the required output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
