"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _DEADLINE = 120.0
    n, m, p = DIMENSIONS  # 2, 2, 14
    num_blocks = p // 2  # 7 blocks of 2x2
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each of 7 blocks, we have 7 Strassen terms
    rank = num_blocks * 7
    
    # U: shape (n*m, rank) = (4, rank)
    # Row index: i*m + j where i,j are indices into A
    # U[i*m+j, col] = coefficient of A[i,j] in the A-linear form
    
    # V: shape (m*p, rank) = (28, rank)
    # Row index: j*p + k where j,k are indices into B
    # V[j*p+k, col] = coefficient of B[j,k] in the B-linear form
    
    # W: shape (p*n, rank) = (28, rank)
    # Row index: k*n + i where k,i are indices into C (W[k*n+i])
    # W[k*n+i, col] = coefficient for C[i,k]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen terms: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict mapping (i,j) -> coeff for A[i,j]
    # B_coeffs: dict mapping (local_j, local_k) -> coeff for B[local_j, local_k] within the 2x2 block
    # C_coeffs: dict mapping (out_i, out_k) -> coeff for C[out_i, out_k]
    
    # For a 2x2 block at position b:
    # e = B[0, 2b], f = B[0, 2b+1], g = B[1, 2b], h = B[1, 2b+1]
    # a = A[0,0], b_ = A[0,1], c = A[1,0], d = A[1,1]
    
    # Strassen terms defined as:
    # term: (a_coeff, b_coeff, c_coeff, d_coeff, e_coeff, f_coeff, g_coeff, h_coeff, c11_coeff, c12_coeff, c21_coeff, c22_coeff)
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),
        # M2 = (c+d)e
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, -1),
        # M3 = a(f-h)
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1),
        # M4 = d(g-e)
        (0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0),
        # M5 = (a+b)h
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),
        # M6 = (c-a)(e+f)
        (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),
        # M7 = (b-d)(g+h)
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0),
    ]
    
    for block in range(num_blocks):
        for t_idx, term in enumerate(strassen_terms):
            col = block * 7 + t_idx
            a_c, b_c, c_c, d_c, e_c, f_c, g_c, h_c, c11_c, c12_c, c21_c, c22_c = term
            
            # A part: U[i*m+j, col]
            U[0 * m + 0][col] = a_c  # A[0,0]
            U[0 * m + 1][col] = b_c  # A[0,1]
            U[1 * m + 0][col] = c_c  # A[1,0]
            U[1 * m + 1][col] = d_c  # A[1,1]
            
            # B part: V[j*p+k, col]
            # e = B[0, 2*block], f = B[0, 2*block+1], g = B[1, 2*block], h = B[1, 2*block+1]
            V[0 * p + (2 * block)][col] = e_c
            V[0 * p + (2 * block + 1)][col] = f_c
            V[1 * p + (2 * block)][col] = g_c
            V[1 * p + (2 * block + 1)][col] = h_c
            
            # C part: W[k*n+i, col]
            # C[0, 2*block] = c11, C[0, 2*block+1] = c12, C[1, 2*block] = c21, C[1, 2*block+1] = c22
            W[(2 * block) * n + 0][col] = c11_c
            W[(2 * block + 1) * n + 0][col] = c12_c
            W[(2 * block) * n + 1][col] = c21_c
            W[(2 * block + 1) * n + 1][col] = c22_c
    
    if time.time() - _t0 > _DEADLINE - 5:
        # Safety: should not happen for this O(1) construction
        pass
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
