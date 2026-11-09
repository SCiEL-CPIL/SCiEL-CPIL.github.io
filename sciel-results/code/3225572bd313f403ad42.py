"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    # We use the 3x3 block partitioning where each 2x2 block pair is multiplied
    # using the rank-7 Strassen algorithm.
    # This yields a rank of 3^3 * 7 = 189.
    
    # If we could find a rank-23 decomposition for 3x3 matrix multiplication,
    # we could use a 2x2 block Strassen structure with 3x3 blocks, yielding
    # a rank of 7 * 23 = 161.
    # However, the rank-23 coefficients are complex. We stick to the robust 189.
    
    rank = 189
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    # Strassen terms for 2x2 matrix multiplication
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = M1 - M4 + M6 + M7
    # C12 = M2 + M3
    # C21 = M4 + M5
    # C22 = M1 - M2 + M3 + M5
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We map these to the factor matrices.
    # The tensor T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # For each block (i, j, k) in the 3x3 grid:
    # The block A[i, k] is at rows 2*i..2*i+1, cols 2*k..2*k+1?
    # Wait, the block structure is:
    # A is 6x6. Split into 3x3 blocks of 2x2.
    # Block (i, k) of A is A[2*i:2*i+2, 2*k:2*k+2].
    # Block (k, j) of B is B[2*k:2*k+2, 2*j:2*j+2].
    # Block (i, j) of C is C[2*i:2*i+2, 2*j:2*j+2].
    
    # The Strassen algorithm operates on the 3x3 block matrix.
    # Let A_b be the 3x3 block matrix of A.
    # A_b[i, k] is the 2x2 block A[2*i:2*i+2, 2*k:2*k+2].
    # Similarly for B_b[k, j] and C_b[i, j].
    
    # The 7 multiplications M_t are linear combinations of blocks.
    # Each M_t is a 2x2 matrix.
    # We decompose each 2x2 matrix multiplication using Strassen (rank 7).
    
    # This results in 27 * 7 = 189 terms.
    
    strassen_terms = [
        # M1
        {'U': [(0, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1), (1, 1, 1)]},
        # M2
        {'U': [(1, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1)], 'W': [(1, 0, 1), (1, 1, -1)]},
        # M3
        {'U': [(0, 0, 1)], 'V': [(0, 1, 1), (1, 1, -1)], 'W': [(0, 1, 1), (1, 1, 1)]},
        # M4
        {'U': [(1, 1, 1)], 'V': [(1, 0, 1), (0, 0, -1)], 'W': [(0, 0, 1), (1, 0, 1)]},
        # M5
        {'U': [(0, 0, 1), (0, 1, 1)], 'V': [(1, 1, 1)], 'W': [(0, 0, -1), (0, 1, 1)]},
        # M6
        {'U': [(1, 0, 1), (0, 0, -1)], 'V': [(0, 0, 1), (0, 1, 1)], 'W': [(1, 1, 1)]},
        # M7
        {'U': [(0, 1, 1), (1, 1, -1)], 'V': [(1, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1)]}
    ]
    
    # The block indices for A are (i, k), for B are (k, j), for C are (i, j).
    # i, j, k range from 0 to 2.
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for t in range(7):
                    col_idx = (i * 9 + j * 3 + k) * 7 + t
                    term = strassen_terms[t]
                    
                    # U factors: A block (i, k)
                    # The term 'U' specifies which sub-blocks of A[i, k] are used.
                    # A[i, k] is at rows 2*i..2*i+1, cols 2*k..2*k+1.
                    # The sub-block (lr, lc) within A[i, k] is at:
                    # row = 2*i + lr, col = 2*k + lc
                    for lr, lc, coeff in term['U']:
                        u_row = (2 * i + lr) * m + (2 * k + lc)
                        U[u_row, col_idx] += coeff
                        
                    # V factors: B block (k, j)
                    # B[k, j] is at rows 2*k..2*k+1, cols 2*j..2*j+1.
                    # The sub-block (lr, lc) within B[k, j] is at:
                    # row = 2*k + lr, col = 2*j + lc
                    for lr, lc, coeff in term['V']:
                        v_row = (2 * k + lr) * p + (2 * j + lc)
                        V[v_row, col_idx] += coeff
                        
                    # W factors: C block (i, j)
                    # C[i, j] is at rows 2*i..2*i+1, cols 2*j..2*j+1.
                    # The sub-block (lr, lc) within C[i, j] is at:
                    # row = 2*i + lr, col = 2*j + lc
                    # W index is k*n + i. Here the "k" in W[k*n+i] corresponds to the column of C.
                    # So W row index = (2*j + lc) * n + (2*i + lr)
                    for lr, lc, coeff in term['W']:
                        w_row = (2 * j + lc) * n + (2 * i + lr)
                        W[w_row, col_idx] += coeff

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
