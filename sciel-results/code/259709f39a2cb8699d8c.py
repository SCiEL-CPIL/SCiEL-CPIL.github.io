"""Matrix multiplication tensor decomposition for 3x4 x 4x8."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start = time.time()
    deadline = start + 140.0

    n, m, p = 3, 4, 8
    rank_total = 88

    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)

    col = 0

    # Part 1: i=0, schoolbook (32 terms)
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1

    # Part 2: i=1,2, Strassen on 2x4x8 (56 terms)
    # Strassen's 7 terms for 2x2x2 matrix multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # For our 2x4x8: A is 2x4, B is 4x8
    # Partition A into 2x2 blocks: A[0:2, 0:2], A[0:2, 2:4]
    # Partition B into 2x2 blocks: B[0:2, 0:2], B[0:2, 2:4], B[2:4, 0:2], B[2:4, 2:4]
    #
    # We have 2 j-blocks and 4 k-blocks, so 8 sub-problems of 2x2x2
    # Each uses 7 Strassen terms

    # Strassen coefficient tables for 2x2x2
    # A indices: (local_i, local_j) where local_i in {0,1}, local_j in {0,1}
    # B indices: (local_j, local_k) where local_j in {0,1}, local_k in {0,1}
    # C indices: (local_i, local_k) where local_i in {0,1}, local_k in {0,1}
    
    # For each Strassen term t, we need coefficients for A, B, C
    # A coefficients: dict mapping (li, lj) -> coeff
    # B coefficients: dict mapping (lj, lk) -> coeff  
    # C coefficients: dict mapping (li, lk) -> coeff
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # M1 = (A11+A22)(B11+B22)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}),
        # M2 = (A21+A22)B11
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3 = A11(B21-B22)
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,0):1}),
        # M4 = A22(B12-B11)
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(1,1):1}),
        # M5 = (A11+A12)B22
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1}),
        # M6 = (A21-A11)(B11+B12)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (1,1):1}),
        # M7 = (A12-A22)(B21+B22)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):1, (1,1):1}),
    ]

    # For the 2x4x8 subproblem (i=1,2), we partition:
    # j dimension (4) into 2 blocks of 2: j_block 0 -> j=0,1; j_block 1 -> j=2,3
    # k dimension (8) into 4 blocks of 2: k_block 0 -> k=0,1; k_block 1 -> k=2,3; etc.
    
    for j_block in range(2):
        for k_block in range(4):
            for t in range(7):
                a_coeffs, b_coeffs, c_coeffs = strassen_terms[t]
                
                # Add A coefficients to U
                for (li, lj), coeff in a_coeffs.items():
                    global_i = 1 + li  # i=1 or 2
                    global_j = j_block * 2 + lj
                    U[global_i * m + global_j, col] += coeff
                
                # Add B coefficients to V
                for (lj, lk), coeff in b_coeffs.items():
                    global_j = j_block * 2 + lj
                    global_k = k_block * 2 + lk
                    V[global_j * p + global_k, col] += coeff
                
                # Add C coefficients to W
                for (li, lk), coeff in c_coeffs.items():
                    global_i = 1 + li
                    global_k = k_block * 2 + lk
                    W[global_k * n + global_i, col] += coeff
                
                col += 1

    assert col == rank_total, f"Expected {rank_total} terms, got {col}"

    # Verify the decomposition
    def verify():
        # Check that for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    total = np.sum(u_val * v_val * w_val)
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    if verify():
        # Convert to list of lists as required
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback: schoolbook (should not happen)
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school), dtype=float)
        V_s = np.zeros((m * p, rank_school), dtype=float)
        W_s = np.zeros((p * n, rank_school), dtype=float)
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col_s] = 1.0
                    V_s[j * p + k, col_s] = 1.0
                    W_s[k * n + i, col_s] = 1.0
                    col_s += 1
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }
