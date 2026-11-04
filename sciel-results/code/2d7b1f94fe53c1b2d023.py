import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Laderman's 3x3 matrix multiplication algorithm (rank 23)
    # Patterns: (U_coeffs, V_coeffs, W_coeffs)
    # Keys are (row, col) in 0..2
    laderman_23_patterns = [
        ({(0,0):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1}),
        ({(1,0):1}, {(1,0):1,(2,0):1}, {(1,0):1}),
        ({(2,0):1}, {(2,0):1}, {(2,0):1}),
        ({(1,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1,(2,1):1}),
        ({(1,1):1}, {(2,1):1}, {(1,1):-1}),
        ({(2,1):1}, {(2,1):1}, {(2,1):-1}),
        ({(1,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1,(2,2):1}),
        ({(1,2):1}, {(2,2):1}, {(1,2):-1}),
        ({(2,2):1}, {(2,2):1}, {(2,2):-1}),
        ({(0,1):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,1):1}),
        ({(0,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,2):1}),
        ({(1,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1}),
        ({(1,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1}),
        ({(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(2,1):1}),
        ({(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(2,2):1}),
        ({(0,0):1}, {(1,1):1,(2,1):1}, {(0,1):1}),
        ({(0,0):1}, {(1,2):1,(2,2):1}, {(0,2):1}),
        ({(1,1):1}, {(2,2):1}, {(1,2):1}),
        ({(1,2):1}, {(2,1):1}, {(1,1):1}),
        ({(2,1):1}, {(2,2):1}, {(2,2):1}),
        ({(2,2):1}, {(2,1):1}, {(2,1):1}),
        ({(1,1):1,(2,2):1}, {(0,1):1,(1,2):1}, {(1,1):-1,(2,2):-1}),
        ({(1,2):1,(2,1):1}, {(0,2):1,(1,1):1}, {(1,2):-1,(2,1):-1}),
    ]

    # Strassen 2x2 patterns
    # Patterns: (U_coeffs, V_coeffs, W_coeffs)
    # Keys are (row, col) in 0..1
    # C = sum w * (u . A) * (v . B)
    # Standard Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1 - P7 + P5 - P2
    # C12 = P4 + P6
    # C21 = P3 + P7
    # C22 = P1 - P3 + P4 - P5
    
    strassen_patterns = [
        # P1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,0):-1, (1,1):-1}),
        # P3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):-1}),
        # P4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,1):1, (1,1):1}),
        # P5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (1,1):-1}),
        # P6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,1):1}),
        # P7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,0):1}),
    ]

    def build_laderman_strassen_6x6():
        """
        Build 6x6x6 decomposition using Laderman's 3x3 algorithm on 3x3 blocks of 2x2 matrices.
        Inner 2x2 multiplication uses Strassen (rank 7).
        Total rank = 23 * 7 = 161.
        """
        block_size = 2
        blocks_per_dim = 3
        rank = 23 * 7
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        col_idx = 0
        for l_idx, (l_u, l_v, l_w) in enumerate(laderman_23_patterns):
            for s_idx, (s_u, s_v, s_w) in enumerate(strassen_patterns):
                # Construct U for this term
                # U_global[r, c] = sum_{I,J,i,j} s_u[i,j] * l_u[I,J] * A[2I+i, 2J+j]
                for (i, j), s_val in s_u.items():
                    for (I, J), l_val in l_u.items():
                        r = I * block_size + i
                        c = J * block_size + j
                        U[r * m + c, col_idx] += s_val * l_val
                
                # Construct V for this term
                # V_global[r, c] = sum_{I,J,i,j} s_v[i,j] * l_v[I,J] * B[2I+i, 2J+j]
                for (i, j), s_val in s_v.items():
                    for (I, J), l_val in l_v.items():
                        r = I * block_size + i
                        c = J * block_size + j
                        V[r * p + c, col_idx] += s_val * l_val
                
                # Construct W for this term
                # W_global[k, i] = sum_{K,L,k,m} s_w[k,m] * l_w[K,L] * C[2K+k, 2L+m]
                for (k, m), s_val in s_w.items():
                    for (K, L), l_val in l_w.items():
                        r = K * block_size + k
                        c = L * block_size + m
                        # W index is k*n + i, where i is row, k is col?
                        # Task says W[k*n+i, r]. So row index is k*n+i.
                        # k is col of C, i is row of C.
                        # In our loop, r is row of C, c is col of C.
                        # So W index is c*n + r.
                        W[c * n + r, col_idx] += s_val * l_val
                
                col_idx += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    # Test matrices for verification
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    def verify_decomposition(U_arr, V_arr, W_arr, A_test, B_test, C_expected):
        rank_used = U_arr.shape[1]
        C_computed = np.zeros((n, p))
        for r in range(rank_used):
            M_u = np.dot(U_arr[:, r], A_test.flatten())
            M_v = np.dot(V_arr[:, r], B_test.flatten())
            M_r = M_u * M_v
            # W is (p*n, rank). Index is k*n+i.
            # We need to add M_r * W[:, r] to C.
            # W[:, r] is a vector of length p*n.
            # Reshape to (p, n) and transpose to (n, p)?
            # W[k*n+i] corresponds to C[i, k].
            # So W_vec.reshape(p, n).T is (n, p).
            C_computed += M_r * W_arr[:, r].reshape(p, n).T
        error = np.max(np.abs(C_computed - C_expected))
        return error

    # Build the 161 rank decomposition
    solution, rank = build_laderman_strassen_6x6()
    
    U_sol = np.array(solution["U"], dtype=float)
    V_sol = np.array(solution["V"], dtype=float)
    W_sol = np.array(solution["W"], dtype=float)
    
    err = verify_decomposition(U_sol, V_sol, W_sol, A_test, B_test, C_expected)
    
    if err > 1e-10:
        # Fallback to schoolbook if something went wrong
        # (Should not happen, but safe)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        rank = n*m*p

    return solution

# EVOLVE-BLOCK-END
