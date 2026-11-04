import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Laderman's 3x3 matrix multiplication algorithm (rank 23)
    # Patterns are (u_coeffs, v_coeffs, w_coeffs)
    # Keys are (row, col)
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
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Note: The standard Strassen decomposition has rank 7.
    # We represent each term as (u_coeffs, v_coeffs, w_coeffs) for the 2x2 blocks.
    # u_coeffs: coefficients for A blocks (I, J)
    # v_coeffs: coefficients for B blocks (J, K)
    # w_coeffs: coefficients for C blocks (I, K)
    
    strassen_7_patterns = [
        # M1 = (A00+A11)(B00+B11)
        # U: A00, A11 have coeff 1
        # V: B00, B11 have coeff 1
        # W: C00, C11 have coeff 1 (part of C11 and C22)
        # Actually, let's define the 7 products explicitly as rank-1 contributions to the block matrix product.
        # Product 1: (A00+A11) * (B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # Product 2: (A10+A11) * B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # Product 3: A00 * (B10-B00)
        ({(0,0):1}, {(1,0):-1, (0,0):1}, {(0,1):1, (1,1):1}), # Wait, standard Strassen:
        # Let's use the standard 7 terms from literature to be safe.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
    ]
    
    # Let's re-verify the Strassen patterns to ensure correctness.
    # The above list in the current program was:
    # ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
    # ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}), # M2 -> C11=-1, C21=1?
    # Let's check C11 = M1+M4-M5+M7.
    # If we just use the patterns from the previous program which were verified to work for 2x2, we can reuse them.
    # The previous program had:
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def verify_decomposition(U_arr, V_arr, W_arr, A_test, B_test, C_expected):
        rank_used = U_arr.shape[1]
        C_computed = np.zeros((n, p))
        for r in range(rank_used):
            M_u = np.dot(U_arr[:, r], A_test.flatten())
            M_v = np.dot(V_arr[:, r], B_test.flatten())
            M_r = M_u * M_v
            C_computed += M_r * W_arr[:, r].reshape(p, n).T
        error = np.max(np.abs(C_computed - C_expected))
        return error

    def build_strassen_laderman_6x6():
        """
        Build 6x6x6 using Strassen (rank 7) on 3x3 blocks, each computed by Laderman (rank 23).
        Total rank = 7 * 23 = 161.
        """
        rank = 7 * 23
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # Strassen patterns are for 2x2 blocks.
        # Laderman patterns are for 3x3 blocks.
        
        # Indices for Strassen blocks: I, J, K in {0, 1}
        # Indices for Laderman blocks: r, c, s in {0, 1, 2}
        
        for t_idx, (u_s_coeffs, v_s_coeffs, w_s_coeffs) in enumerate(strassen_patterns):
            for s_idx, (u_l_coeffs, v_l_coeffs, w_l_coeffs) in enumerate(laderman_23_patterns):
                rank_idx = t_idx * 23 + s_idx
                
                # Populate U
                # u_s_coeffs maps (I, J) -> val_s
                # u_l_coeffs maps (r, c) -> val_l
                # Global A index: row = 3*I + r, col = 3*J + c
                for (I, J), val_s in u_s_coeffs.items():
                    for (r, c), val_l in u_l_coeffs.items():
                        row = I * 3 + r
                        col = J * 3 + c
                        U[row * m + col, rank_idx] += val_s * val_l
                        
                # Populate V
                # v_s_coeffs maps (J, K) -> val_s
                # v_l_coeffs maps (r, c) -> val_l (here r is row of B block, c is col of B block)
                # Global B index: row = 3*J + r, col = 3*K + c
                for (J, K), val_s in v_s_coeffs.items():
                    for (r, c), val_l in v_l_coeffs.items():
                        row = J * 3 + r
                        col = K * 3 + c
                        V[row * p + col, rank_idx] += val_s * val_l
                        
                # Populate W
                # w_s_coeffs maps (I, K) -> val_s
                # w_l_coeffs maps (r, c) -> val_l (here r is row of C block, c is col of C block)
                # Global C index: row = 3*I + r, col = 3*K + c
                # Trace convention: W[k*n+i, r] where i is row, k is col.
                # So index is (3*K + c) * n + (3*I + r)
                for (I, K), val_s in w_s_coeffs.items():
                    for (r, c), val_l in w_l_coeffs.items():
                        i = I * 3 + r
                        k = K * 3 + c
                        W[k * n + i, rank_idx] += val_s * val_l

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def build_laderman_6x6():
        """Build 6x6x6 decomposition using Laderman's 3x3 algorithm on 2x2 blocks. Rank = 8*23 = 184."""
        block_size = 3
        blocks_per_dim = 2
        rank = (blocks_per_dim ** 3) * 23
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for a_coeffs, b_coeffs, c_coeffs in laderman_23_patterns:
                        for (pi, pj), val in a_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val
                        for (pi, pj), val in b_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val
                        for (pi, pj), val in c_coeffs.items():
                            i = I * block_size + pi
                            k = K * block_size + pj
                            W[k * n + i, col_idx] += val
                        col_idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def build_schoolbook_6x6():
        """Schoolbook rank 216."""
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    # Test matrices for verification
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    best_solution = None
    best_rank = float('inf')

    # 1) Strassen-Laderman hybrid (rank 161)
    if time.time() < deadline - 30:
        sol, r = build_strassen_laderman_6x6()
        U_l = np.array(sol["U"], dtype=float)
        V_l = np.array(sol["V"], dtype=float)
        W_l = np.array(sol["W"], dtype=float)
        err = verify_decomposition(U_l, V_l, W_l, A_test, B_test, C_expected)
        if err < 1e-10 and r < best_rank:
            best_solution = sol
            best_rank = r

    # 2) Laderman-based 6x6 (rank 184)
    if (best_solution is None or time.time() < deadline - 30) and best_rank > 184:
        laderman_sol, laderman_rank = build_laderman_6x6()
        U_l = np.array(laderman_sol["U"], dtype=float)
        V_l = np.array(laderman_sol["V"], dtype=float)
        W_l = np.array(laderman_sol["W"], dtype=float)
        err_l = verify_decomposition(U_l, V_l, W_l, A_test, B_test, C_expected)
        if err_l < 1e-10 and laderman_rank < best_rank:
            best_solution = laderman_sol
            best_rank = laderman_rank

    # 3) Fallback: schoolbook (rank 216)
    if best_solution is None:
        schoolbook_sol, schoolbook_rank = build_schoolbook_6x6()
        best_solution = schoolbook_sol
        best_rank = schoolbook_rank

    return best_solution

# EVOLVE-BLOCK-END
