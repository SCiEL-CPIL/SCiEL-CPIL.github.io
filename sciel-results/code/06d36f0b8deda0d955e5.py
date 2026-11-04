"""Optimized 6x6x6 matrix multiplication rank minimization."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    def verify_decomposition(U_arr, V_arr, W_arr, A_test, B_test, C_expected):
        """Verify a decomposition is correct using the W[k*n+i, r] convention."""
        rank_used = U_arr.shape[1]
        C_computed = np.zeros((n, p))
        for r in range(rank_used):
            mu = np.dot(U_arr[:, r], A_test.flatten())
            mv = np.dot(V_arr[:, r], B_test.flatten())
            C_computed += (mu * mv) * W_arr[:, r].reshape(p, n).T
        return np.max(np.abs(C_computed - C_expected))

    def build_laderman_6x6():
        """
        Laderman's rank-23 algorithm for 3x3 matrix multiplication,
        applied to 2x2 blocks of a 6x6 matrix. Rank = 8 * 23 = 184.
        
        C = A*B where A is 6x6, B is 6x6, C is 6x6.
        We view A, B, C as 2x2 blocks of 3x3 submatrices.
        For each block triple (I,J,K), we use Laderman's 23 products.
        """
        block_size = 3
        blocks_per_dim = 2
        rank = (blocks_per_dim ** 3) * 23  # 8 * 23 = 184
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Laderman's 23 patterns for 3x3 matrix multiplication
        # Each pattern: (U_dict, V_dict, W_dict)
        # U_dict maps (row, col) in A -> coefficient
        # V_dict maps (row, col) in B -> coefficient  
        # W_dict maps (row, col) in C -> coefficient
        # The contribution to C[i,j] is sum over r: W[r][i,j] * U[r]·A * V[r]·B
        
        laderman_patterns = [
            # M1: A[0,0] * (B[0,0] + B[1,0] + B[2,0])
            ({(0,0): 1}, {(0,0): 1, (1,0): 1, (2,0): 1}, {(0,0): 1}),
            # M2: A[1,0] * (B[1,0] + B[2,0])
            ({(1,0): 1}, {(1,0): 1, (2,0): 1}, {(1,0): 1}),
            # M3: A[2,0] * B[2,0]
            ({(2,0): 1}, {(2,0): 1}, {(2,0): 1}),
            # M4: (A[1,1] + A[2,1]) * (B[0,1] + B[1,1] + B[2,1])
            ({(1,1): 1, (2,1): 1}, {(0,1): 1, (1,1): 1, (2,1): 1}, {(1,1): 1, (2,1): 1}),
            # M5: A[1,1] * B[2,1]
            ({(1,1): 1}, {(2,1): 1}, {(1,1): -1}),
            # M6: A[2,1] * B[2,1]
            ({(2,1): 1}, {(2,1): 1}, {(2,1): -1}),
            # M7: (A[1,2] + A[2,2]) * (B[0,2] + B[1,2] + B[2,2])
            ({(1,2): 1, (2,2): 1}, {(0,2): 1, (1,2): 1, (2,2): 1}, {(1,2): 1, (2,2): 1}),
            # M8: A[1,2] * B[2,2]
            ({(1,2): 1}, {(2,2): 1}, {(1,2): -1}),
            # M9: A[2,2] * B[2,2]
            ({(2,2): 1}, {(2,2): 1}, {(2,2): -1}),
            # M10: A[0,1] * (B[0,0] + B[1,0] + B[2,0])
            ({(0,1): 1}, {(0,0): 1, (1,0): 1, (2,0): 1}, {(0,1): 1}),
            # M11: A[0,2] * (B[0,0] + B[1,0] + B[2,0])
            ({(0,2): 1}, {(0,0): 1, (1,0): 1, (2,0): 1}, {(0,2): 1}),
            # M12: A[1,1] * (B[0,1] + B[1,1] + B[2,1])
            ({(1,1): 1}, {(0,1): 1, (1,1): 1, (2,1): 1}, {(1,1): 1}),
            # M13: A[1,2] * (B[0,2] + B[1,2] + B[2,2])
            ({(1,2): 1}, {(0,2): 1, (1,2): 1, (2,2): 1}, {(1,2): 1}),
            # M14: A[2,1] * (B[0,1] + B[1,1] + B[2,1])
            ({(2,1): 1}, {(0,1): 1, (1,1): 1, (2,1): 1}, {(2,1): 1}),
            # M15: A[2,2] * (B[0,2] + B[1,2] + B[2,2])
            ({(2,2): 1}, {(0,2): 1, (1,2): 1, (2,2): 1}, {(2,2): 1}),
            # M16: A[0,0] * (B[1,1] + B[2,1])
            ({(0,0): 1}, {(1,1): 1, (2,1): 1}, {(0,1): 1}),
            # M17: A[0,0] * (B[1,2] + B[2,2])
            ({(0,0): 1}, {(1,2): 1, (2,2): 1}, {(0,2): 1}),
            # M18: A[1,1] * B[2,2]
            ({(1,1): 1}, {(2,2): 1}, {(1,2): 1}),
            # M19: A[1,2] * B[2,1]
            ({(1,2): 1}, {(2,1): 1}, {(1,1): 1}),
            # M20: A[2,1] * B[2,2]
            ({(2,1): 1}, {(2,2): 1}, {(2,2): 1}),
            # M21: A[2,2] * B[2,1]
            ({(2,2): 1}, {(2,1): 1}, {(2,1): 1}),
            # M22: (A[1,1] + A[2,2]) * (B[0,1] + B[1,2])
            ({(1,1): 1, (2,2): 1}, {(0,1): 1, (1,2): 1}, {(1,1): -1, (2,2): -1}),
            # M23: (A[1,2] + A[2,1]) * (B[0,2] + B[1,1])
            ({(1,2): 1, (2,1): 1}, {(0,2): 1, (1,1): 1}, {(1,2): -1, (2,1): -1}),
        ]
        
        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in laderman_patterns:
                        # Fill U: A block (I,J) has rows I*3..I*3+2, cols J*3..J*3+2
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val
                        
                        # Fill V: B block (J,K) has rows J*3..J*3+2, cols K*3..K*3+2
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val
                        
                        # Fill W: C block (I,K) has rows I*3..I*3+2, cols K*3..K*3+2
                        # W index is k*n + i where i is row of C, k is col of C
                        for (pi, pj), val in w_coeffs.items():
                            i = I * block_size + pi
                            k = K * block_size + pj
                            W[k * n + i, col_idx] += val
                        
                        col_idx += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def build_strassen_6x6():
        """
        Strassen's rank-7 algorithm for 2x2 matrix multiplication,
        applied to 3x3 blocks of a 6x6 matrix. Rank = 27 * 7 = 189.
        """
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7  # 27 * 7 = 189
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen's 7 patterns for 2x2 matrix multiplication
        strassen_patterns = [
            # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2: (A[1,0]+A[1,1]) * B[0,0]
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M3: A[0,0] * (B[0,1]+B[1,1])
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4: A[1,1] * (B[1,0]+B[0,0])
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5: (A[0,0]+A[0,1]) * B[1,1]
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # M6: (A[1,0]+A[0,0]) * (B[0,0]+B[0,1])
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7: (A[0,1]+A[1,1]) * (B[1,0]+B[1,1])
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val
                        
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val
                        
                        for (pi, pj), val in w_coeffs.items():
                            i = I * block_size + pi
                            k = K * block_size + pj
                            W[k * n + i, col_idx] += val
                        
                        col_idx += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def build_winograd_4x4_6x6():
        """
        Winograd's rank-34 algorithm for 4x4 matrix multiplication.
        Applied to 6x6 by padding to 8x8 (2x2 blocks of 4x4).
        Rank = 4 * 34 = 136... but we only need 6x6, so we can trim.
        Actually, let's use a different approach: apply to the 6x6 directly
        using a 4x4 core with appropriate handling.
        
        Alternative: Use Winograd 4x4 on a 4x4 subproblem and handle the rest.
        This is complex, so let's try a simpler mixed approach.
        """
        # For now, return None to skip this approach if it's too complex
        return None, None

    def build_mixed_approach():
        """
        Mixed approach: Use Laderman for some blocks and Strassen for others
        to potentially get a lower rank than either pure approach.
        
        Actually, the cleanest improvement is to verify Laderman works correctly.
        Let's just return the best of Laderman and Strassen.
        """
        return None, None

    # Test matrices for verification
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    best_solution = None
    best_rank = float('inf')

    # 1) Laderman-based 6x6 (rank 184)
    if time.time() < deadline - 20:
        laderman_sol, laderman_rank = build_laderman_6x6()
        U_l = np.array(laderman_sol["U"], dtype=float)
        V_l = np.array(laderman_sol["V"], dtype=float)
        W_l = np.array(laderman_sol["W"], dtype=float)
        err_l = verify_decomposition(U_l, V_l, W_l, A_test, B_test, C_expected)
        if err_l < 1e-10 and laderman_rank < best_rank:
            best_solution = laderman_sol
            best_rank = laderman_rank

    # 2) Strassen-based 6x6 (rank 189)
    if time.time() < deadline - 20:
        strassen_sol, strassen_rank = build_strassen_6x6()
        U_s = np.array(strassen_sol["U"], dtype=float)
        V_s = np.array(strassen_sol["V"], dtype=float)
        W_s = np.array(strassen_sol["W"], dtype=float)
        err_s = verify_decomposition(U_s, V_s, W_s, A_test, B_test, C_expected)
        if err_s < 1e-10 and strassen_rank < best_rank:
            best_solution = strassen_sol
            best_rank = strassen_rank

    # 3) Fallback: schoolbook (rank 216)
    if best_solution is None:
        rank_trivial = n * m * p  # 216
        U_t = np.zeros((n * m, rank_trivial))
        V_t = np.zeros((m * p, rank_trivial))
        W_t = np.zeros((p * n, rank_trivial))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_t[i * m + j, idx] = 1
                    V_t[j * p + k, idx] = 1
                    W_t[k * n + i, idx] = 1
                    idx += 1
        best_solution = {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
        best_rank = rank_trivial

    return best_solution
# EVOLVE-BLOCK-END
