"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for 6x6x6 matrix multiplication rank minimization."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock limit

    # Strassen 2x2 patterns (rank 7)
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def verify_2x2_patterns(patterns):
        """Verify that the 2x2 patterns correctly compute C = A*B."""
        np.random.seed(456)
        for trial in range(5):
            A_t = np.random.randn(2, 2)
            B_t = np.random.randn(2, 2)
            C_expected = A_t @ B_t
            C_computed = np.zeros((2, 2))
            for u_c, v_c, w_c in patterns:
                mu = sum(val * A_t[r, c] for (r, c), val in u_c.items())
                mv = sum(val * B_t[r, c] for (r, c), val in v_c.items())
                for (r, c), val in w_c.items():
                    C_computed[r, c] += val * mu * mv
            if np.max(np.abs(C_computed - C_expected)) > 1e-10:
                return False
        return True

    def verify_decomposition(U_arr, V_arr, W_arr, A_test, B_test, C_expected):
        """Verify a decomposition is correct."""
        rank_used = U_arr.shape[1]
        C_computed = np.zeros((n, p))
        for r in range(rank_used):
            M_u = np.dot(U_arr[:, r], A_test.flatten())
            M_v = np.dot(V_arr[:, r], B_test.flatten())
            M_r = M_u * M_v
            C_computed += M_r * W_arr[:, r].reshape(p, n).T
        error = np.max(np.abs(C_computed - C_expected))
        return error

    def build_strassen_6x6():
        """Build the 6x6x6 decomposition using Strassen on 2x2 blocks (rank 189)."""
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7  # 27 * 7 = 189
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

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

    def build_kronecker_2x3():
        """
        Build a 6x6x6 decomposition using Kronecker product structure.
        Decompose 6x6 as 2x3 blocks. Use Strassen (rank 7) for 2x2 block mult,
        and schoolbook (rank 9) for 3x3 block mult.
        Total rank = 7 * 9 = 63.
        """
        # Strassen 2x2 patterns
        sU = np.zeros((4, 7))
        sV = np.zeros((4, 7))
        sW = np.zeros((4, 7))
        for c, (u_c, v_c, w_c) in enumerate(strassen_patterns):
            for (r, cc), val in u_c.items():
                sU[r * 2 + cc, c] = val
            for (r, cc), val in v_c.items():
                sV[r * 2 + cc, c] = val
            for (r, cc), val in w_c.items():
                sW[r * 2 + cc, c] = val

        # Schoolbook 3x3 patterns (rank 9)
        schoolbook_3x3 = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    schoolbook_3x3.append((
                        {(i, j): 1},
                        {(j, k): 1},
                        {(i, k): 1}
                    ))

        tU = np.zeros((9, 9))
        tV = np.zeros((9, 9))
        tW = np.zeros((9, 9))
        for c, (u_c, v_c, w_c) in enumerate(schoolbook_3x3):
            for (r, cc), val in u_c.items():
                tU[r * 3 + cc, c] = val
            for (r, cc), val in v_c.items():
                tV[r * 3 + cc, c] = val
            for (r, cc), val in w_c.items():
                tW[r * 3 + cc, c] = val

        # Build 6x6x6 using Kronecker product
        # 6 = 2 * 3, so blocks are 2x2 in one dim and 3x3 in another?
        # Actually, let's use 2x3 blocks: 6 = 2 * 3
        # U: (6, 6) -> split into 2x3 blocks: 2 block rows, 3 block cols
        # V: (6, 6) -> split into 3x2 blocks? No, V is (m, p) = (6, 6)
        # Let's use a simpler approach: 6 = 2 * 3, so we have 2x3 grid of blocks
        # But blocks need to be square for matrix mult. Let's use 2x2 and 3x3 separately.
        
        # Alternative: Use 6 = 2 * 3, with 2x2 Strassen for the 2-part and 3x3 schoolbook for the 3-part
        # This gives a rank of 7 * 9 = 63, but the block structure is mixed.
        
        # Let's just use the straightforward Kronecker product of Strassen(2) and schoolbook(3)
        # This requires the tensor to be decomposable as a Kronecker product, which it is
        # if we view 6x6x6 as (2x3) x (2x3) x (2x3)
        
        # Actually, the cleanest approach: use 2x2 Strassen for the "2" part and 3x3 schoolbook for the "3" part
        # The total rank is 7 * 9 = 63
        
        # U: (36, 63) = (6*6, 7*9)
        # V: (36, 63) = (6*6, 7*9)  
        # W: (36, 63) = (6*6, 7*9)
        
        rank = 7 * 9  # 63
        U = np.zeros((36, rank))
        V = np.zeros((36, rank))
        W = np.zeros((36, rank))
        
        # For each Strassen term c1 (0..6) and schoolbook term c2 (0..8):
        # U row: (i1, i2, j1, j2) where i1 in {0,1}, i2 in {0,1,2}, j1 in {0,1}, j2 in {0,1,2}
        # U index: (i1*3 + i2) * 6 + (j1*3 + j2) = i1*18 + i2*6 + j1*3 + j2
        # V row: (j1*3 + j2, k1*3 + k2) -> index: (j1*3 + j2)*6 + (k1*3 + k2)
        # W row: (k1*3 + k2, i1*3 + i2) -> index: (k1*3 + k2)*6 + (i1*3 + i2)
        
        for c1 in range(7):
            for c2 in range(9):
                col = c1 * 9 + c2
                for i1 in range(2):
                    for j1 in range(2):
                        u_val = sU[i1 * 2 + j1, c1]
                        if u_val != 0:
                            for i2 in range(3):
                                for j2 in range(3):
                                    t_val = tU[i2 * 3 + j2, c2]
                                    if t_val != 0:
                                        u_row = (i1 * 3 + i2) * 6 + (j1 * 3 + j2)
                                        U[u_row, col] += u_val * t_val
                for j1 in range(2):
                    for k1 in range(2):
                        v_val = sV[j1 * 2 + k1, c1]
                        if v_val != 0:
                            for j2 in range(3):
                                for k2 in range(3):
                                    t_val = tV[j2 * 3 + k2, c2]
                                    if t_val != 0:
                                        v_row = (j1 * 3 + j2) * 6 + (k1 * 3 + k2)
                                        V[v_row, col] += v_val * t_val
                for k1 in range(2):
                    for i1 in range(2):
                        w_val = sW[k1 * 2 + i1, c1]
                        if w_val != 0:
                            for k2 in range(3):
                                for i2 in range(3):
                                    t_val = tW[k2 * 3 + i2, c2]
                                    if t_val != 0:
                                        w_row = (k1 * 3 + k2) * 6 + (i1 * 3 + i2)
                                        W[w_row, col] += w_val * t_val
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def build_schoolbook_6x6():
        """Fallback: schoolbook (rank 216)."""
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
        return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}, rank_trivial

    # Test matrices for verification
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    # Verify base patterns
    strassen_valid = verify_2x2_patterns(strassen_patterns)

    # Try known constructions and pick the lowest-rank valid one
    best_solution = None
    best_rank = float('inf')

    # 1) Kronecker product 2x3 (rank 63) - best known
    if strassen_valid and time.time() < deadline - 20:
        kron_sol, kron_rank = build_kronecker_2x3()
        U_k = np.array(kron_sol["U"], dtype=float)
        V_k = np.array(kron_sol["V"], dtype=float)
        W_k = np.array(kron_sol["W"], dtype=float)
        err_k = verify_decomposition(U_k, V_k, W_k, A_test, B_test, C_expected)
        if err_k < 1e-10 and kron_rank < best_rank:
            best_solution = kron_sol
            best_rank = kron_rank

    # 2) Strassen-based 6x6 (rank 189)
    if strassen_valid and (best_solution is None or time.time() < deadline - 20):
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
        schoolbook_sol, schoolbook_rank = build_schoolbook_6x6()
        best_solution = schoolbook_sol
        best_rank = schoolbook_rank

    return best_solution
# EVOLVE-BLOCK-END
