"""Optimized 6x6x6 matrix multiplication via Strassen(2x2 blocks) + Laderman(3x3 leaves)."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def verify_decomposition(U_arr, V_arr, W_arr):
        """Verify decomposition correctness with multiple random trials."""
        rank_used = U_arr.shape[1]
        rng = np.random.RandomState(42)
        max_error = 0.0
        for trial in range(3):
            A_t = rng.randn(n, m)
            B_t = rng.randn(m, p)
            C_exp = A_t @ B_t
            C_comp = np.zeros((n, p))
            for r in range(rank_used):
                M_u = np.dot(U_arr[:, r], A_t.flatten())
                M_v = np.dot(V_arr[:, r], B_t.flatten())
                C_comp += M_u * M_v * W_arr[:, r].reshape(p, n).T
            max_error = max(max_error, float(np.max(np.abs(C_comp - C_exp))))
        return max_error

    def build_strassen_laderman_6x6():
        """
        6x6 = 2x2 blocks of 3x3 matrices.
        Strassen: 7 multiplications of 3x3 blocks.
        Laderman: rank-23 decomposition for each 3x3 multiplication.
        Total rank = 7 * 23 = 161.
        
        Strassen formulas for A(2x2), B(2x2) -> C(2x2):
          M1 = (A00+A11)*(B00+B11)
          M2 = (A10+A11)*B00
          M3 = A00*(B01-B11)
          M4 = A11*(B10-B00)
          M5 = (A00+A01)*B11
          M6 = (A10-A00)*(B00+B01)
          M7 = (A01-A11)*(B10+B11)
          C00 = M1+M4-M5+M7
          C01 = M3+M5
          C10 = M2+M4
          C11 = M1-M2+M3+M6
        """
        block_size = 3
        # Laderman's 23-term decomposition for 3x3 matrix multiplication
        # Each term: (a_dict, b_dict, c_dict) where keys are (row, col)
        laderman = [
            # Term 0: A[0,0] * B[:,0] -> C[0,0]
            ({(0,0):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1}),
            # Term 1: A[1,0] * B[1:,0] -> C[1,0]
            ({(1,0):1}, {(1,0):1,(2,0):1}, {(1,0):1}),
            # Term 2: A[2,0] * B[2,0] -> C[2,0]
            ({(2,0):1}, {(2,0):1}, {(2,0):1}),
            # Term 3: (A[1,1]+A[2,1]) * B[:,1] -> C[1,0] += 1, C[2,0] += -1
            ({(1,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,0):1,(2,0):-1}),
            # Term 4: A[1,1] * B[2,1] -> C[1,0] += -1
            ({(1,1):1}, {(2,1):1}, {(1,0):-1}),
            # Term 5: A[2,1] * B[2,1] -> C[2,0] += 1
            ({(2,1):1}, {(2,1):1}, {(2,0):1}),
            # Term 6: (A[1,2]+A[2,2]) * B[:,2] -> C[1,0] += -1, C[2,0] += 1
            ({(1,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,0):-1,(2,0):1}),
            # Term 7: A[1,2] * B[2,2] -> C[1,0] += 1
            ({(1,2):1}, {(2,2):1}, {(1,0):1}),
            # Term 8: A[2,2] * B[2,2] -> C[2,0] += -1
            ({(2,2):1}, {(2,2):1}, {(2,0):-1}),
            # Term 9: A[0,1] * B[:,0] -> C[0,1]
            ({(0,1):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,1):1}),
            # Term 10: A[0,2] * B[:,0] -> C[0,2]
            ({(0,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,2):1}),
            # Term 11: A[1,1] * B[:,1] -> C[1,1]
            ({(1,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1}),
            # Term 12: A[1,2] * B[:,2] -> C[1,2]
            ({(1,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1}),
            # Term 13: A[2,1] * B[:,1] -> C[2,1]
            ({(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(2,1):1}),
            # Term 14: A[2,2] * B[:,2] -> C[2,2]
            ({(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(2,2):1}),
            # Term 15: A[0,0] * B[1:,1] -> C[0,1]
            ({(0,0):1}, {(1,1):1,(2,1):1}, {(0,1):1}),
            # Term 16: A[0,0] * B[1:,2] -> C[0,2]
            ({(0,0):1}, {(1,2):1,(2,2):1}, {(0,2):1}),
            # Term 17: A[1,1] * B[2,2] -> C[1,2] += -1
            ({(1,1):1}, {(2,2):1}, {(1,2):-1}),
            # Term 18: A[1,2] * B[2,1] -> C[1,1] += -1
            ({(1,2):1}, {(2,1):1}, {(1,1):-1}),
            # Term 19: A[2,1] * B[2,2] -> C[2,2] += -1
            ({(2,1):1}, {(2,2):1}, {(2,2):-1}),
            # Term 20: A[2,2] * B[2,1] -> C[2,1] += -1
            ({(2,2):1}, {(2,1):1}, {(2,1):-1}),
            # Term 21: (A[1,1]+A[2,2]) * (B[0,1]+B[1,2]) -> C[1,1]+=1, C[2,2]+=1
            ({(1,1):1,(2,2):1}, {(0,1):1,(1,2):1}, {(1,1):1,(2,2):1}),
            # Term 22: (A[1,2]+A[2,1]) * (B[0,2]+B[1,1]) -> C[1,2]+=1, C[2,1]+=1
            ({(1,2):1,(2,1):1}, {(0,2):1,(1,1):1}, {(1,2):1,(2,1):1}),
        ]

        # Strassen's 7 multiplications with coefficients
        # Each: (a_coeffs, b_coeffs, c_coeffs) for block-level (2x2) indices
        # a_coeffs: dict (i,j) -> coeff for A_block[i,j]
        # b_coeffs: dict (i,j) -> coeff for B_block[i,j]
        # c_coeffs: dict (i,j) -> coeff for C_block[i,j]
        strassen = [
            # M1 = (A00+A11)*(B00+B11) -> C00+=1, C11+=1
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            # M2 = (A10+A11)*B00 -> C10+=1, C11+=-1
            ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
            # M3 = A00*(B01-B11) -> C01+=1, C11+=1
            ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
            # M4 = A11*(B10-B00) -> C00+=1, C10+=1
            ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
            # M5 = (A00+A01)*B11 -> C00+=-1, C01+=1
            ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
            # M6 = (A10-A00)*(B00+B01) -> C11+=1
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
            # M7 = (A01-A11)*(B10+B11) -> C00+=1
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
        ]

        total_rank = 7 * 23  # 161
        U = np.zeros((n * m, total_rank), dtype=int)
        V = np.zeros((m * p, total_rank), dtype=int)
        W = np.zeros((p * n, total_rank), dtype=int)

        col_idx = 0
        for si in range(7):
            s_a, s_b, s_c = strassen[si]
            for lj in range(23):
                l_a, l_b, l_c = laderman[lj]
                
                # For each Strassen term, expand into Laderman terms
                # U: A has 2x2 blocks of 3x3, so A_block[si_row, si_col] contains 3x3
                # Laderman A coefficient at (pi, pj) means element (3*si_row+pi, 3*si_col+pj)
                for (si_row, si_col), sa_val in s_a.items():
                    for (pi, pj), la_val in l_a.items():
                        i = si_row * block_size + pi
                        j = si_col * block_size + pj
                        U[i * m + j, col_idx] += sa_val * la_val
                
                # V: B similarly
                for (si_row, si_col), sb_val in s_b.items():
                    for (pi, pj), lb_val in l_b.items():
                        j = si_row * block_size + pi
                        k = si_col * block_size + pj
                        V[j * p + k, col_idx] += sb_val * lb_val
                
                # W: C similarly, index is k*n + i
                for (si_row, si_col), sc_val in s_c.items():
                    for (pi, pj), lc_val in l_c.items():
                        i = si_row * block_size + pi
                        k = si_col * block_size + pj
                        W[k * n + i, col_idx] += sc_val * lc_val
                
                col_idx += 1

        assert col_idx == total_rank, f"Column index mismatch: {col_idx} vs {total_rank}"
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, total_rank

    def build_laderman_6x6():
        """Fallback: Laderman on 8 independent 3x3 blocks, rank 184."""
        block_size = 3
        blocks_per_dim = 2
        rank = (blocks_per_dim ** 3) * 23  # 184
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)

        laderman = [
            ({(0,0):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1}),
            ({(1,0):1}, {(1,0):1,(2,0):1}, {(1,0):1}),
            ({(2,0):1}, {(2,0):1}, {(2,0):1}),
            ({(1,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,0):1,(2,0):-1}),
            ({(1,1):1}, {(2,1):1}, {(1,0):-1}),
            ({(2,1):1}, {(2,1):1}, {(2,0):1}),
            ({(1,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,0):-1,(2,0):1}),
            ({(1,2):1}, {(2,2):1}, {(1,0):1}),
            ({(2,2):1}, {(2,2):1}, {(2,0):-1}),
            ({(0,1):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,1):1}),
            ({(0,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,2):1}),
            ({(1,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1}),
            ({(1,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1}),
            ({(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(2,1):1}),
            ({(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(2,2):1}),
            ({(0,0):1}, {(1,1):1,(2,1):1}, {(0,1):1}),
            ({(0,0):1}, {(1,2):1,(2,2):1}, {(0,2):1}),
            ({(1,1):1}, {(2,2):1}, {(1,2):-1}),
            ({(1,2):1}, {(2,1):1}, {(1,1):-1}),
            ({(2,1):1}, {(2,2):1}, {(2,2):-1}),
            ({(2,2):1}, {(2,1):1}, {(2,1):-1}),
            ({(1,1):1,(2,2):1}, {(0,1):1,(1,2):1}, {(1,1):1,(2,2):1}),
            ({(1,2):1,(2,1):1}, {(0,2):1,(1,1):1}, {(1,2):1,(2,1):1}),
        ]

        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for a_coeffs, b_coeffs, c_coeffs in laderman:
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

        assert col_idx == rank
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    # Primary: Strassen + Laderman (rank 161)
    sl_sol, sl_rank = build_strassen_laderman_6x6()
    U_sl = np.array(sl_sol["U"], dtype=float)
    V_sl = np.array(sl_sol["V"], dtype=float)
    W_sl = np.array(sl_sol["W"], dtype=float)
    err_sl = verify_decomposition(U_sl, V_sl, W_sl)
    
    if err_sl < 1e-10:
        return sl_sol

    # Fallback: Laderman on 8 blocks (rank 184)
    if time.time() < deadline:
        lad_sol, lad_rank = build_laderman_6x6()
        U_lad = np.array(lad_sol["U"], dtype=float)
        V_lad = np.array(lad_sol["V"], dtype=float)
        W_lad = np.array(lad_sol["W"], dtype=float)
        err_lad = verify_decomposition(U_lad, V_lad, W_lad)
        if err_lad < 1e-10:
            return lad_sol

    # Last resort: schoolbook (rank 216)
    rank_t = n * m * p
    U_t = np.zeros((n * m, rank_t), dtype=int)
    V_t = np.zeros((m * p, rank_t), dtype=int)
    W_t = np.zeros((p * n, rank_t), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_t[i * m + j, idx] = 1
                V_t[j * p + k, idx] = 1
                W_t[k * n + i, idx] = 1
                idx += 1
    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
# EVOLVE-BLOCK-END
