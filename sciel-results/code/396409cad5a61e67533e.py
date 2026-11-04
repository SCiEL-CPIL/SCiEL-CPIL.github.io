import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0

    # Rank 98 Strassen-based decomposition
    rank = 98

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Precompute Strassen base matrices
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    # c=0
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    # c=1
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    # c=2
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    # c=3
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    # c=4
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    # c=5
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    # c=6
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # A_specs: How to form A_eff for each P_k
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]

    # B_specs: How to form B_eff for each P_k
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)],   # P6: B10 + B11
    ]

    # C_specs: How P_k contributes to C blocks
    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(1, 2), (-1, 3)],  # P1: C10 - C11
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)],           # P6: C00
    ]

    # Precompute U contributions for each (k_p, m_s) - independent of half
    U_contribs = np.zeros((7, 7, n * m))
    for k_p in range(7):
        for m_s in range(7):
            for coeff_a, block_idx in A_specs[k_p]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        gr = bi * 2 + r
                        gc = bj * 2 + c
                        a_idx = gr * 4 + gc
                        eff = r * 2 + c
                        U_contribs[k_p, m_s, a_idx] += coeff_a * U_2[eff, m_s]

    # Precompute V contributions for each (k_p, m_s, half)
    V_contribs = np.zeros((7, 7, 2, m * p))
    for k_p in range(7):
        for m_s in range(7):
            for half in range(2):
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            gk = gc + half * 4
                            b_idx = gr * 8 + gk
                            eff = r * 2 + c
                            V_contribs[k_p, m_s, half, b_idx] += coeff_b * V_2[eff, m_s]

    # Precompute W contributions for each (k_p, m_s, half)
    W_contribs = np.zeros((7, 7, 2, p * n))
    for k_p in range(7):
        for m_s in range(7):
            for half in range(2):
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bi * 2 + r
                            gk = bj * 2 + c + half * 4
                            w_idx = gk * 4 + gi
                            eff = r * 2 + c
                            W_contribs[k_p, m_s, half, w_idx] += coeff_c * W_2[eff, m_s]

    # Build the decomposition using precomputed contributions
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s
                U[:, col] = U_contribs[k_p, m_s, :]
                V[:, col] = V_contribs[k_p, m_s, half, :]
                W[:, col] = W_contribs[k_p, m_s, half, :]

    # Quick self-check: verify tensor decomposition identity
    if time.time() < deadline - 5.0:
        # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # For matrix mult: T[i, j, k] = 1 if i==j==k (in flattened form)
        # Actually: T[a_idx, b_idx, w_idx] = sum_r U[a_idx,r]*V[b_idx,r]*W[w_idx,r]
        # should equal 1 if a_idx corresponds to A[i,m], b_idx to B[m,k], w_idx to C[k,i]
        # and i,k are consistent, else 0
        # Let's do a spot check
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        # Compute C from decomposition
        C_decomp = np.zeros((p, n))
        for r in range(rank):
            C_decomp += np.outer(W[:, r], np.outer(U[:, r], V[:, r]).reshape(n, m, p).transpose(1, 2, 0).reshape(p * n, 1))[:, 0] if False else 0
        # Simpler: C[k, i] = sum_m A[i, m] * B[m, k] = sum_r U[i*4+m, r] * V[m*8+k, r] * W[k*4+i, r]
        C_decomp2 = np.zeros((n, p))
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            # u_col has shape (16,), v_col (32,), w_col (32,)
            # u_col[a_idx] where a_idx = i*4+m
            # v_col[b_idx] where b_idx = m*8+k  
            # w_col[w_idx] where w_idx = k*4+i
            u_mat = u_col.reshape(n, m)  # (4,4)
            v_mat = v_col.reshape(m, p)  # (4,8)
            w_mat = w_col.reshape(p, n)  # (8,4)
            C_decomp2 += (u_mat @ v_mat) * w_mat.T  # This isn't quite right
        # Actually the decomposition is: C[k,i] = sum_r W[k*4+i, r] * (sum_m U[i*4+m,r] * V[m*8+k,r])
        # Let's just trust the construction and do a direct numerical check
        C_num = np.zeros((n, p))
        for r in range(rank):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            # u_r[i*4+m] * v_r[m*8+k] summed over m gives a 4x8 matrix
            # then multiplied by w_r[k*4+i]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for mm in range(m):
                        s += u_r[i * 4 + mm] * v_r[mm * 8 + k]
                    C_num[i, k] += w_r[k * 4 + i] * s
        err = np.max(np.abs(C_num - C_expected))
        if err > 1e-10:
            # If check fails, return schoolbook as fallback
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb))
            V_fb = np.zeros((m * p, rank_fb))
            W_fb = np.zeros((p * n, rank_fb))
            for i in range(n):
                for mm in range(m):
                    for k in range(p):
                        r_idx = i * m * p + mm * p + k
                        U_fb[i * 4 + mm, r_idx] = 1
                        V_fb[mm * 8 + k, r_idx] = 1
                        W_fb[k * 4 + i, r_idx] = 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strassen 2x2x2 base matrices (4x7)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    # c=0
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    # c=1
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    # c=2
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    # c=3
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    # c=4
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    # c=5
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    # c=6
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # A_specs: How to form A_eff for each P_k
    # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]

    # B_specs: How to form B_eff for each P_k
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)],   # P6: B10 + B11
    ]

    # C_specs: How P_k contributes to C blocks
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(1, 2), (-1, 3)],  # P1: C10 - C11
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)],           # P6: C00
    ]

    # Build the decomposition
    for half in range(2):  # 0 for left half, 1 for right half
        for k_p in range(7):  # P_k index
            for m_s in range(7):  # S_m index
                col = half * 49 + k_p * 7 + m_s

                # --- U (Input A) ---
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            a_idx = gr * 4 + gc
                            eff = r * 2 + c
                            U[a_idx, col] += coeff_a * U_2[eff, m_s]

                # --- V (Input B) ---
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            gk = gc + half * 4
                            b_idx = gr * 8 + gk
                            eff = r * 2 + c
                            V[b_idx, col] += coeff_b * V_2[eff, m_s]

                # --- W (Output C) ---
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bi * 2 + r
                            gk = bj * 2 + c + half * 4
                            w_idx = gk * 4 + gi
                            eff = r * 2 + c
                            W[w_idx, col] += coeff_c * W_2[eff, m_s]

                if time.time() > deadline:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
