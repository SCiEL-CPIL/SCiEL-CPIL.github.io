import numpy as np
import time

def construct():
    """
    Returns a rank-98 tensor decomposition for 4x4 * 4x8 matrix multiplication.
    The decomposition is built by applying the known rank-49 4x4x4 Strassen
    decomposition independently to the two 4x4 blocks of the 4x8 right factor.
    """
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 4, 4, 8
    rank = 98

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # --- 2x2x2 Strassen decomposition (rank 7) ---
    # U_2[i, k], V_2[j, k], W_2[r, k] for the 2x2x2 tensor
    U_2 = np.zeros((4, 7), dtype=float)
    V_2 = np.zeros((4, 7), dtype=float)
    W_2 = np.zeros((4, 7), dtype=float)

    # k=0: (A00+A11)(B00+B11) -> C00, C11
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    # k=1: (A10+A11)B00 -> C10 - C11
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    # k=2: A00(B01-B11) -> C01 + C11
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    # k=3: A11(B10-B00) -> C00 + C10
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    # k=4: (A00+A01)B11 -> -C00 + C01
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    # k=5: (A10-A00)(B00+B01) -> C11
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    # k=6: (A01-A11)(B10+B11) -> C00
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # --- 4x4x4 Strassen (rank 49) ---
    # For each of the 7 top-level products (k_p), and each of the 7 inner products (m_s),
    # we need to specify how the 4x4 A, B, C blocks map to the 2x2x2 decomposition.

    # A_specs[k_p]: list of (coeff, block_idx) for the 2x2 A_eff
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)]   # P6: A01 - A11
    ]

    # B_specs[k_p]: list of (coeff, block_idx) for the 2x2 B_eff
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)]    # P6: B10 + B11
    ]

    # C_specs[k_p]: list of (coeff, block_idx) for the 2x2 C_eff
    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(1, 2), (-1, 3)],  # P1: C10 - C11
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)]            # P6: C00
    ]

    # Build the 4x4x8 decomposition
    # half=0: columns 0-3 of B, half=1: columns 4-7 of B
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s

                # Fill U: A is 4x4, indexed by global_r * 4 + global_c
                # A_eff is 2x2, indexed by r * 2 + c
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]

                # Fill V: B is 4x8, indexed by global_r * 8 + global_k
                # B_eff is 2x2, indexed by r * 2 + c
                # global_k = c + half * 4
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]

                # Fill W: C is 8x4 (p*n), indexed by global_k * 4 + global_i
                # C_eff is 2x2, indexed by r * 2 + c
                # global_i = r (row of C), global_k = c + half * 4 (col of C)
                # Wait, C is (p*n) x rank. p=8, n=4.
                # The product is C = A * B. A is 4x4, B is 4x8. C is 4x8.
                # The task says W has shape (p*n, rank) = (8*4, rank) = (32, rank).
                # The trace convention is W[k*n+i, r].
                # Here, the output matrix is 4x8. Let's call it M. M[i, k] where i in 0..3, k in 0..7.
                # The index in W is k*n + i = k*4 + i.
                # In the 2x2x2 decomposition, the output C is 2x2.
                # C_eff[r, c] corresponds to global row i = bi*2+r and global col k = bj*2+c + half*4.
                # So global_w_idx = (bj*2+c + half*4) * 4 + (bi*2+r).
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]

    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
