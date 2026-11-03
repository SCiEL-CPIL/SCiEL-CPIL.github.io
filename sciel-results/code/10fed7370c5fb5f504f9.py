import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6

    # Block-Strassen decomposition: split into 2x2 blocks of size 3x3.
    # Outer Strassen gives 7 multiplications of 3x3 blocks.
    # Inner schoolbook for 3x3 gives rank 27.
    # Total rank = 7 * 27 = 189.
    bs = 3
    rank = 7 * 27

    # Use NumPy arrays for efficiency and clarity
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen definitions: 7 multiplications M_s = A_s * B_s
    # Blocks indexed 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_defs = [
        # M0: (A00 + A11)(B00 + B11)
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        # M1: (A10 + A11)B00
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        # M2: A00(B01 - B11)
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        # M3: A11(B10 - B00)
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        # M4: (A00 + A01)B11
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        # M5: (A10 - A00)(B00 + B01)
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        # M6: (A01 - A11)(B10 + B11)
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]

    # Reconstruction: C_block = sum_s gamma_s * M_s
    # C blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    recon_coeffs = [
        # C00 = M0 + M3 - M4 + M6
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        # C01 = M2 + M4
        [(2, 1.0), (4, 1.0)],
        # C10 = M1 + M3
        [(1, 1.0), (3, 1.0)],
        # C11 = M0 - M1 + M2 + M5
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]

    r_idx = 0
    start_time = time.time()
    deadline = 150.0  # Safety margin

    for s in range(7):
        if time.time() > deadline - 5.0:
            break # Safety break, though this shouldn't happen for static code
        
        s_def = strassen_defs[s]
        # Inner schoolbook rank-27 decomposition for 3x3*3x3
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    # Factor r_idx corresponds to product term A_s[u,v] * B_s[v,j]

                    # U factor: contribution from A_s[u,v]
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        U[u_row, r_idx] += coeff_a

                    # V factor: contribution from B_s[v,j]
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row, r_idx] += coeff_b

                    # W factor: contribution to C from M_s
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            # W index: W[k*n + i, r] where k is col of C, i is row of C
                            w_row = j_glob * n + i_glob
                            W[w_row, r_idx] += gamma

                    r_idx += 1

    assert r_idx == rank
    # Convert to list of lists to match interface if required, 
    # but the task says "factor matrices of shapes ... with integer/half-integer entries".
    # Usually numpy arrays are accepted or converted. 
    # The seed program returned lists. 
    # To be safe and match the "interface" exactly as in the current program (which returned lists), 
    # we should return lists.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
