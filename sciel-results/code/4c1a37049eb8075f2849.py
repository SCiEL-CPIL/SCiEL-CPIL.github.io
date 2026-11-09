"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin

    # Block structure: 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    num_blocks = 3
    offsets = [0]
    for bs in block_sizes:
        offsets.append(offsets[-1] + bs)

    U_terms = []
    V_terms = []
    W_terms = []

    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                si = block_sizes[bi]
                sj = block_sizes[bj]
                sk = block_sizes[bk]

                a_row_start = offsets[bi]
                a_col_start = offsets[bj]
                b_row_start = offsets[bj]
                b_col_start = offsets[bk]
                c_row_start = offsets[bi]
                c_col_start = offsets[bk]

                if si == 2 and sj == 2 and sk == 2:
                    # Strassen for 2x2 blocks: 7 rank-1 terms
                    # A indices: A00=(ar,ac), A01=(ar,ac+1), A10=(ar+1,ac), A11=(ar+1,ac+1)
                    # B indices: B00=(br,bc), B01=(br,bc+1), B10=(br+1,bc), B11=(br+1,bc+1)
                    # C indices: C00=(cr,cc), C01=(cr,cc+1), C10=(cr+1,cc), C11=(cr+1,cc+1)
                    #
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    #
                    # M1 = (A00+A11)(B00+B11)
                    # M2 = (A10+A11)(B00)
                    # M3 = A00(B01-B11)
                    # M4 = A11(B10-B00)
                    # M5 = (A00+A01)(B11)
                    # M6 = (A10-A00)(B00+B01)
                    # M7 = (A01-A11)(B10+B11)

                    ar0 = a_row_start + 0
                    ar1 = a_row_start + 1
                    ac0 = a_col_start + 0
                    ac1 = a_col_start + 1
                    br0 = b_row_start + 0
                    br1 = b_row_start + 1
                    bc0 = b_col_start + 0
                    bc1 = b_col_start + 1
                    cr0 = c_row_start + 0
                    cr1 = c_row_start + 1
                    cc0 = c_col_start + 0
                    cc1 = c_col_start + 1

                    # M1 = (A00+A11)(B00+B11)
                    # U: A00 and A11 have coefficient 1
                    # V: B00 and B11 have coefficient 1
                    # W: C00 coeff +1, C11 coeff +1
                    U_terms.append([(ar0*5+ac0, 1), (ar1*5+ac1, 1)])
                    V_terms.append([(br0*5+bc0, 1), (br1*5+bc1, 1)])
                    W_terms.append([(cc0*5+cr0, 1), (cc1*5+cr1, 1)])

                    # M2 = (A10+A11)(B00)
                    # U: A10 and A11 have coefficient 1
                    # V: B00 has coefficient 1
                    # W: C10 coeff +1
                    U_terms.append([(ar1*5+ac0, 1), (ar1*5+ac1, 1)])
                    V_terms.append([(br0*5+bc0, 1)])
                    W_terms.append([(cc0*5+cr1, 1)])

                    # M3 = A00(B01-B11)
                    # U: A00 has coefficient 1
                    # V: B01 coeff +1, B11 coeff -1
                    # W: C01 coeff +1, C11 coeff +1
                    U_terms.append([(ar0*5+ac0, 1)])
                    V_terms.append([(br0*5+bc1, 1), (br1*5+bc1, -1)])
                    W_terms.append([(cc1*5+cr0, 1), (cc1*5+cr1, 1)])

                    # M4 = A11(B10-B00)
                    # U: A11 has coefficient 1
                    # V: B10 coeff +1, B00 coeff -1
                    # W: C00 coeff +1, C10 coeff +1
                    U_terms.append([(ar1*5+ac1, 1)])
                    V_terms.append([(br1*5+bc0, 1), (br0*5+bc0, -1)])
                    W_terms.append([(cc0*5+cr0, 1), (cc0*5+cr1, 1)])

                    # M5 = (A00+A01)(B11)
                    # U: A00 and A01 have coefficient 1
                    # V: B11 has coefficient 1
                    # W: C00 coeff -1, C01 coeff +1
                    U_terms.append([(ar0*5+ac0, 1), (ar0*5+ac1, 1)])
                    V_terms.append([(br1*5+bc1, 1)])
                    W_terms.append([(cc0*5+cr0, -1), (cc1*5+cr0, 1)])

                    # M6 = (A10-A00)(B00+B01)
                    # U: A10 coeff +1, A00 coeff -1
                    # V: B00 and B01 have coefficient 1
                    # W: C11 coeff +1
                    U_terms.append([(ar1*5+ac0, 1), (ar0*5+ac0, -1)])
                    V_terms.append([(br0*5+bc0, 1), (br0*5+bc1, 1)])
                    W_terms.append([(cc1*5+cr1, 1)])

                    # M7 = (A01-A11)(B10+B11)
                    # U: A01 coeff +1, A11 coeff -1
                    # V: B10 and B11 have coefficient 1
                    # W: C00 coeff +1
                    U_terms.append([(ar0*5+ac1, 1), (ar1*5+ac1, -1)])
                    V_terms.append([(br1*5+bc0, 1), (br1*5+bc1, 1)])
                    W_terms.append([(cc0*5+cr0, 1)])

                else:
                    # Schoolbook for this block
                    for ii in range(si):
                        for jj in range(sj):
                            for kk in range(sk):
                                U_terms.append([(a_row_start + ii, a_col_start + jj, 1)])
                                V_terms.append([(b_row_start + jj, b_col_start + kk, 1)])
                                W_terms.append([(c_row_start + ii, c_col_start + kk, 1)])

    rank = len(U_terms)
    
    # Build U, V, W matrices
    # U shape: (n*m, rank) = (25, rank)
    # V shape: (m*p, rank) = (25, rank)  
    # W shape: (p*n, rank) = (25, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for col in range(rank):
        for (row_idx, val) in U_terms[col]:
            U[row_idx, col] = val
        for (row_idx, val) in V_terms[col]:
            V[row_idx, col] = val
        for (row_idx, val) in W_terms[col]:
            W[row_idx, col] = val

    # Verify correctness using vectorized numpy
    # Check: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err = abs(lhs - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > 1e-9:
                    # Fallback to schoolbook
                    rank_fb = n * m * p
                    U_fb = np.zeros((n * m, rank_fb), dtype=float)
                    V_fb = np.zeros((m * p, rank_fb), dtype=float)
                    W_fb = np.zeros((p * n, rank_fb), dtype=float)
                    column = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj, column] = 1
                                V_fb[jj * p + kk, column] = 1
                                W_fb[kk * n + ii, column] = 1
                                column += 1
                    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
