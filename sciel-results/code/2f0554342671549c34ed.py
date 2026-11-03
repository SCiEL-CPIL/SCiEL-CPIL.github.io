import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS

    def verify(U, V, W, rank):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # 2+3 split: I={0,1},{2,3,4}; J={0,1},{2,3,4}; K={0,1},{2,3,4}
    I_sizes = [2, 3]
    J_sizes = [2, 3]
    K_sizes = [2, 3]
    I_offsets = [0, 2]
    J_offsets = [0, 2]
    K_offsets = [0, 2]

    def strassen_block_2x2():
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[1, 1] = 1; W_s[3, 1] = -1
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[2, 2] = 1; W_s[3, 2] = 1
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[1, 3] = 1
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[2, 4] = -1; W_s[3, 4] = 1
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1
        return U_s, V_s, W_s

    def schoolbook_block(Iir, Jjr, Kkr):
        rank = Iir * Jjr * Kkr
        U_b = np.zeros((Iir * Jjr, rank))
        V_b = np.zeros((Jjr * Kkr, rank))
        W_b = np.zeros((Kkr * Iir, rank))
        col = 0
        for li in range(Iir):
            for lj in range(Jjr):
                for lk in range(Kkr):
                    U_b[li * Jjr + lj, col] = 1
                    V_b[lj * Kkr + lk, col] = 1
                    W_b[lk * Iir + li, col] = 1
                    col += 1
        return U_b, V_b, W_b

    blocks = []
    for ir in range(2):
        for kr in range(2):
            for jr in range(2):
                Iir = I_sizes[ir]
                Jjr = J_sizes[jr]
                Kkr = K_sizes[kr]
                blocks.append((ir, jr, kr, Iir, Jjr, Kkr))

    def get_block_rank(Iir, Jjr, Kkr):
        if Iir == 2 and Jjr == 2 and Kkr == 2:
            return 7
        return Iir * Jjr * Kkr

    total_rank = 0
    block_ranks = []
    for (ir, jr, kr, Iir, Jjr, Kkr) in blocks:
        r = get_block_rank(Iir, Jjr, Kkr)
        block_ranks.append(r)
        total_rank += r

    U_full = np.zeros((25, total_rank))
    V_full = np.zeros((25, total_rank))
    W_full = np.zeros((25, total_rank))

    col_offset = 0
    for idx, (ir, jr, kr, Iir, Jjr, Kkr) in enumerate(blocks):
        r = block_ranks[idx]
        Io = I_offsets[ir]
        Jo = J_offsets[jr]
        Ko = K_offsets[kr]

        if Iir == 2 and Jjr == 2 and Kkr == 2:
            U_b, V_b, W_b = strassen_block_2x2()
            for li in range(2):
                for lj in range(2):
                    for c in range(7):
                        U_full[(Io+li)*5 + (Jo+lj), col_offset+c] += U_b[li*2+lj, c]
            for lj in range(2):
                for lk in range(2):
                    for c in range(7):
                        V_full[(Jo+lj)*5 + (Ko+lk), col_offset+c] += V_b[lj*2+lk, c]
            for lk in range(2):
                for li in range(2):
                    for c in range(7):
                        W_full[(Ko+lk)*5 + (Io+li), col_offset+c] += W_b[lk*2+li, c]
        else:
            U_b, V_b, W_b = schoolbook_block(Iir, Jjr, Kkr)
            for li in range(Iir):
                for lj in range(Jjr):
                    for c in range(r):
                        U_full[(Io+li)*5 + (Jo+lj), col_offset+c] += U_b[li*Jjr+lj, c]
            for lj in range(Jjr):
                for lk in range(Kkr):
                    for c in range(r):
                        V_full[(Jo+lj)*5 + (Ko+lk), col_offset+c] += V_b[lj*Kkr+lk, c]
            for lk in range(Kkr):
                for li in range(Iir):
                    for c in range(r):
                        W_full[(Ko+lk)*5 + (Io+li), col_offset+c] += W_b[lk*Iir+li, c]

        col_offset += r

    err = verify(U_full, V_full, W_full, total_rank)
    if err < 1e-8:
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        U_school = np.zeros((25, 125))
        V_school = np.zeros((25, 125))
        W_school = np.zeros((25, 125))
        col = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U_school[i*5+j, col] = 1
                    V_school[j*5+k, col] = 1
                    W_school[k*5+i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
