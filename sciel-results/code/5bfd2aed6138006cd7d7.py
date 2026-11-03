import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin

    n, m, p = DIMENSIONS

    def verify(U, V, W, rank):
        """Verify decomposition. Returns max error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # We use a block decomposition with splits:
    # I: {0,1}, {2,3}, {4}
    # J: {0,1}, {2,3}, {4}
    # K: {0,1}, {2,3}, {4}
    #
    # Blocks of A: A[ir][jr], B: B[jr][kr], C: C[ir][kr]
    #
    # We'll use Strassen (rank 7) for 2x2 * 2x2 -> 2x2 blocks
    # and schoolbook for other sizes.
    #
    # Let's enumerate all blocks:
    # ir, jr, kr in {0,1,2}
    # Block sizes:
    #   I0=2, I1=2, I2=1
    #   J0=2, J1=2, J2=1
    #   K0=2, K1=2, K2=1
    #
    # Block (ir, jr, kr): A[Iir x Jjr] * B[Jjr x Kkr] -> C[Iir x Kkr]
    #
    # We need to find which blocks contribute to C[ir][kr]:
    # C[ir][kr] = sum_jr A[ir][jr] * B[jr][kr]
    #
    # So for each (ir, kr), we sum over jr.
    #
    # Let's list all contributions:
    # (ir, jr, kr) -> rank of that block multiplication
    #
    # I sizes: [2, 2, 1]
    # J sizes: [2, 2, 1]
    # K sizes: [2, 2, 1]
    #
    # Block (ir, jr, kr) has size: A is Iir x Jjr, B is Jjr x Kkr
    # Rank depends on the multiplication:
    #   If all three are 2x2 (i.e., Iir=Jjr=Kkr=2): rank 7 (Strassen)
    #   Otherwise: schoolbook rank = Iir * Jjr * Kkr
    #
    # Let's enumerate:
    blocks = []
    for ir in range(3):
        for kr in range(3):
            for jr in range(3):
                Iir = [2, 2, 1][ir]
                Jjr = [2, 2, 1][jr]
                Kkr = [2, 2, 1][kr]
                # This block contributes to C[ir][kr]
                blocks.append((ir, jr, kr, Iir, Jjr, Kkr))

    # Now we need to assign indices.
    # Global index for A element (i, j): i*5 + j
    # Global index for B element (j, k): j*5 + k
    # Global index for C element (k, i): k*5 + i

    # Let's define the index offsets for each block:
    I_offsets = [0, 2, 4]
    J_offsets = [0, 2, 4]
    K_offsets = [0, 2, 4]

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

    print(f"Total rank: {total_rank}")

    # Now construct U, V, W
    U_full = np.zeros((25, total_rank))
    V_full = np.zeros((25, total_rank))
    W_full = np.zeros((25, total_rank))

    col_offset = 0

    def strassen_block_2x2():
        """
        Strassen decomposition for 2x2 * 2x2 -> 2x2.
        Returns U, V, W of shape (4, 7).
        U index: i*2 + j
        V index: j*2 + k
        W index: k*2 + i
        """
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))

        # M1 = (A11+A22)(B11+B22)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1  # C11, C22

        # M2 = (A21+A22)B11
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[1, 1] = 1; W_s[3, 1] = -1  # C21, C22

        # M3 = A11(B12-B22)
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[2, 2] = 1; W_s[3, 2] = 1  # C12, C22

        # M4 = A22(B21-B11)
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[1, 3] = 1  # C11, C21

        # M5 = (A11+A12)B22
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[2, 4] = -1; W_s[3, 4] = 1  # C12, C22

        # M6 = (A21-A11)(B11+B12)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1  # C22

        # M7 = (A12-A22)(B21+B22)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1  # C11

        return U_s, V_s, W_s

    def schoolbook_block(Iir, Jjr, Kkr):
        """
        Schoolbook decomposition for Iir x Jjr * Jjr x Kkr -> Iir x Kkr.
        Returns U, V, W of shapes (Iir*Jjr, rank), (Jjr*Kkr, rank), (Kkr*Iir, rank).
        """
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

    for idx, (ir, jr, kr, Iir, Jjr, Kkr) in enumerate(blocks):
        r = block_ranks[idx]
        Io = I_offsets[ir]
        Jo = J_offsets[jr]
        Ko = K_offsets[kr]

        if Iir == 2 and Jjr == 2 and Kkr == 2:
            U_b, V_b, W_b = strassen_block_2x2()
            # U: local (li, lj) -> global (Io+li, Jo+lj)
            for li in range(2):
                for lj in range(2):
                    for c in range(7):
                        U_full[(Io+li)*5 + (Jo+lj), col_offset+c] += U_b[li*2+lj, c]
            # V: local (lj, lk) -> global (Jo+lj, Ko+lk)
            for lj in range(2):
                for lk in range(2):
                    for c in range(7):
                        V_full[(Jo+lj)*5 + (Ko+lk), col_offset+c] += V_b[lj*2+lk, c]
            # W: local (lk, li) -> global (Ko+lk, Io+li)
            for lk in range(2):
                for li in range(2):
                    for c in range(7):
                        W_full[(Ko+lk)*5 + (Io+li), col_offset+c] += W_b[lk*2+li, c]
        else:
            U_b, V_b, W_b = schoolbook_block(Iir, Jjr, Kkr)
            # U: local (li, lj) -> global (Io+li, Jo+lj)
            for li in range(Iir):
                for lj in range(Jjr):
                    for c in range(r):
                        U_full[(Io+li)*5 + (Jo+lj), col_offset+c] += U_b[li*Jjr+lj, c]
            # V: local (lj, lk) -> global (Jo+lj, Ko+lk)
            for lj in range(Jjr):
                for lk in range(Kkr):
                    for c in range(r):
                        V_full[(Jo+lj)*5 + (Ko+lk), col_offset+c] += V_b[lj*Kkr+lk, c]
            # W: local (lk, li) -> global (Ko+lk, Io+li)
            for lk in range(Kkr):
                for li in range(Iir):
                    for c in range(r):
                        W_full[(Ko+lk)*5 + (Io+li), col_offset+c] += W_b[lk*Iir+li, c]

        col_offset += r

    # Verify
    err = verify(U_full, V_full, W_full, total_rank)
    if err < 1e-8:
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        # Fallback to schoolbook
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
