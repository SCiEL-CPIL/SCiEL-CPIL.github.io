import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 105

    n, m, p = 2, 4, 8

    # Strassen 2x2x2 local factors (rank 7 for 2x2 matrix multiplication)
    # U[i,j] for input A (2x2), V[i,j] for input B (2x2), W[i,j] for output C (2x2)
    # Indices: i in {0,1} row, j in {0,1} col
    # U_local[a_idx, r], V_local[b_idx, r], W_local[c_idx, r]
    # a_idx = i*2+j for A[i,j], b_idx = i*2+j for B[i,j], c_idx = i*2+j for C[i,j]
    
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)

    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 00=0, 01=1, 10=2, 11=3
    # C indices: 00=0, 01=1, 10=2, 11=3

    # r=0: M1 = (A00+A11)(B00+B11)
    U_local[0, 0] = 1  # A00
    U_local[3, 0] = 1  # A11
    V_local[0, 0] = 1  # B00
    V_local[3, 0] = 1  # B11

    # r=1: M2 = (A10+A11)B00
    U_local[2, 1] = 1  # A10
    U_local[3, 1] = 1  # A11
    V_local[0, 1] = 1  # B00

    # r=2: M3 = A00(B10-B11)
    U_local[0, 2] = 1  # A00
    V_local[2, 2] = 1  # B10
    V_local[3, 2] = -1 # B11

    # r=3: M4 = A11(B01-B00)
    U_local[3, 3] = 1  # A11
    V_local[1, 3] = 1  # B01
    V_local[0, 3] = -1 # B00

    # r=4: M5 = (A00+A01)B11
    U_local[0, 4] = 1  # A00
    U_local[1, 4] = 1  # A01
    V_local[3, 4] = 1  # B11

    # r=5: M6 = (A10-A00)(B00+B01)
    U_local[2, 5] = 1  # A10
    U_local[0, 5] = -1 # A00
    V_local[0, 5] = 1  # B00
    V_local[1, 5] = 1  # B01

    # r=6: M7 = (A01-A11)(B10+B11)
    U_local[1, 6] = 1  # A01
    U_local[3, 6] = -1 # A11
    V_local[2, 6] = 1  # B10
    V_local[3, 6] = 1  # B11

    # Output combination
    # C00 = M1 + M4 - M5 + M7
    W_local[0, 0] = 1  # M1
    W_local[0, 3] = 1  # M4
    W_local[0, 4] = -1 # M5
    W_local[0, 6] = 1  # M7

    # C01 = M3 + M5
    W_local[1, 2] = 1  # M3
    W_local[1, 4] = 1  # M5

    # C10 = M2 + M4
    W_local[2, 1] = 1  # M2
    W_local[2, 3] = 1  # M4

    # C11 = M1 - M2 + M3 + M6
    W_local[3, 0] = 1  # M1
    W_local[3, 1] = -1 # M2
    W_local[3, 2] = 1  # M3
    W_local[3, 5] = 1  # M6

    def verify(U, V, W):
        R = U.shape[1]
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    s = 0
                    for r in range(R):
                        s += U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
                    if s != 1:
                        return False
        return True

    # Build rank-56 Strassen decomposition:
    # Partition: A (2x4), B (4x8), C (8x2)
    # For each 2x2 block of A (2 blocks along column: cols 0-1, cols 2-3)
    # and each 2x2 block of B (4 blocks along rows: rows 0-1, rows 2-3)
    # we get a 2x2 product contributing to a 2x2 block of C
    # Total: 2 * 4 = 8 blocks, each rank 7 => rank 56
    
    total_rank = 56
    U = np.zeros((8, total_rank), dtype=int)
    V = np.zeros((32, total_rank), dtype=int)
    W = np.zeros((16, total_rank), dtype=int)

    # j_pair: which pair of columns of A (0: cols 0,1; 1: cols 2,3)
    # k_pair: which pair of rows of B (0: rows 0,1; 1: rows 2,3)
    for j_pair in range(2):
        for k_pair in range(4):
            base_r = (j_pair * 4 + k_pair) * 7

            # Map local A indices to global U indices
            # U index = i*4 + j where i in {0,1}, j in {0,1,2,3}
            # For j_pair=0: A columns 0,1 -> j=0,1
            # For j_pair=1: A columns 2,3 -> j=2,3
            if j_pair == 0:
                a_map = [0*4+0, 0*4+1, 1*4+0, 1*4+1]  # A00,A01,A10,A11 for j=0,1
            else:
                a_map = [0*4+2, 0*4+3, 1*4+2, 1*4+3]  # A00,A01,A10,A11 for j=2,3

            # Map local B indices to global V indices
            # V index = j*8 + k where j in {0,1,2,3}, k in {0..7}
            # For k_pair=0: B rows 0,1 -> j=0,1; k_pair=1: B rows 2,3 -> j=2,3
            k0 = k_pair * 2
            if k_pair < 2:
                # rows 0,1
                b_map = [0*8+k0, 0*8+k0+1, 1*8+k0, 1*8+k0+1]
            else:
                # rows 2,3
                b_map = [2*8+k0, 2*8+k0+1, 3*8+k0, 3*8+k0+1]

            # Map local C indices to global W indices
            # W index = k*2 + i where i in {0,1}, k in {0..7}
            # For k_pair=0: k=0,1; k_pair=1: k=2,3
            c_map = [k0*2+0, k0*2+1, (k0+1)*2+0, (k0+1)*2+1]

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val

    # Verify the decomposition
    if not verify(U, V, W):
        # Fallback to naive 64-rank
        U = np.zeros((8, 64), dtype=int)
        V = np.zeros((32, 64), dtype=int)
        W = np.zeros((16, 64), dtype=int)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j, r] = 1
                    V[j*8+k, r] = 1
                    W[k*2+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
