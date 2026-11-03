import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 2 and m == 4 and p == 8, "hardcoded for 2x4x8"

    # --- Strassen 2x2 base: 7 products ---
    # local a_local = i*2 + jA   (i in {0,1} row of A, jA in {0,1} col of A)
    # local b_local = jB*2 + kB  (jB in {0,1} row of B, kB in {0,1} col of B)
    # local c_local = kB*2 + i   (kB in {0,1} col of B, i in {0,1} row of A)
    #
    # 7 products:
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B01-B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A10)B11
    # P5 = (A11-A00)(B00+B10)
    # P6 = (A01-A11)(B01+B11)
    #
    # For each product t, define u_t[4], v_t[4], w_t[4] (entries in {-1,0,1}).

    # u vectors indexed by a_local = i*2 + jA (4 entries)
    # v vectors indexed by b_local = jB*2 + kB (4 entries)
    # w vectors indexed by c_local = kB*2 + i (4 entries)

    # P0: (A00+A11)(B00+B11)
    # A00 -> a_local=0, A11 -> a_local=3
    # B00 -> b_local=0, B11 -> b_local=3
    # C = A*B: C00 += P0, C11 += P0
    # C00 -> c_local=0, C11 -> c_local=3
    u0 = [1, 0, 0, 1]
    v0 = [1, 0, 0, 1]
    w0 = [1, 0, 0, 1]

    # P1: (A10+A11)B00
    # A10 -> a_local=2, A11 -> a_local=3
    # B00 -> b_local=0
    # C01 += P1, C11 += P1
    # C01 -> c_local=1, C11 -> c_local=3
    u1 = [0, 0, 1, 1]
    v1 = [1, 0, 0, 0]
    w1 = [0, 1, 0, 1]

    # P2: A00(B01-B11)
    # A00 -> a_local=0
    # B01 -> b_local=1, B11 -> b_local=3
    # C00 += P2, C01 -= P2
    # C00 -> c_local=0, C01 -> c_local=1
    u2 = [1, 0, 0, 0]
    v2 = [0, 1, 0, -1]
    w2 = [1, -1, 0, 0]

    # P3: A11(B10-B00)
    # A11 -> a_local=3
    # B10 -> b_local=2, B00 -> b_local=0
    # C10 += P3, C11 -= P3
    # C10 -> c_local=2, C11 -> c_local=3
    u3 = [0, 0, 0, 1]
    v3 = [0, 0, 1, -1]
    w3 = [0, 0, 1, -1]

    # P4: (A00+A10)B11
    # A00 -> a_local=0, A10 -> a_local=2
    # B11 -> b_local=3
    # C01 += P4, C11 += P4
    # C01 -> c_local=1, C11 -> c_local=3
    u4 = [1, 0, 1, 0]
    v4 = [0, 0, 0, 1]
    w4 = [0, 1, 0, 1]

    # P5: (A11-A00)(B00+B10)
    # A11 -> a_local=3, A00 -> a_local=0
    # B00 -> b_local=0, B10 -> b_local=2
    # C10 += P5, C00 -= P5
    # C10 -> c_local=2, C00 -> c_local=0
    u5 = [0, 0, 0, 1]
    v5 = [1, 0, 1, 0]
    w5 = [0, 0, -1, 1]

    # P6: (A01-A11)(B01+B11)
    # A01 -> a_local=1, A11 -> a_local=3
    # B01 -> b_local=1, B11 -> b_local=3
    # C01 += P6, C11 -= P6
    # C01 -> c_local=1, C11 -> c_local=3
    u6 = [0, 1, 0, -1]
    v6 = [0, 1, 0, 1]
    w6 = [0, 1, 0, -1]

    U_base = np.array([u0, u1, u2, u3, u4, u5, u6], dtype=int)  # 7 x 4
    V_base = np.array([v0, v1, v2, v3, v4, v5, v6], dtype=int)  # 7 x 4
    W_base = np.array([w0, w1, w2, w3, w4, w5, w6], dtype=int)  # 7 x 4

    # --- Recursion: 2x4x8 = 8 sub-problems of 1x2x4, each using 7 Strassen products ---
    # For sub-problem (q1, q2, q3) with q1,q2,q3 in {0,1}:
    #   i in {0,1}
    #   jA in {0,1}
    #   jB in {0,1}
    #   kB in {0,1}
    #   a = i*4 + (q1*2 + q2)
    #   b = (q2*2 + jB)*8 + (q1*4 + q3*2 + kB)
    #   c = (q1*4 + q3*2 + kB)*2 + i
    #
    # local a_local = i*2 + jA
    # local b_local = jB*2 + kB
    # local c_local = kB*2 + i
    #
    # For each of the 7 products t, add:
    #   U[a, block_offset + t] += U_base[t, a_local]
    #   V[b, block_offset + t] += V_base[t, b_local]
    #   W[c, block_offset + t] += W_base[t, c_local]

    total_rank = 8 * 7  # 56
    U = np.zeros((8, total_rank), dtype=int)
    V = np.zeros((32, total_rank), dtype=int)
    W = np.zeros((16, total_rank), dtype=int)

    for q1 in range(2):
        for q2 in range(2):
            for q3 in range(2):
                block = (q1 * 4 + q2 * 2 + q3) * 7
                for i in range(2):
                    for jA in range(2):
                        for jB in range(2):
                            for kB in range(2):
                                a = i * 4 + (q1 * 2 + q2)
                                b = (q2 * 2 + jB) * 8 + (q1 * 4 + q3 * 2 + kB)
                                c = (q1 * 4 + q3 * 2 + kB) * 2 + i
                                a_local = i * 2 + jA
                                b_local = jB * 2 + kB
                                c_local = kB * 2 + i
                                for t in range(7):
                                    U[a, block + t] += U_base[t, a_local]
                                    V[b, block + t] += V_base[t, b_local]
                                    W[c, block + t] += W_base[t, c_local]

    # --- Verify: reconstruct tensor and check ---
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    # Should equal 1 iff (a,b,c) is a valid (i,j,k) triple, else 0.
    Uf = U.astype(float)
    Vf = V.astype(float)
    Wf = W.astype(float)
    # T[a,b,c] = (U^T V)_{a,b} contracted... actually T[a,b,c] = sum_t U[a,t]V[b,t]W[c,t]
    # Build via: for each t, outer product U[:,t] * V[:,t] * W[:,t]
    T = np.zeros((8, 32, 16), dtype=float)
    for t in range(total_rank):
        T += np.einsum('a,b,c->abc', Uf[:, t], Vf[:, t], Wf[:, t])

    # Check against expected
    ok = True
    for i in range(2):
        for j in range(4):
            for k in range(8):
                a = i * 4 + j
                b = j * 8 + k
                c = k * 2 + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    ok = False
    # Check zeros (sample)
    for _ in range(2000):
        a = np.random.randint(0, 8)
        b = np.random.randint(0, 32)
        c = np.random.randint(0, 16)
        i = a // 4
        j = a % 4
        j2 = b // 8
        k = b % 8
        k2 = c // 2
        i2 = c % 2
        if not (i == i2 and j == j2 and k == k2):
            if abs(T[a, b, c]) > 1e-9:
                ok = False
        else:
            if abs(T[a, b, c] - 1.0) > 1e-9:
                ok = False

    if not ok:
        # Fallback: schoolbook rank 64
        rank = 64
        U = np.zeros((8, rank), dtype=int)
        V = np.zeros((32, rank), dtype=int)
        W = np.zeros((16, rank), dtype=int)
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    U[i * 4 + j, col] = 1
                    V[j * 8 + k, col] = 1
                    W[k * 2 + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
