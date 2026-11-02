import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        # For each (i,j,k): sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result = np.dot(U_np[i * m + j, :], V_np[j * p + k, :] * W_np[k * n + i, :])
                    if abs(result - 1) > 1e-6:
                        return False
        return True

    # ------------------------------------------------------------------
    # PRIMARY: Rank 50 decomposition
    #   Block A: j in {0,1}, k in {0..5}  -> 2x2x6  = 3 Strassen blocks  = 21
    #   Block B: j in {2,3,4}, k in {0..5} -> 2x3x6
    #            = 1 Strassen 2x2x2 (j={2,3},k={0,1}) + 2 schoolbook cols (j=4)
    #            = 7 + 2 + 2 + 2 + 2 = 15  (j={2,3}k={2..5}: 4 schoolbook, j=4: 6 schoolbook)
    #   Total = 21 + 15 = 36?  Let me recount carefully below.
    # ------------------------------------------------------------------
    # Actually, let me build it explicitly and count.

    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return False
        for u_i, c in u_coeffs:
            U[u_i][col] += c
        for v_i, c in v_coeffs:
            V[v_i][col] += c
        for w_i, c in w_coeffs:
            W[w_i][col] += c
        col += 1
        return True

    def strassen_222(j1, j2, k1, k2):
        """Strassen 2x2x2 for i in {0,1}, j in {j1,j2}, k in {k1,k2}. Returns 7 terms."""
        u00 = u_idx(0, j1); u01 = u_idx(0, j2)
        u10 = u_idx(1, j1); u11 = u_idx(1, j2)
        v00 = v_idx(j1, k1); v01 = v_idx(j1, k2)
        v10 = v_idx(j2, k1); v11 = v_idx(j2, k2)
        w00 = w_idx(k1, 0); w01 = w_idx(k2, 0)
        w10 = w_idx(k1, 1); w11 = w_idx(k2, 1)

        # P1 = (a00+a11)(b00+b11) -> C00, C11
        add_term([(u00,1),(u11,1)], [(v00,1),(v11,1)], [(w00,1),(w11,1)])
        # P2 = (a10+a11)b00 -> C10, -C11
        add_term([(u10,1),(u11,1)], [(v00,1)], [(w10,1),(w11,-1)])
        # P3 = a00(b01-b11) -> C01, C11
        add_term([(u00,1)], [(v01,1),(v11,-1)], [(w01,1),(w11,1)])
        # P4 = a11(b10-b00) -> C00, C10
        add_term([(u11,1)], [(v10,1),(v00,-1)], [(w00,1),(w10,1)])
        # P5 = (a00+a01)b11 -> -C00, C01
        add_term([(u00,1),(u01,1)], [(v11,1)], [(w00,-1),(w01,1)])
        # P6 = (a10-a00)(b00+b01) -> C11
        add_term([(u10,1),(u00,-1)], [(v00,1),(v01,1)], [(w11,1)])
        # P7 = (a01-a11)(b10+b11) -> C00
        add_term([(u01,1),(u11,-1)], [(v10,1),(v11,1)], [(w00,1)])

    def schoolbook(i, j, k):
        add_term([(u_idx(i,j),1)], [(v_idx(j,k),1)], [(w_idx(k,i),1)])

    # Block A: j in {0,1}, k in {0..5} -> 3 Strassen blocks (21 terms)
    strassen_222(0, 1, 0, 1)
    strassen_222(0, 1, 2, 3)
    strassen_222(0, 1, 4, 5)

    # Block B: j in {2,3,4}, k in {0..5}
    #   j={2,3}, k={0,1}: Strassen (7 terms)
    strassen_222(2, 3, 0, 1)
    #   j={2,3}, k={2,3}: Strassen (7 terms)
    strassen_222(2, 3, 2, 3)
    #   j={2,3}, k={4,5}: Strassen (7 terms)
    strassen_222(2, 3, 4, 5)
    #   j=4, k={0..5}: 6 schoolbook terms
    for k in range(6):
        schoolbook(0, 4, k)
        schoolbook(1, 4, k)

    # Count: 3*7 + 3*7 + 6 = 21 + 21 + 6 = 48
    # Hmm, that's 48, not 50. Let me check: we used j={0,1} for all k (3 strassen = 21),
    # and j={2,3} for all k (3 strassen = 21), and j=4 for all k (6 schoolbook = 6).
    # Total = 48. That's even better! But wait, j only goes 0..4 (m=5), so j={0,1,2,3,4}.
    # j={0,1}: 3 strassen blocks over k pairs = 21
    # j={2,3}: 3 strassen blocks over k pairs = 21
    # j={4}: 6 schoolbook = 6
    # Total = 48

    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # ------------------------------------------------------------------
    # FALLBACK: Rank 54 (6 Strassen 2x2x2 blocks + schoolbook for j=4)
    # ------------------------------------------------------------------
    rank54 = 54
    U2 = [[0] * rank54 for _ in range(n * m)]
    V2 = [[0] * rank54 for _ in range(m * p)]
    W2 = [[0] * rank54 for _ in range(p * n)]
    col2 = 0

    def add_term54(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col2
        if col2 >= rank54:
            return
        for u_i, c in u_coeffs:
            U2[u_i][col2] += c
        for v_i, c in v_coeffs:
            V2[v_i][col2] += c
        for w_i, c in w_coeffs:
            W2[w_i][col2] += c
        col2 += 1

    def strassen_222_54(j1, j2, k1, k2):
        u00 = u_idx(0, j1); u01 = u_idx(0, j2)
        u10 = u_idx(1, j1); u11 = u_idx(1, j2)
        v00 = v_idx(j1, k1); v01 = v_idx(j1, k2)
        v10 = v_idx(j2, k1); v11 = v_idx(j2, k2)
        w00 = w_idx(k1, 0); w01 = w_idx(k2, 0)
        w10 = w_idx(k1, 1); w11 = w_idx(k2, 1)

        add_term54([(u00,1),(u11,1)], [(v00,1),(v11,1)], [(w00,1),(w11,1)])
        add_term54([(u10,1),(u11,1)], [(v00,1)], [(w10,1),(w11,-1)])
        add_term54([(u00,1)], [(v01,1),(v11,-1)], [(w01,1),(w11,1)])
        add_term54([(u11,1)], [(v10,1),(v00,-1)], [(w00,1),(w10,1)])
        add_term54([(u00,1),(u01,1)], [(v11,1)], [(w00,-1),(w01,1)])
        add_term54([(u10,1),(u00,-1)], [(v00,1),(v01,1)], [(w11,1)])
        add_term54([(u01,1),(u11,-1)], [(v10,1),(v11,1)], [(w00,1)])

    # 6 Strassen blocks: j pairs (0,1),(2,3) x k pairs (0,1),(2,3),(4,5)
    for j1, j2 in [(0,1),(2,3)]:
        for k1, k2 in [(0,1),(2,3),(4,5)]:
            strassen_222_54(j1, j2, k1, k2)

    # j=4, all k: 6 schoolbook
    for k in range(6):
        add_term54([(u_idx(0,4),1)], [(v_idx(4,k),1)], [(w_idx(k,0),1)])
        add_term54([(u_idx(1,4),1)], [(v_idx(4,k),1)], [(w_idx(k,1),1)])

    # Total: 6*7 + 6 = 48. Wait, that's 48 again. Let me recount.
    # 6 strassen blocks * 7 = 42, plus 6 schoolbook = 48.
    # But the original rank-54 used j pairs (0,1),(2,3) and k pairs (0,1),(2,3),(4,5) = 6 blocks,
    # leaving j=4 uncovered. That's 42 + 6 = 48. The "54" was wrong in the original.
    # Anyway, verify and return.

    if verify(U2, V2, W2):
        return {"U": U2, "V": V2, "W": W2}

    # ------------------------------------------------------------------
    # ULTIMATE FALLBACK: Rank 60 trivial
    # ------------------------------------------------------------------
    rank60 = n * m * p
    U3 = [[0] * rank60 for _ in range(n * m)]
    V3 = [[0] * rank60 for _ in range(m * p)]
    W3 = [[0] * rank60 for _ in range(p * n)]
    col3 = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U3[i*m+j][col3] = 1
                V3[j*p+k][col3] = 1
                W3[k*n+i][col3] = 1
                col3 += 1
    return {"U": U3, "V": V3, "W": W3}
