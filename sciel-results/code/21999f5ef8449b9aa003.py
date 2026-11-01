"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    rank = 60
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1

    # Strassen block for (i∈{0,1}, j∈{ja,jb}, k∈{ka,kb})
    # u indices: i*m + j
    # v indices: j*p + k
    # w indices: k*n + i
    def strassen_block(ja, jb, ka, kb):
        # u: (0,ja)=0*m+ja=ja, (1,ja)=m+ja=5+ja
        #    (0,jb)=jb, (1,jb)=m+jb=5+jb
        # v: (ja,ka)=ja*p+ka, (ja,kb)=ja*p+kb
        #    (jb,ka)=jb*p+ka, (jb,kb)=jb*p+kb
        # w: (ka,0)=ka*n+0=ka*2, (ka,1)=ka*n+1=ka*2+1
        #    (kb,0)=kb*2, (kb,1)=kb*2+1
        ua = ja       # (0, ja)
        ub = 5 + ja   # (1, ja)
        va = jb       # (0, jb)
        vb = 5 + jb   # (1, jb)
        wka0 = 2 * ka   # (ka, 0)
        wka1 = 2 * ka + 1  # (ka, 1)
        wkb0 = 2 * kb   # (kb, 0)
        wkb1 = 2 * kb + 1  # (kb, 1)
        vka = ja * p + ka  # (ja, ka)
        vkb = ja * p + kb  # (ja, kb)
        wka2 = jb * p + ka  # (jb, ka)
        wkb2 = jb * p + kb  # (jb, kb)

        # M1
        add_term([(ua, 1), (vb, 1)], [(vka, 1), (wkb2, 1)], [(wka0, 1), (wkb1, 1)])
        # M2
        add_term([(ub, 1), (vb, 1)], [(vka, 1)], [(wka1, 1), (wkb1, -1)])
        # M3
        add_term([(ua, 1)], [(vkb, 1), (wkb2, -1)], [(wka0 + 2, 1), (wkb1, 1)])
        # M4
        add_term([(vb, 1)], [(wka2, 1), (vka, -1)], [(wka0, 1), (wka1, 1)])
        # M5
        add_term([(ua, 1), (va, 1)], [(wkb2, 1)], [(wka0, -1), (wka0 + 2, 1)])
        # M6
        add_term([(ub, 1), (ua, -1)], [(vka, 1), (vkb, 1)], [(wkb1, 1)])
        # M7
        add_term([(va, 1), (vb, -1)], [(wka2, 1), (wkb2, 1)], [(wka0, 1)])

    # 2 Strassen blocks:
    # Block A: j∈{0,1}, k∈{0,1}
    strassen_block(0, 1, 0, 1)
    # Block B: j∈{2,3}, k∈{0,1}
    strassen_block(2, 3, 0, 1)

    # All (i,j,k) in the Strassen blocks:
    strassen_set = set()
    for i in range(n):
        for j in [0, 1, 2, 3]:
            for k in [0, 1]:
                strassen_set.add((i, j, k))

    # Standard terms for remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_set:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

    # col should be 7 + 7 + (20 - 8) = 14 + 12 = 26
    # But we allocated rank=60, so we have extra columns that are all zeros.
    # We need to return only the used columns.
    # Actually, the interface says shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # where rank is the actual rank. So we should trim to col.

    U_used = U[:, :col]
    V_used = V[:, :col]
    W_used = W[:, :col]

    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_used[i * m + j, :]
                v_row = V_used[j * p + k, :]
                w_row = W_used[k * n + i, :]
                result = np.sum(u_row * v_row * w_row)
                if abs(result - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return {"U": U_used.tolist(), "V": V_used.tolist(), "W": W_used.tolist()}
    else:
        # Fallback to trivial rank 60
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, c] = 1
                    V_fallback[j * p + k, c] = 1
                    W_fallback[k * n + i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
# EVOLVE-BLOCK-END
