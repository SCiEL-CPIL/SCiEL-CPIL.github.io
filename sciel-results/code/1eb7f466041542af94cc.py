"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    # Rank estimate: 6 Strassen blocks (7 terms each) + 1 tail term = 43
    rank = 43
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

    def strassen_block(ja, jb, ka, kb):
        # u indices: i*m + j
        # v indices: j*p + k
        # w indices: k*n + i
        ua = ja       # (0, ja)
        ub = 5 + ja   # (1, ja)
        va = jb       # (0, jb) -- Wait, in current code va/vb were u-indices?
        # Let's re-read current code's strassen_block
        # ua = ja (u index for i=0, j=ja)
        # ub = 5 + ja (u index for i=1, j=ja)
        # va = jb (u index for i=0, j=jb) -- NO, in current code:
        # va = jb was used in M5, M7 as u-index?
        # Let's check current code:
        # M5: add_term([(ua, 1), (va, 1)], ...)
        # ua is u-index. va is u-index?
        # In current code:
        # ua = ja
        # ub = 5 + ja
        # va = jb
        # vb = 5 + jb
        # So va, vb are u-indices for j=jb.
        # vka = ja * p + ka (v-index)
        # vkb = ja * p + kb (v-index)
        # wka2 = jb * p + ka (v-index)
        # wkb2 = jb * p + kb (v-index)
        # wka0, wka1, wkb0, wkb1 are w-indices.
        
        # My previous analysis of current code:
        # ua, ub, va, vb are U indices.
        # vka, vkb, wka2, wkb2 are V indices.
        # wka0, wka1, wkb0, wkb1 are W indices.
        
        # Let's stick to the variable names in the current code to avoid confusion.
        
        ua = ja       # U[0, ja]
        ub = 5 + ja   # U[1, ja]
        va = jb       # U[0, jb]
        vb = 5 + jb   # U[1, jb]
        
        vka = ja * p + ka  # V[ja, ka]
        vkb = ja * p + kb  # V[ja, kb]
        wka2 = jb * p + ka # V[jb, ka]
        wkb2 = jb * p + kb # V[jb, kb]
        
        wka0 = 2 * ka   # W[ka, 0]
        wka1 = 2 * ka + 1  # W[ka, 1]
        wkb0 = 2 * kb   # W[kb, 0]
        wkb1 = 2 * kb + 1  # W[kb, 1]

        # M1
        add_term([(ua, 1), (vb, 1)], [(vka, 1), (wkb2, 1)], [(wka0, 1), (wkb1, 1)])
        # M2
        add_term([(ub, 1), (vb, 1)], [(vka, 1)], [(wka1, 1), (wkb1, -1)])
        # M3
        add_term([(ua, 1)], [(vkb, 1), (wkb2, -1)], [(wka0 + 2, 1), (wkb1, 1)])
        # Note: wka0 + 2 is wkb0?
        # wka0 = 2*ka. wkb0 = 2*kb.
        # If ka=0, kb=1, wka0=0, wkb0=2. wka0+2=2. Yes.
        # M4
        add_term([(vb, 1)], [(wka2, 1), (vka, -1)], [(wka0, 1), (wka1, 1)])
        # M5
        add_term([(ua, 1), (va, 1)], [(wkb2, 1)], [(wka0, -1), (wka0 + 2, 1)])
        # M6
        add_term([(ub, 1), (ua, -1)], [(vka, 1), (vkb, 1)], [(wkb1, 1)])
        # M7
        add_term([(va, 1), (vb, -1)], [(wka2, 1), (wkb2, 1)], [(wka0, 1)])

    # 6 Strassen blocks
    strassen_block(0, 1, 0, 1)
    strassen_block(2, 3, 0, 1)
    strassen_block(0, 1, 2, 3)
    strassen_block(2, 3, 2, 3)
    strassen_block(0, 1, 4, 5)
    strassen_block(2, 3, 4, 5)

    # 1 Tail term for j=4
    # U indices: 4, 9
    # V indices: 24, 25, 26, 27, 28, 29
    # W indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    
    u_coeffs = [(4, 1), (9, 1)]
    v_coeffs = [(24+k, 1) for k in range(6)]
    w_coeffs = [(k*n + i, 1) for k in range(6) for i in range(2)]
    
    add_term(u_coeffs, v_coeffs, w_coeffs)

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
