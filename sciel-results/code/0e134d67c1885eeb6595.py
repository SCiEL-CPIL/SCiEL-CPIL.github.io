import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 5]

def _build_target(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                indices.append((a, b, c))
    return T, indices

def _tensor_from_factors(U, V, W):
    return np.einsum('ar,br,cr->abc', U, V, W)

def _is_valid_fast(U, V, W, target, indices, tol=1e-9):
    # Check only the non-zero entries
    # est[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # We can compute this for the specific indices
    # But vectorizing the sum for specific indices is tricky if we do it one by one.
    # Better: compute the full tensor if it's small, or use einsum with masks?
    # 16*20*20 = 6400 elements. Full tensor is small.
    est = _tensor_from_factors(U, V, W)
    return np.max(np.abs(est - target)) < tol

def _try_round_half(U, V, W, target, tol=1e-8):
    Ur = np.round(U * 2) / 2.0
    Vr = np.round(V * 2) / 2.0
    Wr = np.round(W * 2) / 2.0
    if _is_valid_fast(Ur, Vr, Wr, target, []):
        return Ur, Vr, Wr, True
    Ui = np.round(U)
    Vi = np.round(V)
    Wi = np.round(W)
    if _is_valid_fast(Ui, Vi, Wi, target, []):
        return Ui, Vi, Wi, True
    return U, V, W, False

def _strassen_2x2_terms():
    terms = []
    u = np.zeros(4); u[0] = 1; u[3] = 1
    v = np.zeros(4); v[0] = 1; v[3] = 1
    w = np.zeros(4); w[0] = 1; w[3] = 1
    terms.append((u, v, w))
    
    u = np.zeros(4); u[2] = 1
    v = np.zeros(4); v[0] = 1; v[1] = 1
    w = np.zeros(4); w[1] = 1; w[2] = 1
    terms.append((u, v, w))
    
    u = np.zeros(4); u[0] = 1; u[2] = -1
    v = np.zeros(4); v[2] = 1; v[3] = 1
    w = np.zeros(4); w[1] = 1; w[2] = 1
    terms.append((u, v, w))
    
    u = np.zeros(4); u[3] = 1
    v = np.zeros(4); v[3] = 1; v[1] = -1
    w = np.zeros(4); w[0] = 1; w[3] = 1
    terms.append((u, v, w))
    
    u = np.zeros(4); u[1] = 1; u[3] = 1
    v = np.zeros(4); v[0] = 1; v[2] = -1
    w = np.zeros(4); w[0] = -1; w[2] = 1
    terms.append((u, v, w))
    
    u = np.zeros(4); u[0] = 1; u[2] = 1
    v = np.zeros(4); v[1] = 1; v[3] = -1
    w = np.zeros(4); w[1] = 1; w[3] = 1
    terms.append((u, v, w))
    
    u = np.zeros(4); u[1] = 1; u[2] = -1
    v = np.zeros(4); v[2] = 1; v[3] = 1
    w = np.zeros(4); w[0] = 1; w[2] = -1
    terms.append((u, v, w))
    
    return terms

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    target, indices = _build_target(n, m, p)
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    terms = []
    
    def add_strassen(i_off, j_off, k_off):
        s = 2
        local_terms = _strassen_2x2_terms()
        for u, v, w in local_terms:
            uf = np.zeros(n * m)
            vf = np.zeros(m * p)
            wf = np.zeros(p * n)
            for idx in range(s * s):
                ii = idx // s
                jj = idx % s
                a = (i_off + ii) * m + (j_off + jj)
                uf[a] = u[idx]
                
                jj_local = idx // s
                kk_local = idx % s
                b = (j_off + jj_local) * p + (k_off + kk_local)
                vf[b] = v[idx]
                
                kk_local_w = idx // s
                ii_local_w = idx % s
                c = (k_off + kk_local_w) * n + (i_off + ii_local_w)
                wf[c] = w[idx]
            terms.append((uf, vf, wf))

    def add_schoolbook_vec(i_off, j_off, k_off):
        for ii in range(2):
            for jj in range(2):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                a = (i_off + ii) * m + (j_off + jj)
                b = (j_off + jj) * p + k_off
                c = k_off * n + (i_off + ii)
                u[a] = 1
                v[b] = 1
                w[c] = 1
                terms.append((u, v, w))

    # 4x4x4 part (k=0..3)
    add_strassen(0, 0, 0)
    add_strassen(0, 2, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 2)
    add_strassen(2, 0, 0)
    add_strassen(2, 2, 0)
    add_strassen(2, 0, 2)
    add_strassen(2, 2, 2)
    
    # 4x4x1 part (k=4)
    add_schoolbook_vec(0, 0, 4)
    add_schoolbook_vec(0, 2, 4)
    add_schoolbook_vec(2, 0, 4)
    add_schoolbook_vec(2, 2, 4)
    
    R0 = len(terms)
    U0 = np.column_stack([t[0] for t in terms])
    V0 = np.column_stack([t[1] for t in terms])
    W0 = np.column_stack([t[2] for t in terms])
    
    if _is_valid_fast(U0, V0, W0, target, indices):
        best_rank = R0
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        
    # Local search
    U_cur, V_cur, W_cur = best_U.copy(), best_V.copy(), best_W.copy()
    R_cur = best_rank
    merged = True
    while merged and time.time() < deadline - 15:
        merged = False
        for r1 in range(R_cur):
            if time.time() > deadline - 10:
                break
            for r2 in range(r1 + 1, R_cur):
                U_test = U_cur.copy()
                V_test = V_cur.copy()
                W_test = W_cur.copy()
                U_test[:, r2] = U_test[:, r1] + U_test[:, r2]
                V_test[:, r2] = V_test[:, r1] + V_test[:, r2]
                W_test[:, r2] = W_test[:, r1] + W_test[:, r2]
                U_test = np.delete(U_test, r1, axis=1)
                V_test = np.delete(V_test, r1, axis=1)
                W_test = np.delete(W_test, r1, axis=1)
                
                if _is_valid_fast(U_test, V_test, W_test, target, indices):
                    U_cur, V_cur, W_cur = U_test, V_test, W_test
                    R_cur -= 1
                    best_rank = R_cur
                    best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                    merged = True
                    break
            if merged:
                break
                
    U_final, V_final, W_final, _ = _try_round_half(best_U, best_V, best_W, target)
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
