import time
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 11]
DEADLINE = 150.0
SAFETY = 5.0  # seconds to leave before hard kill

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS

    # --- Guaranteed rank-39 construction (Strassen blocks + schoolbook tail) ---
    def build_rank39():
        rank = 5 * 7 + 1 * 4
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0; v_f = k1; v_g = p + k0; v_h = p + k1
            w_c00 = k0 * n; w_c10 = k0 * n + 1
            w_c01 = k1 * n; w_c11 = k1 * n + 1
            # m1
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            # m2
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            # m3
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            # m4
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            # m5
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            # m6
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            # m7
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        # single column k2 = 10, schoolbook 4
        k2 = 10
        v_e2 = k2; v_g2 = p + k2
        w_c02 = k2 * n; w_c12 = k2 * n + 1
        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
        return {"U": U, "V": V, "W": W}

    best = build_rank39()

    # --- Verification helpers ---
    def verify(U, V, W):
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        # T shape: (n*m, m*p, p*n)
        T = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
        
        # Construct target tensor M of shape (n*m, m*p, p*n)
        # M[a, b, c] = 1 if indices are consistent, else 0
        # a = i*m + j
        # b = j*p + k
        # c = k*n + i
        M = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    M[a, b, c] = 1.0
        
        return np.allclose(T, M, atol=1e-9, rtol=0.0)

    def round_halfint(mat):
        return np.round(np.asarray(mat, dtype=float) * 2.0) / 2.0

    def find_2x2x3_rank10():
        n_sub, m_sub, p_sub = 2, 2, 3
        T = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    T[a, b, c] = 1.0
        T_flat = T.flatten()
        
        def loss(params):
            rank = 10
            U = params[:n_sub * m_sub * rank].reshape(n_sub * m_sub, rank)
            V = params[n_sub * m_sub * rank : n_sub * m_sub * rank + m_sub * p_sub * rank].reshape(m_sub * p_sub, rank)
            W = params[n_sub * m_sub * rank + m_sub * p_sub * rank :].reshape(p_sub * n_sub, rank)
            T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
            return np.sum((T_rec.flatten() - T_flat) ** 2)

        rng = np.random.default_rng(42)
        best_params = None
        best_loss = float('inf')
        
        for init_idx in range(10):
            if time.time() - t_start > DEADLINE - SAFETY:
                break
            U_init = rng.normal(0, 1, (4, 10))
            V_init = rng.normal(0, 1, (6, 10))
            W_init = rng.normal(0, 1, (6, 10))
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            res = minimize(loss, params_init, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
            if best_loss < 1e-6:
                break

        if best_params is None or best_loss > 1e-4:
            return None

        U = best_params[:40].reshape(4, 10)
        V = best_params[40:100].reshape(6, 10)
        W = best_params[100:160].reshape(6, 10)
        U_r = round_halfint(U)
        V_r = round_halfint(V)
        W_r = round_halfint(W)
        
        if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        return None

    def build_strassen_block(col_offset):
        rank = 7
        U_s = np.zeros((4, rank))
        V_s = np.zeros((22, rank))
        W_s = np.zeros((22, rank))
        k0 = col_offset
        k1 = col_offset + 1
        v_e = k0; v_f = k1; v_g = 11 + k0; v_h = 11 + k1
        w_c00 = k0 * 2; w_c10 = k0 * 2 + 1
        w_c01 = k1 * 2; w_c11 = k1 * 2 + 1
        
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[v_e, 0] = 1; V_s[v_h, 0] = 1
        W_s[w_c00, 0] = 1; W_s[w_c11, 0] = 1
        
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[v_e, 1] = 1
        W_s[w_c10, 1] = 1; W_s[w_c11, 1] = -1
        
        U_s[0, 2] = 1
        V_s[v_f, 2] = 1; V_s[v_h, 2] = -1
        W_s[w_c01, 2] = 1; W_s[w_c11, 2] = 1
        
        U_s[3, 3] = 1
        V_s[v_g, 3] = 1; V_s[v_e, 3] = -1
        W_s[w_c00, 3] = 1; W_s[w_c10, 3] = 1
        
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[v_h, 4] = 1
        W_s[w_c00, 4] = -1; W_s[w_c01, 4] = 1
        
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[v_e, 5] = 1; V_s[v_f, 5] = 1
        W_s[w_c11, 5] = 1
        
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[v_g, 6] = 1; V_s[v_h, 6] = 1
        W_s[w_c00, 6] = 1
        
        return U_s, V_s, W_s

    if time.time() - t_start < DEADLINE - 20:
        block_2x2x3 = find_2x2x3_rank10()
        if block_2x2x3:
            U_list = []
            V_list = []
            W_list = []
            
            for i in range(3):
                col_offset = i * 3
                U_b = np.array(block_2x2x3["U"])
                V_b = np.array(block_2x2x3["V"])
                W_b = np.array(block_2x2x3["W"])
                
                U_list.append(U_b)
                
                V_g = np.zeros((22, 10))
                for j in range(2):
                    for k_local in range(3):
                        l = j * 3 + k_local
                        k_global = col_offset + k_local
                        v_idx = j * 11 + k_global
                        V_g[v_idx, :] = V_b[l, :]
                V_list.append(V_g)
                
                W_g = np.zeros((22, 10))
                for k_local in range(3):
                    for i_sub in range(2):
                        c_local = k_local * 2 + i_sub
                        k_global = col_offset + k_local
                        c_global = k_global * 2 + i_sub
                        W_g[c_global, :] = W_b[c_local, :]
                W_list.append(W_g)
            
            U_s, V_s, W_s = build_strassen_block(9)
            U_list.append(U_s)
            V_list.append(V_s)
            W_list.append(W_s)
            
            U_final = np.hstack(U_list).tolist()
            V_final = np.hstack(V_list).tolist()
            W_final = np.hstack(W_list).tolist()
            
            if verify(U_final, V_final, W_final):
                return {"U": U_final, "V": V_final, "W": W_final}

    return best
