"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_rank(r, max_trials=3):
        for trial in range(max_trials):
            np.random.seed(trial * 100 + r)
            start_time = time.time()
            
            # SVD Init
            T_mat = T.reshape(nm, mp * pn)
            U_svd, s, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
            r_eff = min(r, len(s))
            
            # Initialize V, W randomly, U from SVD
            U = U_svd[:, :r_eff].copy()
            V = np.random.randn(mp, r_eff)
            W = np.random.randn(pn, r_eff)
            
            for it in range(20):
                if time.time() - start_time > 2.0:
                    break
                
                # Update U
                M_VW = np.zeros((mp * pn, r_eff))
                for c in range(r_eff):
                    M_VW[:, c] = (V[:, c] * W[:, c]).flatten()
                U = T_mat @ np.linalg.pinv(M_VW.T)
                
                # Update V
                T_mat_V = T.reshape(mp, nm * pn)
                M_UW = np.zeros((nm * pn, r_eff))
                for c in range(r_eff):
                    M_UW[:, c] = (U[:, c] * W[:, c]).flatten()
                V = T_mat_V @ np.linalg.pinv(M_UW.T)
                
                # Update W
                T_mat_W = T.reshape(pn, nm * mp)
                M_UV = np.zeros((nm * mp, r_eff))
                for c in range(r_eff):
                    M_UV[:, c] = (U[:, c] * V[:, c]).flatten()
                W = T_mat_W @ np.linalg.pinv(M_UV.T)
                
                # Check residual
                res_tensor = np.einsum('ac,bc,dc->abd', U, V, W)
                residual = np.linalg.norm(T - res_tensor)
                
                if residual < 1e-8:
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    res_round = np.linalg.norm(T - np.einsum('ac,bc,dc->abd', U_round, V_round, W_round))
                    if res_round < 1e-9:
                        return {"U": U_round, "V": V_round, "W": W_round}
                        
        return None

    # Try ranks 35 to 41
    for r in range(35, 42):
        res = try_rank(r)
        if res is not None:
            return res
            
    return schoolbook()
# EVOLVE-BLOCK-END
