import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 2, 15
    nm, mp, pn = n*m, m*p, p*n
    
    def build_target_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        return T

    def get_rank_53():
        rank = 53
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        U_s = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ]
        V_s = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ]
        W_s = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        
        U_sc = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        V_sc = [
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]
        W_sc = [
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        
        col = 0
        for b in range(7):
            k0 = 2 * b
            k1 = 2 * b + 1
            for r in range(7):
                c = col + r
                for i in range(2):
                    for j in range(2):
                        U[i*m + j][c] = U_s[i*2 + j][r]
                for j in range(2):
                    V[j*p + k0][c] = V_s[j*2 + 0][r]
                    V[j*p + k1][c] = V_s[j*2 + 1][r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(2):
                        W[kk*n + i][c] = W_s[k*2 + i][r]
            col += 7
            
        k_last = 14
        for r in range(4):
            c = col + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_sc[i*2 + j][r]
            for j in range(2):
                V[j*p + k_last][c] = V_sc[j][r]
            for i in range(2):
                W[k_last*n + i][c] = W_sc[i][r]
                
        return {"U": U, "V": V, "W": W}

    def solve_als(T, target_rank, max_iters=200, tol=1e-10, seed=42, time_budget=30):
        np.random.seed(seed)
        nm, mp, pn = T.shape
        
        # Initialize
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        # Scale to avoid overflow
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for it in range(max_iters):
            if time.time() > time_budget:
                break
            
            # Update U
            # T_flat (nm, mp*pn) = U (nm, R) @ M_U (R, mp*pn)
            # M_U[r, b*pn+c] = V[b,r] * W[c,r]
            M_U = np.zeros((target_rank, mp * pn))
            for r in range(target_rank):
                M_U[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            
            T_flat = T.reshape(nm, mp * pn)
            # Solve U = T_flat @ M_U.T @ inv(M_U @ M_U.T)
            # Using lstsq is safer
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_U.T, T_flat.T, rcond=None)
                U = U_new.T
            except:
                break
            
            # Update V
            # T_flat2 (mp, nm*pn) = V (mp, R) @ M_V (R, nm*pn)
            # M_V[r, a*pn+c] = U[a,r] * W[c,r]
            M_V = np.zeros((target_rank, nm * pn))
            for r in range(target_rank):
                M_V[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            
            T_flat2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_V.T, T_flat2.T, rcond=None)
                V = V_new.T
            except:
                break
            
            # Update W
            # T_flat3 (pn, nm*mp) = W (pn, R) @ M_W (R, nm*mp)
            # M_W[r, a*mp+b] = U[a,r] * V[b,r]
            M_W = np.zeros((target_rank, nm * mp))
            for r in range(target_rank):
                M_W[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            
            T_flat3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_W.T, T_flat3.T, rcond=None)
                W = W_new.T
            except:
                break
            
            # Check convergence
            T_curr = np.zeros((nm, mp, pn))
            for r in range(target_rank):
                T_curr += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            err = np.linalg.norm(T_curr - T)
            if err < tol:
                return U, V, W
        
        # Final check
        T_curr = np.zeros((nm, mp, pn))
        for r in range(target_rank):
            T_curr += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        if np.linalg.norm(T_curr - T) < 1e-8:
            return U, V, W
        return None

    def try_lower_rank():
        T = build_target_tensor(n, m, p)
        
        # Try ranks from 52 downwards
        for rank in [52, 51, 50]:
            if time.time() > deadline - 20:
                break
            
            # Try multiple seeds
            for seed in [42, 123, 456]:
                if time.time() > deadline - 10:
                    break
                result = solve_als(T, rank, max_iters=100, seed=seed, time_budget=10)
                if result is not None:
                    U, V, W = result
                    U_list = [[U[i, r] for r in range(rank)] for i in range(nm)]
                    V_list = [[V[i, r] for r in range(rank)] for i in range(mp)]
                    W_list = [[W[i, r] for r in range(rank)] for i in range(pn)]
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    result = try_lower_rank()
    if result is not None:
        return result
    
    return get_rank_53()
