# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook
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
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    target_ranks = [55, 54, 53, 52, 51, 50]
    
    def verify_solution(U, V, W, r):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Reshape to (n, m, r), (m, p, r), (p, n, r)
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        # Compute T_approx
        # T_approx[i, j, k] = sum_c U3[i, j, c] V3[j, k, c] W3[k, i, c]
        # Use einsum
        # 'ijc,jkc,kic->ijk'
        T_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        # Check if all entries are 1
        if np.allclose(T_approx, 1.0, atol=1e-9):
            return True
        return False

    def try_rank(r, max_iters=500, n_restarts=10):
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize
            # Use random integers
            U = np.random.randint(-1, 2, (n, m, r))
            V = np.random.randint(-1, 2, (m, p, r))
            W = np.random.randint(-1, 2, (p, n, r))
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                # s[i, j, c] = sum_k V[j, k, c] W[k, i, c]
                s = np.einsum('jkc,kic->ijc', V, W)
                denom = np.sum(s**2, axis=2, keepdims=True)
                # Safe division
                U = np.divide(s, denom, out=np.zeros_like(s), where=denom>1e-12)
                
                # Update V
                # t[j, k, c] = sum_i U[i, j, c] W[k, i, c]
                t = np.einsum('ijc,kic->jkc', U, W)
                denom_v = np.sum(t**2, axis=2, keepdims=True)
                V = np.divide(t, denom_v, out=np.zeros_like(t), where=denom_v>1e-12)
                
                # Update W
                # w[k, i, c] = sum_j U[i, j, c] V[j, k, c]
                w = np.einsum('ijc,jkc->kic', U, V)
                denom_w = np.sum(w**2, axis=2, keepdims=True)
                W = np.divide(w, denom_w, out=np.zeros_like(w), where=denom_w>1e-12)
                
                # Check every 50 iters
                if it % 50 == 0:
                    U_flat = U.reshape(-1, r)
                    V_flat = V.reshape(-1, r)
                    W_flat = W.reshape(-1, r)
                    
                    # Try snapping to half-integers
                    U_snap = np.round(U_flat * 2) / 2
                    V_snap = np.round(V_flat * 2) / 2
                    W_snap = np.round(W_flat * 2) / 2
                    
                    if verify_solution(U_snap, V_snap, W_snap, r):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                    
                    # Try integers
                    U_int = np.round(U_flat)
                    V_int = np.round(V_flat)
                    W_int = np.round(W_flat)
                    if verify_solution(U_int, V_int, W_int, r):
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        
        return None

    for r in target_ranks:
        res = try_rank(r)
        if res is not None:
            best_rank = r
            best_result = res
            # Continue to try lower ranks
    return best_result
# EVOLVE-BLOCK-END
