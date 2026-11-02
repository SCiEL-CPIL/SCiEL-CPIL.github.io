import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    Nu = n * m
    Nv = m * p
    Nw = p * n
    
    # Build target tensor T of shape (Nu, Nv, Nw)
    T = np.zeros((Nu, Nv, Nw), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    def generate_schoolbook():
        rank = n * m * p
        U = np.zeros((Nu, rank), dtype=np.float64)
        V = np.zeros((Nv, rank), dtype=np.float64)
        W = np.zeros((Nw, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    # Start with schoolbook as the best known solution
    U_best, V_best, W_best, rank_best = generate_schoolbook()
    
    # Time limit guard
    start_time = time.time()
    time_limit = 110.0 # 10s margin from 120s limit
    
    # Try to find lower rank
    # We will search for ranks from 23 down to 18
    for r in range(23, 17, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random restarts for this rank
        found = False
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(42 + restart + r)
            U = np.random.randn(Nu, r)
            V = np.random.randn(Nv, r)
            W = np.random.randn(Nw, r)
            
            # Normalize to avoid scaling issues
            norm_U = np.linalg.norm(U)
            norm_V = np.linalg.norm(V)
            norm_W = np.linalg.norm(W)
            if norm_U > 0: U /= norm_U
            if norm_V > 0: V /= norm_V
            if norm_W > 0: W /= norm_W
            
            # ALS iterations
            for it in range(50):
                if time.time() - start_time > time_limit:
                    break
                    
                # Update U
                # T[u, v, w] = sum_c U[u, c] V[v, c] W[w, c]
                # K[v, w, c] = V[v, c] W[w, c]
                K = np.einsum('vc,wc->vwc', V, W)
                K_flat = K.reshape(-1, r) # (Nv*Nw, r)
                T_flat = T.reshape(Nu, -1) # (Nu, Nv*Nw)
                
                # Solve K_flat.T @ U.T = T_flat.T
                # A = K_flat.T (r x Nv*Nw), B = T_flat.T (Nv*Nw x Nu)
                # X = lstsq(A, B) -> (r x Nu)
                # U = X.T
                try:
                    U_new, _, _, _ = np.linalg.lstsq(K_flat.T, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                    
                # Update V
                # L[u, w, c] = U[u, c] W[w, c]
                L = np.einsum('uc,wc->uwc', U, W)
                L_flat = L.reshape(-1, r) # (Nu*Nw, r)
                T_v = T.transpose(1, 0, 2).reshape(Nv, -1) # (Nv, Nu*Nw)
                
                # Solve L_flat.T @ V.T = T_v.T
                # A = L_flat.T (r x Nu*Nw), B = T_v.T (Nu*Nw x Nv)
                # X = lstsq(A, B) -> (r x Nv)
                # V = X.T
                try:
                    V_new, _, _, _ = np.linalg.lstsq(L_flat.T, T_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                    
                # Update W
                # M[u, v, c] = U[u, c] V[v, c]
                M = np.einsum('uc,vc->uvc', U, V)
                M_flat = M.reshape(-1, r) # (Nu*Nv, r)
                T_w = T.transpose(2, 0, 1).reshape(Nw, -1) # (Nw, Nu*Nv)
                
                # Solve M_flat.T @ W.T = T_w.T
                # A = M_flat.T (r x Nu*Nv), B = T_w.T (Nu*Nv x Nw)
                # X = lstsq(A, B) -> (r x Nw)
                # W = X.T
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_flat.T, T_w.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                
                # Check residual
                # Reconstruct T_rec
                T_rec = np.einsum('uc,vc,wc->uvw', U, V, W)
                residual = np.linalg.norm(T - T_rec)
                if residual < 1e-6:
                    # Try to snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify exactness with snapped values
                    T_snap = np.einsum('uc,vc,wc->uvw', U_snap, V_snap, W_snap)
                    if np.allclose(T_snap, T, atol=1e-9):
                        # Update best
                        U_best, V_best, W_best = U_snap, V_snap, W_snap
                        rank_best = r
                        found = True
                        break
            
            if found:
                break
        
        # If we found a solution for this rank, we can stop searching for lower ranks?
        # The task asks for rank-minimal. So we should continue to lower ranks.
        # But if we found rank r, we must check r-1.
        
    # Fallback to best found (which is at least schoolbook)
    U_list = [[float(x) for x in row] for row in U_best]
    V_list = [[float(x) for x in row] for row in V_best]
    W_list = [[float(x) for x in row] for row in W_best]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
