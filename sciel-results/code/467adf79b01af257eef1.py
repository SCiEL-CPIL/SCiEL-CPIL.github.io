import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Target Tensor
    # T[u, v, w] = 1 if consistent indices
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin

    # Try to find lower rank
    for rank in range(25, 30):
        if time.time() > deadline:
            break
        
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Initialize
            # Small random integers or floats?
            # Using small floats helps convergence, then round.
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # ALS
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Update U
                for r in range(rank):
                    # K_r = sum(V[r]^2 * W[r]^2)
                    K_r = np.sum(V[:, r]**2 * W[:, r]**2)
                    if K_r < 1e-10:
                        continue
                    # N_u_r = sum_v,w T[u,v,w] * V[v,r] * W[w,r]
                    # T is fixed, V,W are fixed for this step
                    # We need T' = T - sum_{s != r} U_s V_s W_s
                    # This is expensive if done naively.
                    # Approximation: Use T directly? No, that's not ALS.
                    # Standard ALS uses the residual.
                    # Residual R = T - sum_{s=0}^{rank-1} U_s V_s W_s
                    # We can maintain the residual.
                    
                    # Let's implement residual maintenance
                    # But for simplicity in this short script, maybe just use T?
                    # No, that doesn't work for rank > 1.
                    
                    # Efficient Residual Update:
                    # This implementation is getting complex for 120s.
                    # Let's use a simpler check: 
                    # If we just do random search for integer solutions?
                    # Unlikely to work.
                    
                    # Let's stick to the residual calculation but optimize it.
                    # R = T - np.einsum('ir,jr,kr->ijk', U, V, W)
                    # This is O(n*m*p*rank). 30*25 = 750 ops. Very fast.
                    # We can recompute R every iteration.
                    pass
                
                # Recompute Residual
                R = T - np.einsum('ir,jr,kr->ijk', U, V, W)
                
                # Update U
                for r in range(rank):
                    K_r = np.sum(V[:, r]**2 * W[:, r]**2)
                    if K_r < 1e-10: continue
                    # N = np.einsum('uvw,vr,wr->u', R, V[:, r], W[:, r])
                    N = np.einsum('uvw,v, w->u', R, V[:, r], W[:, r])
                    U[:, r] = N / K_r

                # Update V
                for r in range(rank):
                    K_r = np.sum(U[:, r]**2 * W[:, r]**2)
                    if K_r < 1e-10: continue
                    N = np.einsum('uvw,u,w->v', R, U[:, r], W[:, r])
                    V[:, r] = N / K_r

                # Update W
                for r in range(rank):
                    K_r = np.sum(U[:, r]**2 * V[:, r]**2)
                    if K_r < 1e-10: continue
                    N = np.einsum('uvw,u,v->w', R, U[:, r], V[:, r])
                    W[:, r] = N / K_r
                    
            # Check convergence
            R_final = T - np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(R_final)
            if err < 1e-5:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                R_check = T - np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                if np.linalg.norm(R_check) < 1e-9:
                    # Convert to lists
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }

    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
