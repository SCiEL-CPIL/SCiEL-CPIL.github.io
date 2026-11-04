"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from scipy.optimize import least_squares
    
    n, m, p = DIMENSIONS
    
    # Target tensor indices
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    # Flatten T for vectorization
    T_flat = T.flatten()
    
    def objective(params, r):
        # Unpack params into U, V, W
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Total params: (n*m + m*p + p*n) * r
        nm = n * m
        mp = m * p
        pn = p * n
        
        U = params[:nm * r].reshape(nm, r)
        V = params[nm * r : (nm + mp) * r].reshape(mp, r)
        W = params[(nm + mp) * r : (nm + mp + pn) * r].reshape(pn, r)
        
        # Compute reconstructed tensor
        # T_recon[u, v, w] = sum_t U[u, t] * V[v, t] * W[w, t]
        # Efficiently:
        # Sum over t of U[:, t] * V[:, t] * W[:, t]
        # This is a tensor product sum.
        # Can be done via einsum
        T_recon = np.einsum('ut,vt,wt->uvw', U, V, W)
        
        # Residual
        residual = T_recon.flatten() - T_flat
        return residual
    
    # Try ranks from 14 to 17
    # Schoolbook is 18
    for r in range(14, 18):
        # Random initialization
        np.random.seed(42) # Fixed seed for reproducibility in search, but try a few
        for attempt in range(5):
            np.random.seed(42 + attempt * 100 + r)
            nm = n * m
            mp = m * p
            pn = p * n
            # Initialize with small random values
            params = np.random.randn((nm + mp + pn) * r) * 0.1
            
            # Solve
            try:
                sol = least_squares(objective, params, args=(r,), method='lm', max_nfev=2000)
                if sol.success or np.linalg.norm(sol.fun) < 1e-5:
                    # Check if solution is close to half-integers
                    U = sol.x[:nm * r].reshape(nm, r)
                    V = sol.x[nm * r : (nm + mp) * r].reshape(mp, r)
                    W = sol.x[(nm + mp) * r : (nm + mp + pn) * r].reshape(pn, r)
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    T_recon = np.einsum('ut,vt,wt->uvw', U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_recon, T, atol=1e-6):
                        # Convert to list of lists
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
            except Exception:
                pass
        
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
