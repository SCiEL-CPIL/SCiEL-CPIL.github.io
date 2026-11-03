import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T
    T_flat = np.zeros((n*m) * (m*p) * (p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_flat[u * (m*p) * (p*n) + v * (p*n) + w] = 1.0
                
    def residual(params, R):
        U_flat = params[:R*n*m]
        V_flat = params[R*n*m : R*n*m + R*m*p]
        W_flat = params[R*n*m + R*m*p : R*n*m + R*m*p + R*p*n]
        
        U = U_flat.reshape(R, n*m).T
        V = V_flat.reshape(R, m*p).T
        W = W_flat.reshape(R, p*n).T
        
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        return T_recon.ravel() - T_flat

    # Search for rank R
    # We prioritize lower ranks
    for R in range(12, n*m*p):
        # Limit attempts to ensure time limit is respected
        attempts = 3 if R < 18 else 1
        
        for attempt in range(attempts):
            np.random.seed(attempt * 1000 + R * 10)
            
            U_init = np.random.randn(R, n*m) * 0.5
            V_init = np.random.randn(R, m*p) * 0.5
            W_init = np.random.randn(R, p*n) * 0.5
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use LM for speed
                result = least_squares(lambda x: residual(x, R), params_init, method='lm')
                
                # Check if solution is good
                if np.linalg.norm(result.fun) < 1e-6:
                    R_curr = R
                    U_flat = result.x[:R_curr*n*m]
                    V_flat = result.x[R_curr*n*m : R_curr*n*m + R_curr*m*p]
                    W_flat = result.x[R_curr*n*m + R_curr*m*p : R_curr*n*m + R_curr*m*p + R_curr*p*n]
                    
                    U = U_flat.reshape(R_curr, n*m).T
                    V = V_flat.reshape(R_curr, m*p).T
                    W = W_flat.reshape(R_curr, p*n).T
                    
                    # Snap to nearest 0.5
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify exactness
                    T_recon = np.einsum('ur,vr,wr->uvw', U_snap, V_snap, W_snap)
                    if np.allclose(T_recon.ravel(), T_flat, atol=1e-9):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
            except Exception:
                continue
                
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
