import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition for fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to find a decomposition for ranks less than 72
    # We iterate from 71 downwards
    for r in range(schoolbook_rank - 1, 50, -1):
        if time.time() > deadline:
            break
            
        # We will try a few initializations
        # 1. Truncated schoolbook
        # 2. Random
        inits = []
        
        # Truncated schoolbook
        U_init = U_school[:, :r].copy()
        V_init = V_school[:, :r].copy()
        W_init = W_school[:, :r].copy()
        inits.append((U_init, V_init, W_init))
        
        # Random initializations
        for _ in range(3):
            U_rand = np.random.randn(n * m, r)
            V_rand = np.random.randn(m * p, r)
            W_rand = np.random.randn(p * n, r)
            # Normalize to prevent overflow/underflow in early steps
            U_rand /= np.linalg.norm(U_rand) * 10
            V_rand /= np.linalg.norm(V_rand) * 10
            W_rand /= np.linalg.norm(W_rand) * 10
            inits.append((U_rand, V_rand, W_rand))
            
        for U_init, V_init, W_init in inits:
            if time.time() > deadline:
                break
                
            size_u = n * m
            size_v = m * p
            size_w = p * n
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residuals(params):
                U = params[0:size_u*r].reshape(n, m, r)
                V = params[size_u*r:size_u*r + size_v*r].reshape(m, p, r)
                W = params[size_u*r + size_v*r:].reshape(p, n, r)
                
                # Compute the tensor D_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
                D = np.einsum('ijr,jkr,kir->ijk', U, V, W)
                
                return (D - 1.0).flatten()
            
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=500, ftol=1e-12, xtol=1e-12)
                
                if result.success and result.cost < 1e-10:
                    U_sol = result.x[0:size_u*r].reshape(n, m, r)
                    V_sol = result.x[size_u*r:size_u*r + size_v*r].reshape(m, p, r)
                    W_sol = result.x[size_u*r + size_v*r:].reshape(p, n, r)
                    
                    # Snap to nearest half-integer
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify
                    D_check = np.einsum('ijr,jkr,kir->ijk', U_snap, V_snap, W_snap)
                    
                    if np.allclose(D_check, 1.0, atol=1e-9):
                        # Convert to required format: (n*m, r), (m*p, r), (p*n, r)
                        U_out = U_snap.reshape(n*m, r).tolist()
                        V_out = V_snap.reshape(m*p, r).tolist()
                        W_out = W_snap.reshape(p*n, r).tolist()
                        
                        return {
                            "U": U_out,
                            "V": V_out,
                            "W": W_out
                        }
                        
            except Exception:
                pass
                
    # Fallback
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
