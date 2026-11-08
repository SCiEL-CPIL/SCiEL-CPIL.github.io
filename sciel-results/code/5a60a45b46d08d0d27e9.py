import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    target_shape = (n, m, p)
    # Target tensor T[i, j, k] = 1 for all i, j, k
    T = np.ones((n, m, p))
    
    # Try ranks from lower bound to trivial
    lower_bound = max(n*m, m*p, p*n)
    trivial_rank = n * m * p
    
    best_solution = None
    best_rank = trivial_rank

    # Initialize trivial solution as fallback
    U_trivial = np.zeros((n*m, trivial_rank))
    V_trivial = np.zeros((m*p, trivial_rank))
    W_trivial = np.zeros((p*n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1
                V_trivial[j*p+k, col] = 1
                W_trivial[k*n+i, col] = 1
                col += 1
    best_solution = {"U": U_trivial, "V": V_trivial, "W": W_trivial}

    # Search for lower rank
    for r in range(lower_bound, trivial_rank):
        # Try a few random initializations
        found = False
        for attempt in range(5):
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Flatten
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals(x):
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r + m*p*r:].reshape(p*n, r)
                
                # Compute tensor product sum
                # U[i*m+j, alpha] * V[j*p+k, alpha] * W[k*n+i, alpha]
                # We need to sum over alpha
                # Reshape to 3D for easier einsum
                U_3d = U_flat.reshape(n, m, r)
                V_3d = V_flat.reshape(m, p, r)
                W_3d = W_flat.reshape(p, n, r)
                
                # T_calc[i, j, k] = sum_alpha U[i,j,alpha] * V[j,k,alpha] * W[k,i,alpha]
                # Note: W index is k*n+i, so W_3d[k, i, alpha]
                # einsum: 'jab, jcb, ka -> ijk'? No.
                # U[i, j, a] * V[j, k, a] * W[k, i, a]
                # Sum over a.
                # Result[i, j, k]
                T_calc = np.einsum('ija,jka,ik a->ijk'.replace(' ', ''), U_3d, V_3d, W_3d)
                
                # Residual is T_calc - T
                return (T_calc - T).flatten()

            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=1000)
                if result.cost < 1e-10:
                    U_final = result.x[:n*m*r].reshape(n*m, r)
                    V_final = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W_final = result.x[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    # Verify
                    U_3d = U_final.reshape(n, m, r)
                    V_3d = V_final.reshape(m, p, r)
                    W_3d = W_final.reshape(p, n, r)
                    T_check = np.einsum('ija,jka,ika->ijk', U_3d, V_3d, W_3d)
                    if np.allclose(T_check, T, atol=1e-9):
                        best_solution = {"U": U_final, "V": V_final, "W": W_final}
                        best_rank = r
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    return best_solution
