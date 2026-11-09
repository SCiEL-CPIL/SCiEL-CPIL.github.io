import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p
    
    # Function to build schoolbook decomposition
    def build_schoolbook():
        U = np.zeros((n * m, rank_sb))
        V = np.zeros((m * p, rank_sb))
        W = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Target tensor
    # T[i, j, k] = 1 for all i, j, k in the multiplication tensor?
    # The multiplication tensor M_{n,m,p} has entries 1 for all indices.
    # Because C_{ik} = sum_j A_{ij} B_{jk}.
    # The term A_{ij} B_{jk} contributes to C_{ik}.
    # So T_{i,j,k} = 1.
    T = np.ones((n, m, p))
    
    # Try to find a lower rank decomposition
    # We try rank = rank_sb - 1
    # If found, return it. Else return schoolbook.
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds for search
    
    # We only try R = rank_sb - 1 to keep it fast.
    # If rank_sb - 1 is not possible, we return rank_sb.
    
    R = rank_sb - 1
    if R > 0:
        # Variables: U (n*m*R), V (m*p*R), W (p*n*R)
        # Total vars: R * (n*m + m*p + p*n)
        def residuals(params):
            U = params[:n*m*R].reshape((n*m, R))
            V = params[n*m*R : n*m*R + m*p*R].reshape((m*p, R))
            W = params[n*m*R + m*p*R :].reshape((p*n, R))
            
            # Compute tensor sum
            # T_recon[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            T_recon = np.zeros((n, m, p))
            for r in range(R):
                u = U[:, r]
                v = V[:, r]
                w = W[:, r]
                # u is (n*m), v is (m*p), w is (p*n)
                # Reshape u to (n, m), v to (m, p), w to (p, n)
                u_2d = u.reshape((n, m))
                v_2d = v.reshape((m, p))
                w_2d = w.reshape((p, n))
                # Outer product sum
                # T_recon[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_recon[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
            
            return (T_recon - T).flatten()

        # Initial guess
        np.random.seed(42)
        params0 = np.random.rand(R * (n*m + m*p + p*n)) * 0.1
        
        try:
            # Use scipy.optimize.least_squares
            from scipy.optimize import least_squares
            # Limit max iterations to keep time low
            res = least_squares(residuals, params0, max_nfev=100)
            
            if res.success and np.linalg.norm(res.fun) < 1e-8:
                # Check if we can round to half-integers
                U_flat = res.x[:n*m*R].reshape((n*m, R))
                V_flat = res.x[n*m*R : n*m*R + m*p*R].reshape((m*p, R))
                W_flat = res.x[n*m*R + m*p*R :].reshape((p*n, R))
                
                # Round to nearest 0.5
                U_rounded = np.round(U_flat * 2) / 2
                V_rounded = np.round(V_flat * 2) / 2
                W_rounded = np.round(W_flat * 2) / 2
                
                # Verify exactness
                T_recon = np.zeros((n, m, p))
                for r in range(R):
                    u_2d = U_rounded[:, r].reshape((n, m))
                    v_2d = V_rounded[:, r].reshape((m, p))
                    w_2d = W_rounded[:, r].reshape((p, n))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_recon[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
                
                if np.allclose(T_recon, T, atol=1e-10):
                    return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
        except Exception:
            pass

    # Fallback to schoolbook
    return build_schoolbook()
