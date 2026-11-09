"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for matrix multiplication 3x3x8.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    deadline = time.time() + 120  # Safety margin
    
    # Schoolbook fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    
    # Try to find lower rank decompositions using numerical optimization
    # The tensor equation: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def to_list(mat):
        """Convert numpy array to list of lists with rounded values."""
        return [[float(x) for x in row] for row in mat]
    
    # Try ranks from 71 down to 40
    for target_rank in range(71, 39, -1):
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            
            try:
                # Random initialization
                U = np.random.randn(n*m, target_rank) * 0.1
                V = np.random.randn(m*p, target_rank) * 0.1
                W = np.random.randn(p*n, target_rank) * 0.1
                
                # Use scipy to solve the nonlinear system
                from scipy.optimize import least_squares
                
                def residuals(params):
                    U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
                    V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_flat = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    res = []
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                res.append(np.sum(U_flat[i*m+j] * V_flat[j*p+k] * W_flat[k*n+i]) - 1.0)
                    return res
                
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                result = least_squares(residuals, x0, method='lm', max_nfev=1000)
                
                if result.success or result.cost < 1e-10:
                    U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    if verify(U_sol, V_sol, W_sol):
                        best_result = {
                            "U": to_list(U_sol),
                            "V": to_list(V_sol),
                            "W": to_list(W_sol)
                        }
                        break  # Found valid decomposition for this rank
            except Exception:
                continue
        
        if best_result["U"] != schoolbook()["U"]:
            break  # Found a better decomposition
    
    return best_result
# EVOLVE-BLOCK-END
