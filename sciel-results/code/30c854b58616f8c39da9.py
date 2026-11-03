"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Target: Find rank of M_{2,3,3} and lift to M_{2,3,9}
    # M_{2,3,3} dimensions: n1=2, m1=3, p1=3
    n1, m1, p1 = 2, 3, 3
    
    # Try ranks from 9 to 18
    for rank in range(9, 19):
        if time.time() > deadline - 10.0:
            break
        
        # Try a few random initializations
        for trial in range(10):
            if time.time() > deadline - 5.0:
                break
            
            np.random.seed(rank * 100 + trial)
            
            # Initialize with small random values
            U = np.random.randn(n1*m1, rank) * 0.5
            V = np.random.randn(m1*p1, rank) * 0.5
            W = np.random.randn(p1*n1, rank) * 0.5
            
            def flatten_params():
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unpack_params(x):
                U_flat = x[:n1*m1*rank]
                V_flat = x[n1*m1*rank:n1*m1*rank + m1*p1*rank]
                W_flat = x[n1*m1*rank + m1*p1*rank:]
                return U_flat.reshape(n1*m1, rank), V_flat.reshape(m1*p1, rank), W_flat.reshape(p1*n1, rank)
            
            def objective(x):
                U_opt, V_opt, W_opt = unpack_params(x)
                err = 0.0
                for i in range(n1):
                    for j in range(m1):
                        for k in range(p1):
                            val = np.sum(U_opt[i*m1+j, :] * V_opt[j*p1+k, :] * W_opt[k*n1+i, :])
                            err += (val - 1.0) ** 2
                return err
            
            x0 = flatten_params()
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                if result.fun < 1e-10:
                    U_opt, V_opt, W_opt = unpack_params(result.x)
                    
                    # Check if close to half-integers
                    # Round to nearest half-integer
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    # Verify rounded solution on M_{2,3,3}
                    err_rounded = 0.0
                    for i in range(n1):
                        for j in range(m1):
                            for k in range(p1):
                                val = np.sum(U_rounded[i*m1+j, :] * V_rounded[j*p1+k, :] * W_rounded[k*n1+i, :])
                                err_rounded += (val - 1.0) ** 2
                    
                    if err_rounded < 1e-9:
                        # Lift to M_{2,3,9}
                        # U_lift[i*m+j, c] = U_rounded[i*m1+j, c]
                        # V_lift[j*p+k, c] = V_rounded[j*m1+floor(k/3), c]  -- Wait, V indices for M_{2,3,3} are j*p1+k.
                        # For M_{2,3,9}, V indices are j*p+k. p=9.
                        # k in 0..8. k1 = floor(k/3). k2 = k%3.
                        # The tensor T[i,j,k] = T'[i,j,k1].
                        # Decomposition: sum_c U[i,j,c] V[j,k,c] W[i,k,c]
                        # We need U[i,j,c] V[j,k,c] W[i,k,c] to sum to T'[i,j,k1].
                        # If we set U_lift = U_rounded, V_lift[j,k,c] = V_rounded[j,k1,c], W_lift[i,k,c] = W_rounded[i,k1,c],
                        # then sum_c U_rounded[i,j,c] V_rounded[j,k1,c] W_rounded[i,k1,c] = T'[i,j,k1].
                        # This works.
                        
                        U_lift = np.zeros((n*m, rank))
                        V_lift = np.zeros((m*p, rank))
                        W_lift = np.zeros((p*n, rank))
                        
                        for i in range(n):
                            for j in range(m):
                                U_lift[i*m+j, :] = U_rounded[i*m1+j, :]
                                for k in range(p):
                                    k1 = k // 3
                                    V_lift[j*p+k, :] = V_rounded[j*m1+k1, :]
                                    W_lift[k*n+i, :] = W_rounded[k1*n1+i, :]
                        
                        # Final verify on M_{2,3,9}
                        if _verify(U_lift, V_lift, W_lift, n, m, p, tol=1e-9):
                            best_U = U_lift.tolist()
                            best_V = V_lift.tolist()
                            best_W = W_lift.tolist()
                            best_rank = rank
                            # Found a good one, break out
                            return {"U": best_U, "V": best_V, "W": best_W}
                            
            except Exception:
                continue
    
    U_out = best_U.tolist() if isinstance(best_U, np.ndarray) else best_U
    V_out = best_V.tolist() if isinstance(best_V, np.ndarray) else best_V
    W_out = best_W.tolist() if isinstance(best_W, np.ndarray) else best_W
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
