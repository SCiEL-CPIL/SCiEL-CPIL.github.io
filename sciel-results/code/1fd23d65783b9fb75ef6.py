import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target tensor T[i, k, j] = 1
    # Indices: i in 0..n-1, k in 0..p-1, j in 0..m-1
    # Note: Prompt says W[k*n+i, r]. So W index is k, i.
    # U index is i, j. V index is j, k.
    
    def compute_residual(params, R):
        # params is a flat vector of size 3 * R * (n*m + m*p + p*n)?
        # No, separate arrays are easier.
        # Let's pass U, V, W.
        pass

    # We will iterate over ranks
    best_result = None
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize with schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1

    # Try to find better ranks
    for R in range(24, best_rank):
        if time.time() > deadline:
            break
            
        # Try multiple restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Random small integers or schoolbook subset
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Heuristic init: Use schoolbook terms but combine?
            # Or random.
            # Let's try random initialization with small integers
            U[:] = np.random.randint(-2, 3, (n * m, R))
            V[:] = np.random.randint(-2, 3, (m * p, R))
            W[:] = np.random.randint(-2, 3, (p * n, R))
            
            # Objective function
            def objective(params):
                U_flat = params[0:R * n * m]
                V_flat = params[R * n * m:R * n * m + R * m * p]
                W_flat = params[R * n * m + R * m * p:]
                
                U = U_flat.reshape(R, n * m).T
                V = V_flat.reshape(R, m * p).T
                W = W_flat.reshape(R, p * n).T
                
                # Compute tensor
                # T[i, k, j] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
                # U[i, j, r] is U[j + i*m, r]? No, U row is i*m+j.
                # So U[i, j, r] = U[i*m + j, r].
                
                tensor = np.zeros((n, p, m))
                for r in range(R):
                    u_r = U[:, r] # size n*m
                    v_r = V[:, r] # size m*p
                    w_r = W[:, r] # size p*n
                    
                    # Reshape to matrices
                    u_mat = u_r.reshape(n, m)
                    v_mat = v_r.reshape(m, p)
                    w_mat = w_r.reshape(p, n)
                    
                    # Term is sum over r of u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                    # This is not a simple matrix product.
                    # We need to compute T[i, k, j] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                    
                    # Efficient computation?
                    # T[i, k, j] = sum_j u[i, j] v[j, k] w[k, i]
                    # For fixed i, k: sum_j u[i, j] v[j, k] is (u_mat * v_mat)[i, k].
                    # Then multiply by w[k, i].
                    # So T[:, :, :] for fixed r is (u_mat * v_mat) * w_mat.T?
                    # Let A_r = u_mat * v_mat. A_r is n x p.
                    # A_r[i, k] = sum_j u[i, j] v[j, k].
                    # Then T_r[i, k, j] = A_r[i, k] * w[k, i].
                    # Wait, w[k, i] is scalar for fixed i, k.
                    # So T_r[i, k, j] is independent of j?
                    # No, the sum is over r.
                    # T[i, k, j] = sum_r (u_mat_r[i, j] * v_mat_r[j, k] * w_mat_r[k, i])
                    # This is NOT sum_r ( (u_mat_r * v_mat_r)[i, k] * w_mat_r[k, i] )
                    # Because the sum over j is inside the product for each r?
                    # No, the tensor is T[i, k, j] = 1.
                    # The decomposition is T[i, k, j] = sum_r U[i, j, r] V[j, k, r] W[k, i, r].
                    # Here j is an index of U and V.
                    # So for a fixed r, the term is U_r[i, j] V_r[j, k] W_r[k, i].
                    # This term depends on j.
                    # So we cannot precompute u_mat * v_mat for all j at once unless we sum over j.
                    # But the target T[i, k, j] is 1 for all j.
                    # So we need sum_r U_r[i, j] V_r[j, k] W_r[k, i] = 1 for all i, k, j.
                    
                    # This is a system of equations.
                    # I will compute the residual by iterating over i, k, j.
                    
                    res = 0
                    for i in range(n):
                        for k in range(p):
                            for j in range(m):
                                val = 0
                                for r in range(R):
                                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                                res += (val - 1.0) ** 2
                    return res

            # This is too slow in Python.
            # I need a vectorized version.
            
            def objective_vec(params):
                U_flat = params[0:R * n * m]
                V_flat = params[R * n * m:R * n * m + R * m * p]
                W_flat = params[R * n * m + R * m * p:]
                
                U = U_flat.reshape(R, n * m).T
                V = V_flat.reshape(R, m * p).T
                W = W_flat.reshape(R, p * n).T
                
                # Compute T[i, k, j]
                # T[i, k, j] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
                # Let's compute for all r.
                # U is (n*m, R). V is (m*p, R). W is (p*n, R).
                
                # We can use einsum?
                # U[i, j, r] -> U_flat[r, i, j]
                # V[j, k, r] -> V_flat[r, j, k]
                # W[k, i, r] -> W_flat[r, k, i]
                
                U_reshaped = U.T.reshape(R, n, m)
                V_reshaped = V.T.reshape(R, m, p)
                W_reshaped = W.T.reshape(R, p, n)
                
                # Term_r[i, k, j] = U_reshaped[r, i, j] * V_reshaped[r, j, k] * W_reshaped[r, k, i]
                # Sum over r.
                
                # Use einsum: 'rij,rjk,rki->ikj'
                # Result shape (n, p, m)
                T_computed = np.einsum('rij,rjk,rki->ikj', U_reshaped, V_reshaped, W_reshaped)
                
                T_target = np.ones((n, p, m))
                return np.sum((T_computed - T_target) ** 2)

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Optimize
            res = minimize(objective_vec, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
            
            if res.success and res.fun < 1e-6:
                # Check for integer/half-integer
                U_sol = res.x[0:R * n * m].reshape(R, n * m).T
                V_sol = res.x[R * n * m:R * n * m + R * m * p].reshape(R, m * p).T
                W_sol = res.x[R * n * m + R * m * p:].reshape(R, p * n).T
                
                # Round to nearest 0.5
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_sol * 2) / 2
                
                # Verify
                U_r = U_round.T.reshape(R, n, m)
                V_r = V_round.T.reshape(R, m, p)
                W_r = W_round.T.reshape(R, p, n)
                T_check = np.einsum('rij,rjk,rki->ikj', U_r, V_r, W_r)
                
                if np.allclose(T_check, np.ones((n, p, m)), atol=1e-9):
                    # Valid solution
                    best_rank = R
                    best_U = U_round.tolist()
                    best_V = V_round.tolist()
                    best_W = W_round.tolist()
                    # Convert to required format
                    # U shape (n*m, R)
                    # U_round is (n*m, R)
                    break
        
        if best_rank == R:
            break # Found solution for this rank, no need to try higher? 
            # Actually we want minimal rank, so we try lower first.
            # If we found R, we can stop if we started from 24.
            # If we didn't find 24, we try 25, etc.

    if best_rank < n * m * p:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W} # Schoolbook
