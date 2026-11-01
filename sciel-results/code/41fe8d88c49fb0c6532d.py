"""
Rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication.

Strategy:
Instead of the deterministic schoolbook construction, we employ a numerical optimization
approach. We formulate the tensor decomposition as a minimization problem for the
Frobenius norm of the difference between the matrix multiplication tensor and the
sum of $r$ rank-1 tensors. We attempt to find an exact (or near-exact, rounded to
half-integers) decomposition for ranks $r$ from the theoretical lower bound up to
39. If a valid decomposition is found (error < 1e-6 after rounding), we return it.
Otherwise, we fall back to the known rank-40 schoolbook decomposition.

This changes the design family from "Constructive/Combinatorial" to "Numerical Optimization".
"""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Returns the rank-minimal tensor decomposition for 2x4 by 4x5 matrix multiplication.
    Attempts to find a rank < 40 decomposition via numerical optimization.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    N, M, P = n, m, p
    tensor_shape = (n, m, p)
    
    # Construct the target matrix multiplication tensor T
    # T[i, j, k] = 1 if we are computing element (i, k) of the product, 
    # which depends on A[i, j] * B[j, k].
    # In the decomposition U (n*m x r), V (m*p x r), W (p*n x r):
    # The tensor T is defined by T[i, j, k] = delta_{i,k} ? No.
    # The product C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T has entries T[i, j, k] such that the coefficient of A[i, j] B[j, k] is 1 if it contributes to C[i, k].
    # So T[i, j, k] = 1 for all i, j, k? No.
    # The term A[i, j] B[j, k] contributes to C[i, k].
    # In the factorization: C[i, k] = sum_r U[i, r] V[k, r] W[r]? No, the shapes are U(n*m, r), V(m*p, r), W(p*n, r).
    # Let's map indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # The product sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 if this term is part of the multiplication.
    # For a fixed (i, j, k), the term A[i, j] B[j, k] appears in C[i, k] exactly once.
    # So the target tensor T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p).
    # Wait, this is the "all ones" tensor?
    # Let's check the schoolbook:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c = i*m*p + j*p + k.
    # Sum over c: For a specific (i, j, k), only the term where c matches (i, j, k) is non-zero in all three?
    # No, U[i*m+j, c] is non-zero only for c corresponding to j? No, c is unique for (i, j, k).
    # So for a fixed (i, j, k), the sum is 1.
    # For a different tuple (i', j', k'), if (i', j', k') != (i, j, k), then at least one of the indices doesn't match the column c.
    # So yes, the target tensor T[i, j, k] = 1 for all i, j, k.
    
    T = np.ones((n, m, p), dtype=float)
    
    def flatten_indices(i, j, k):
        return i * m * p + j * p + k

    # We will try ranks from lower_bound to 39
    # Lower bound is max(n*m, m*p, p*n) = 20.
    # We iterate downwards from 39 to 20 to find the *minimal* rank?
    # Or upwards from 20? Upwards is better because if rank 20 works, we stop.
    # But rank 20 is very unlikely. Rank 39 is also unlikely.
    # We will try a few specific ranks or a range.
    
    best_result = None
    best_rank = n * m * p # 40
    
    # Try ranks from 20 to 39
    # To save time, we can limit the number of restarts per rank.
    max_restarts_per_rank = 5
    max_iterations = 2000
    
    for r in range(20, 40):
        if time.time() > deadline:
            break
            
        found = False
        for restart in range(max_restarts_per_rank):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W randomly
            # U: (n*m, r), V: (m*p, r), W: (p*n, r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Flatten U, V, W into a single vector for optimization
            # x = [U.flatten(), V.flatten(), W.flatten()]
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Objective function: sum of (T[i,j,k] - sum_r U[i,j,r] V[j,k,r] W[k,i,r])^2
            # Note: W index is k*n + i.
            
            def objective(x):
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r + m*p*r :].reshape(p*n, r)
                
                error = 0.0
                # Efficient computation:
                # For each r, compute outer product of U[:, r] and V[:, r], then contract with W[:, r]?
                # T_approx[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # Vectorized approach:
                # Expand U, V, W to full tensor shape?
                # U_exp[i, j, r] = U[i*m+j, r]
                # V_exp[j, k, r] = V[j*p+k, r]
                # W_exp[k, i, r] = W[k*n+i, r]
                
                # T_approx = sum_r U_exp[i,j,r] * V_exp[j,k,r] * W_exp[k,i,r]
                
                # This is a 4th order tensor contraction.
                # Let's do it loop-based for simplicity in the objective, as dimensions are small.
                # n=2, m=4, p=5. Total elements 40.
                # r is up to 39.
                
                # Reshape for easier indexing
                U_reshaped = U_flat.reshape(n, m, r)
                V_reshaped = V_flat.reshape(m, p, r)
                W_reshaped = W_flat.reshape(p, n, r)
                
                # Compute T_approx
                # T_approx[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
                
                # Use einsum
                # U: (i, j, r), V: (j, k, r), W: (k, i, r)
                # We want (i, j, k)
                # Term: U[i,j,r] * V[j,k,r] * W[k,i,r]
                # Sum over r.
                
                # Step 1: Contract U and V over r? No, r is the sum index.
                # We can sum over r first:
                # S[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                
                # Let's compute it in steps to avoid large intermediate tensors if possible, 
                # but 2x4x5x39 is small.
                
                # T_approx = np.einsum('ijr,jkr,ki r->ijk', U_reshaped, V_reshaped, W_reshaped)
                # Note: W is (k, i, r). So 'kir'.
                
                T_approx = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
                
                diff = T_approx - T
                return np.sum(diff * diff)
            
            # Gradient descent is slow. Let's use a simple coordinate descent or just check if random init is close?
            # No, random init is far.
            # We need an optimizer. SciPy is available.
            try:
                from scipy.optimize import minimize
            except ImportError:
                # Fallback to simple gradient descent if scipy is not available (though it is)
                # For now, assume scipy is available.
                pass
            
            # Use L-BFGS-B
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iterations, 'disp': False})
                
                if res.success or res.fun < 1e-6:
                    # Check if the solution can be rounded to integers/half-integers
                    U_sol = res.x[:n*m*r].reshape(n*m, r)
                    V_sol = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W_sol = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    # Verify exactness with rounded values
                    U_r_reshaped = U_rounded.reshape(n, m, r)
                    V_r_reshaped = V_rounded.reshape(m, p, r)
                    W_r_reshaped = W_rounded.reshape(p, n, r)
                    
                    T_approx_r = np.einsum('ijr,jkr,kir->ijk', U_r_reshaped, V_r_reshaped, W_r_reshaped)
                    
                    if np.allclose(T_approx_r, T, atol=1e-9):
                        # Found a valid decomposition of rank r
                        best_result = {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        best_rank = r
                        found = True
                        break
            except Exception as e:
                # Optimization failed
                continue
                
        if found:
            break
            
    if best_result is not None:
        return best_result
    else:
        # Fallback to schoolbook rank 40
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
                    
        U_list = [[int(val) if val == int(val) else val for val in row] for row in U.tolist()]
        V_list = [[int(val) if val == int(val) else val for val in row] for row in V.tolist()]
        W_list = [[int(val) if val == int(val) else val for val in row] for row in W.tolist()]
        
        return {"U": U_list, "V": V_list, "W": W_list}
