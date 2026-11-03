import scipy
import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = 120  # seconds
    
    # Precompute indices for non-zero entries
    # T_{a, b, c} = 1 if a=im+j, b=jp+k, c=kn+i
    # a in [0, nm), b in [0, mp), c in [0, pn)
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_indices.append((a, b, c))
    
    total_entries = (n * m) * (m * p) * (p * n)
    
    # Function to compute tensor from U, V, W
    # U: (nm, r), V: (mp, r), W: (pn, r)
    # We want sum_c U[a,c] V[b,c] W[c_idx,c]
    # This is a tensor contraction.
    # For efficiency, we can compute the full tensor if r is small, 
    # but here r is ~40, dimensions 6, 21, 14. 
    # 6*21*14 = 1764 entries. 
    # Computing all is fast.
    
    def tensor_residuals(U, V, W):
        # U, V, W are flattened vectors
        r = len(U) // (n * m)
        U_mat = U.reshape(n * m, r)
        V_mat = V.reshape(m * p, r)
        W_mat = W.reshape(p * n, r)
        
        # Compute T[a, b, c] = sum_c U[a, c] V[b, c] W[c, c]
        # Using einsum: 'ac,bc,dc->abd'
        # But we only need to compare with target.
        # Target is 1 at nz_indices, 0 elsewhere.
        
        # Compute full tensor
        T = np.einsum('ac,bc,dc->abd', U_mat, V_mat, W_mat)
        
        # Flatten T
        T_flat = T.flatten()
        
        # Target vector
        # We can construct the target vector once
        # But it depends on indices.
        
        # Residuals:
        # For nz indices, T - 1
        # For others, T
        
        # It is faster to just compute the error on the nz indices and a random subset of zeros?
        # No, we need exact.
        # Let's compute the full error.
        
        # Create target array
        # This is expensive if done every iteration.
        # We can precompute the target array.
        
        return T_flat # We will compute error outside
    
    # Precompute target array
    target = np.zeros(total_entries)
    for a, b, c in nz_indices:
        target[a * (m * p) * (p * n) + b * (p * n) + c] = 1.0
        
    # We will try ranks from 41 down to 30
    # If we find a solution, we return it.
    
    # Helper to check if a solution is valid (integer/half-integer)
    def check_solution(U, V, W):
        # U, V, W are arrays of shape (dim, r)
        # Check if 2*U, 2*V, 2*W are close to integers
        # And if the tensor decomposition is exact.
        
        U_int = 2 * U
        V_int = 2 * V
        W_int = 2 * W
        
        if not np.allclose(U_int, np.round(U_int), atol=1e-6):
            return False
        if not np.allclose(V_int, np.round(V_int), atol=1e-6):
            return False
        if not np.allclose(W_int, np.round(W_int), atol=1e-6):
            return False
            
        # Round to integers
        U_r = np.round(U_int).astype(int)
        V_r = np.round(V_int).astype(int)
        W_r = np.round(W_int).astype(int)
        
        # Convert back to half-integers for verification
        U_h = U_r / 2.0
        V_h = V_r / 2.0
        W_h = W_r / 2.0
        
        # Verify tensor
        T = np.einsum('ac,bc,dc->abd', U_h, V_h, W_h)
        T_flat = T.flatten()
        
        if not np.allclose(T_flat, target, atol=1e-9):
            return False
            
        return True

    # Search
    for r in range(41, 29, -1):
        if time.time() - start_time > deadline:
            break
            
        # Try a few random restarts for each rank
        for restart in range(5):
            if time.time() - start_time > deadline:
                break
                
            # Initialize with small random integers
            # Scale by 2 to search for half-integers
            U_init = np.random.randint(-2, 3, size=(n * m, r))
            V_init = np.random.randint(-2, 3, size=(m * p, r))
            W_init = np.random.randint(-2, 3, size=(p * n, r))
            
            # Flatten
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Define residual function for least_squares
            def fun(x):
                U = x[:n*m*r].reshape(n*m, r)
                V = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W = x[n*m*r + m*p*r:].reshape(p*n, r)
                
                T = np.einsum('ac,bc,dc->abd', U, V, W)
                T_flat = T.flatten()
                
                return T_flat - target
            
            try:
                sol = scipy.optimize.least_squares(fun, x0, method='lm', max_nfev=1000)
                if sol.success or sol.cost < 1e-6:
                    # Extract solution
                    x_sol = sol.x
                    U_sol = x_sol[:n*m*r].reshape(n*m, r)
                    V_sol = x_sol[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W_sol = x_sol[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    if check_solution(U_sol, V_sol, W_sol):
                        # Convert to list of lists of fractions/floats
                        # The prompt asks for integer/half-integer entries.
                        # We can return floats that are exact halves.
                        U_out = U_sol.tolist()
                        V_out = V_sol.tolist()
                        W_out = W_sol.tolist()
                        
                        # Ensure they are exactly representable?
                        # The check_solution already rounded to integers for 2*U.
                        # So we should return the rounded versions / 2.
                        U_final = (np.round(2 * U_sol) / 2.0).tolist()
                        V_final = (np.round(2 * V_sol) / 2.0).tolist()
                        W_final = (np.round(2 * W_sol) / 2.0).tolist()
                        
                        return {"U": U_final, "V": V_final, "W": W_final}
            except Exception:
                pass

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
