import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T
    # T[ (i*m+j), (j*p+k), (k*n+i) ] = 1
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    # We split the problem into two parts:
    # 1. k in [0, p-2] (here p=4, so k in [0, 2]) -> 3x3x3 part
    # 2. k = p-1 (here k=3) -> 3x3x1 part
    
    # Part 2: k = 3
    # This part is independent and has rank 9.
    # We can fix these 9 columns.
    R2 = n * m # 9
    U2 = np.zeros((n*m, R2))
    V2 = np.zeros((m*p, R2))
    W2 = np.zeros((p*n, R2))
    
    col = 0
    for i in range(n):
        for j in range(m):
            # k is fixed to p-1 = 3
            k = p - 1
            U2[i*m + j, col] = 1.0
            V2[j*p + k, col] = 1.0
            W2[k*n + i, col] = 1.0
            col += 1
            
    # Part 1: k in [0, 2]
    # We need to find a decomposition of rank R1 for the sub-tensor T1.
    # T1 has non-zero entries only where k < p-1.
    # Specifically, T1[ i*m+j, j*p+k, k*n+i ] = 1 for k in [0, 2].
    # All other entries in T1 are 0.
    # Note: The W indices for k in [0,2] are 0..8.
    # The W indices for k=3 are 9..11.
    # So T1 and T2 have disjoint W supports.
    
    # We will try to find R1 starting from 23 up to 27.
    # R1 = 27 is the schoolbook rank for 3x3x3.
    
    best_U1 = None
    best_V1 = None
    best_W1 = None
    best_R1 = n * m * (p-1) # 27
    
    start_time = time.time()
    time_limit = 110 # seconds
    
    for R1 in range(23, best_R1 + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a decomposition of rank R1
        # Variables: U1 (n*m x R1), V1 (m*p x R1), W1 (p*n x R1)
        # But W1 only needs to be non-zero for the first n*(p-1) rows?
        # Actually, W1 can be full (p*n x R1), but the target T1 has 0s in the last n rows.
        # So we can constrain W1 to be zero in the last n rows to reduce variables?
        # Or just let it be, the optimizer will drive it to 0.
        
        # To speed up, we can fix the structure? No.
        # We use random initialization.
        
        found = False
        max_restarts = 50 if R1 == 23 else 20
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U1, V1, W1
            # Scale initialization
            U1 = np.random.randn(n*m, R1) * 0.5
            V1 = np.random.randn(m*p, R1) * 0.5
            W1 = np.random.randn(p*n, R1) * 0.5
            
            # We want to match T1.
            # T1 is a tensor of shape (n*m, m*p, p*n).
            # We flatten the residual.
            
            def residual(params):
                # Reshape params back to U1, V1, W1
                size_U = n*m * R1
                size_V = m*p * R1
                size_W = p*n * R1
                
                U1_flat = params[:size_U]
                V1_flat = params[size_U:size_U+size_V]
                W1_flat = params[size_U+size_V:size_U+size_V+size_W]
                
                U1 = U1_flat.reshape((n*m, R1))
                V1 = V1_flat.reshape((m*p, R1))
                W1 = W1_flat.reshape((p*n, R1))
                
                # Compute tensor sum
                # T_calc[i,j,k] = sum_r U1[i,r] * V1[j,r] * W1[k,r]
                # Efficient computation using einsum
                # U1 shape (n*m, R1)
                # V1 shape (m*p, R1)
                # W1 shape (p*n, R1)
                
                # We only care about the non-zero entries of T1.
                # Indices:
                # i in 0..n-1
                # j in 0..m-1
                # k in 0..p-2
                # U_idx = i*m + j
                # V_idx = j*p + k
                # W_idx = k*n + i
                
                # We can compute the full tensor or just the required entries.
                # Full tensor is 9x12x12 = 1296 entries.
                # Required entries are 3x3x3 = 27.
                
                # Let's compute the required entries directly.
                res = np.zeros(27)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p-1):
                            u_val = U1[i*m + j]
                            v_val = V1[j*p + k]
                            w_val = W1[k*n + i]
                            # Sum over r
                            val = np.sum(u_val * v_val * w_val)
                            res[idx] = val - 1.0 # Target is 1
                            idx += 1
                            
                # Also, we need to ensure that the other entries are 0?
                # The tensor T1 has 0s elsewhere.
                # If we don't constrain them, the solution might have non-zero values there.
                # But we only need the decomposition to match T1 exactly.
                # T1 is defined as the multiplication tensor for k<p-1.
                # The other entries of the full tensor T are handled by Part 2 or are 0.
                # Wait, the full tensor T has non-zero entries for ALL k.
                # Part 2 handles k=p-1.
                # Part 1 handles k<p-1.
                # The entries of T1 for k<p-1 are 1.
                # The entries of T1 for k=p-1 are 0 (because Part 2 handles them).
                # So we only need to match the 27 entries.
                # The other entries of the "Part 1" tensor can be anything, as long as they don't interfere?
                # No, the decomposition is a sum of rank-1 tensors.
                # The total tensor is Sum(Part 1) + Sum(Part 2).
                # Sum(Part 2) is zero for k<p-1.
                # So Sum(Part 1) must be zero for k=p-1?
                # No, Sum(Part 1) is a tensor.
                # We need Sum(Part 1) + Sum(Part 2) = T.
                # For k<p-1, T has 1s. Sum(Part 2) is 0. So Sum(Part 1) must be 1.
                # For k=p-1, T has 1s. Sum(Part 1) must be 0.
                # So we MUST constrain Part 1 to be 0 for k=p-1.
                
                # Let's add residuals for the k=p-1 entries of Part 1.
                # There are n*m*(p-1) = 3*3*3 = 27 such entries?
                # No, for k=p-1, the W index is (p-1)*n + i.
                # The U and V indices are the same.
                # So we need U1[i*m+j] * V1[j*p+p-1] * W1[(p-1)*n+i] = 0.
                
                # This adds 27 more constraints.
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p-1): # Wait, k=p-1 is the only one
                            pass # This loop is wrong.
                            
                # Correctly:
                # For k = p-1:
                for i in range(n):
                    for j in range(m):
                        u_val = U1[i*m + j]
                        v_val = V1[j*p + (p-1)]
                        w_val = W1[(p-1)*n + i]
                        val = np.sum(u_val * v_val * w_val)
                        res = np.concatenate([res, [val]]) # Target is 0
                        
                return res
                
            x0 = np.concatenate([U1.flatten(), V1.flatten(), W1.flatten()])
            
            try:
                sol = least_squares(residual, x0, method='lm', max_nfev=1000)
                if sol.cost < 1e-12:
                    # Found a valid decomposition
                    size_U = n*m * R1
                    size_V = m*p * R1
                    U1_sol = sol.x[:size_U].reshape((n*m, R1))
                    V1_sol = sol.x[size_U:size_U+size_V].reshape((m*p, R1))
                    W1_sol = sol.x[size_U+size_V:].reshape((p*n, R1))
                    
                    best_U1 = U1_sol
                    best_V1 = V1_sol
                    best_W1 = W1_sol
                    best_R1 = R1
                    found = True
                    break
            except Exception:
                pass
                
        if found:
            break
            
    # Construct the final result
    if best_U1 is None:
        # Fallback to schoolbook
        R_total = n * m * p
        U = np.zeros((n*m, R_total))
        V = np.zeros((m*p, R_total))
        W = np.zeros((p*n, R_total))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    else:
        R_total = best_R1 + R2
        U = np.hstack([best_U1, U2])
        V = np.hstack([best_V1, V2])
        W = np.hstack([best_W1, W2])
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
