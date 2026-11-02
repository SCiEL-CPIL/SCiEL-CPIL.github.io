import numpy as np
import time
import random

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # Hardcoded rank 11 solution as fallback
    rank_fallback = 11
    U_fallback = [[0] * rank_fallback for _ in range(n * m)]
    V_fallback = [[0] * rank_fallback for _ in range(m * p)]
    W_fallback = [[0] * rank_fallback for _ in range(p * n)]
    
    # ... (copy from current program) ...
    # Strassen part for columns 0 and 1 of B
    U_fallback[0][0] = 1; U_fallback[3][0] = 1
    V_fallback[0][0] = 1; V_fallback[4][0] = 1
    U_fallback[2][1] = 1; U_fallback[3][1] = 1
    V_fallback[0][1] = 1
    U_fallback[0][2] = 1
    V_fallback[1][2] = 1; V_fallback[4][2] = -1
    U_fallback[3][3] = 1
    V_fallback[3][3] = 1; V_fallback[0][3] = -1
    U_fallback[0][4] = 1; U_fallback[1][4] = 1
    V_fallback[4][4] = 1
    U_fallback[2][5] = 1; U_fallback[0][5] = -1
    V_fallback[0][5] = 1; V_fallback[1][5] = 1
    U_fallback[1][6] = 1; U_fallback[3][6] = -1
    V_fallback[3][6] = 1; V_fallback[4][6] = 1
    
    W_fallback[0][0] = 1; W_fallback[0][3] = 1; W_fallback[0][4] = -1; W_fallback[0][6] = 1
    W_fallback[2][2] = 1; W_fallback[2][4] = 1
    W_fallback[1][1] = 1; W_fallback[1][3] = 1
    W_fallback[3][0] = 1; W_fallback[3][1] = -1; W_fallback[3][2] = 1; W_fallback[3][5] = 1
    
    U_fallback[0][7] = 1
    V_fallback[2][7] = 1
    U_fallback[1][8] = 1
    V_fallback[5][8] = 1
    U_fallback[2][9] = 1
    V_fallback[2][9] = 1
    U_fallback[3][10] = 1
    V_fallback[5][10] = 1
    
    W_fallback[4][7] = 1; W_fallback[4][8] = 1
    W_fallback[5][9] = 1; W_fallback[5][10] = 1
    
    # Try to find rank 10
    start_time = time.time()
    deadline = start_time + 110 # Leave 10s margin
    
    # Tensor non-zero indices
    nonzeros = []
    for k in range(m):
        for i in range(n):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                nonzeros.append((u_idx, v_idx, w_idx))
    
    # All possible indices for U, V, W
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    # Total number of elements in the tensor space
    # We only care about the nonzeros and the zeros.
    # Zeros are all other combinations.
    # To be efficient, we can just check the nonzeros and a random sample of zeros, 
    # but for exactness we must ensure all zeros are 0.
    # However, if we minimize the sum of squares over ALL entries, it's expensive.
    # But U_size*V_size*W_size = 4*6*6 = 144. It's small.
    
    all_indices = []
    for u in range(U_size):
        for v in range(V_size):
            for w in range(W_size):
                all_indices.append((u, v, w))
    
    nonzero_set = set(nonzeros)
    
    def evaluate_and_round(params, rank):
        # params is a flat array of size (U_size + V_size + W_size) * rank
        U_flat = params[:U_size * rank].reshape(U_size, rank)
        V_flat = params[U_size * rank : U_size * rank + V_size * rank].reshape(V_size, rank)
        W_flat = params[U_size * rank + V_size * rank :].reshape(W_size, rank)
        
        # Calculate tensor
        T = np.zeros((U_size, V_size, W_size))
        for r in range(rank):
            T += np.einsum('i,j,k->ijk', U_flat[:, r], V_flat[:, r], W_flat[:, r])
            
        # Check error
        error = 0.0
        for u, v, w in all_indices:
            target = 1.0 if (u, v, w) in nonzero_set else 0.0
            error += (T[u, v, w] - target) ** 2
            
        if error < 1e-9:
            # Round to nearest half-integer
            U_rounded = np.round(U_flat * 2) / 2.0
            V_rounded = np.round(V_flat * 2) / 2.0
            W_rounded = np.round(W_flat * 2) / 2.0
            
            # Verify exact match with rounded values
            T_rounded = np.zeros((U_size, V_size, W_size))
            for r in range(rank):
                T_rounded += np.einsum('i,j,k->ijk', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                
            valid = True
            for u, v, w in all_indices:
                target = 1.0 if (u, v, w) in nonzero_set else 0.0
                if abs(T_rounded[u, v, w] - target) > 1e-9:
                    valid = False
                    break
            
            if valid:
                return True, U_rounded, V_rounded, W_rounded
        
        return False, None, None, None

    # Search
    best_result = None
    rank = 10
    
    # Random restarts
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Initialize with small random values
        # Scale matters. Let's try 1.0
        params = np.random.randn((U_size + V_size + W_size) * rank)
        
        # Use scipy minimize
        from scipy.optimize import minimize
        
        def objective(p):
            U_f = p[:U_size * rank].reshape(U_size, rank)
            V_f = p[U_size * rank : U_size * rank + V_size * rank].reshape(V_size, rank)
            W_f = p[U_size * rank + V_size * rank :].reshape(W_size, rank)
            
            T = np.zeros((U_size, V_size, W_size))
            for r in range(rank):
                T += np.einsum('i,j,k->ijk', U_f[:, r], V_f[:, r], W_f[:, r])
                
            # Compute loss
            # Only nonzeros need to be 1, zeros need to be 0.
            # We can vectorize this.
            # Create target tensor
            Target = np.zeros((U_size, V_size, W_size))
            for u, v, w in nonzeros:
                Target[u, v, w] = 1.0
                
            return np.sum((T - Target) ** 2)
            
        res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
        
        if res.success or res.fun < 1e-9:
            found, U_r, V_r, W_r = evaluate_and_round(res.x, rank)
            if found:
                best_result = (U_r, V_r, W_r)
                break

    if best_result:
        U_r, V_r, W_r = best_result
        # Convert to list of lists of floats/ints
        U_out = [[float(x) for x in row] for row in U_r]
        V_out = [[float(x) for x in row] for row in V_r]
        W_out = [[float(x) for x in row] for row in W_r]
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
