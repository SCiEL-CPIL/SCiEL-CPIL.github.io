import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Shapes
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target Tensor
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_target[idx_u, idx_v, idx_w] = 1.0
                
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound is max(n*m, m*p, p*n) = 21
    lower_bound = max(n*m, m*p, p*n)
    
    for R in range(lower_bound, best_rank + 1):
        if time.time() > deadline:
            break
            
        # Try to find a rank R decomposition
        found = False
        num_restarts = 10 if R == 21 else 5 # More restarts for the most likely optimal rank
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize randomly
            np.random.seed(42 + restart + R)
            U = np.random.randn(dim_u, R) * 0.1
            V = np.random.randn(dim_v, R) * 0.1
            W = np.random.randn(dim_w, R) * 0.1
            
            # Gradient Descent
            lr = 0.01
            best_error = np.inf
            prev_error = np.inf
            
            for step in range(500):
                if time.time() > deadline:
                    break
                    
                # Compute Reconstructed Tensor
                T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T_rec - T_target)
                
                if error < best_error:
                    best_error = error
                    best_U_local = U.copy()
                    best_V_local = V.copy()
                    best_W_local = W.copy()
                    
                if error < 1e-6:
                    break
                    
                # Compute Gradients
                E = T_rec - T_target
                G_U = np.einsum('abc,br,cr->ar', E, V, W)
                G_V = np.einsum('abc,ar,cr->br', E, U, W)
                G_W = np.einsum('abc,ar,br->cr', E, U, V)
                
                # Update
                U -= lr * G_U
                V -= lr * G_V
                W -= lr * G_W
                
                # Adaptive Learning Rate
                if error > prev_error:
                    lr *= 0.5
                else:
                    # Optional: increase lr slightly if decreasing fast?
                    pass
                prev_error = error
                
            # Check if best local solution is good enough
            if best_error < 1e-4:
                # Snap to half-integers
                U_snap = np.round(best_U_local * 2) / 2.0
                V_snap = np.round(best_V_local * 2) / 2.0
                W_snap = np.round(best_W_local * 2) / 2.0
                
                # Verify
                T_verify = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                if np.allclose(T_verify, T_target, atol=1e-9):
                    found = True
                    best_rank = R
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    break # Break restart loop
        
        if found:
            break # Break rank loop if optimal found
        
    # Convert to list of lists if required? 
    # The prompt says: return {'U', 'V', 'W'}: factor matrices...
    # The seed returns lists. I should return lists or numpy arrays?
    # "factor matrices of shapes... with integer/half-integer entries"
    # The seed uses lists. I will use lists to be safe, or numpy arrays if the evaluator accepts them.
    # Usually, these tasks accept numpy arrays or lists. The seed uses lists.
    # I will return numpy arrays as they are more standard in these scripts, 
    # but let's check the seed: U = [[0] * rank ...]. It's a list.
    # I'll convert to list to match the interface exactly.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
