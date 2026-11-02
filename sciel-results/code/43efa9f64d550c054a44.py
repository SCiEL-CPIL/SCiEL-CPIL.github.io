import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Indices:
    # U: (n*m, R) -> index i*m + j
    # V: (m*p, R) -> index j*p + k
    # W: (p*n, R) -> index k*n + i  (Note: prompt says W[k*n+i, r])
    
    # Target Tensor T[u, v, w]
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # T[u, v, w] = 1 if indices are consistent, else 0
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # We only need to check the non-zero entries of T for the objective,
    # but the zero entries also constrain the solution.
    # However, checking all 2304 entries is fast with numpy.
    
    # Precompute the list of (u, v, w) indices where T is 1
    target_indices = []
    target_values = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_indices.append((u, v, w))
                target_values.append(1.0)
                
    # We also need to ensure that for all other (u,v,w), the sum is 0.
    # This is equivalent to minimizing the full tensor difference.
    
    def tensor_contraction(U, V, W):
        # U: (size_u, R), V: (size_v, R), W: (size_w, R)
        # Result: (size_u, size_v, size_w)
        # This is expensive if R is large.
        # We can compute it as:
        # C = U @ V.T -> (size_u, size_v) for each r? No.
        # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # This is a 3-mode tensor.
        # We can compute it by iterating over r?
        # Or use einsum.
        # np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.einsum('ur,vr,wr->uvw', U, V, W)

    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        R = len(params) // (size_u + size_v + size_w)
        if R == 0: return 1e10
        
        U = params[:size_u * R].reshape(size_u, R)
        V = params[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
        W = params[size_u * R + size_v * R :].reshape(size_w, R)
        
        T = tensor_contraction(U, V, W)
        
        # We want T to match the target tensor.
        # Target tensor is mostly zeros.
        # We can just compute the squared error on the target indices and a random sample of zero indices?
        # No, to be safe, we should minimize the full error.
        # But creating a full target tensor is 2304 floats, which is fine.
        
        # Let's create the target tensor once outside?
        # It's small enough to create inside or cache it.
        # Let's cache it.
        
        return np.sum((T - TARGET_TENSOR)**2)

    # Initialize Target Tensor
    TARGET_TENSOR = np.zeros((size_u, size_v, size_w))
    for u, v, w in target_indices:
        TARGET_TENSOR[u, v, w] = 1.0

    best_solution = None
    best_rank = n * m * p # 48
    best_params = None
    
    # Schoolbook solution as fallback
    U_school = np.zeros((size_u, best_rank))
    V_school = np.zeros((size_v, best_rank))
    W_school = np.zeros((size_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Try to find better ranks
    # Start from 47 down to 40
    for R in range(47, 39, -1):
        if time.time() > deadline - 10:
            break
            
        # Initialize parameters
        # Random initialization
        np.random.seed(42 + R)
        params = np.random.randn(size_u * R + size_v * R + size_w * R) * 0.1
        
        # Use L-BFGS-B
        # This might be slow. Let's try a few iterations.
        try:
            res = minimize(objective, params, method='L-BFGS-B', 
                           options={'maxiter': 50, 'maxfun': 1000})
            
            if res.success or res.fun < 1e-8:
                # Check if it's a valid solution
                R_opt = R
                U_opt = res.x[:size_u * R_opt].reshape(size_u, R_opt)
                V_opt = res.x[size_u * R_opt : size_u * R_opt + size_v * R_opt].reshape(size_v, R_opt)
                W_opt = res.x[size_u * R_opt + size_v * R_opt :].reshape(size_w, R_opt)
                
                # Snap to half-integers?
                # The task asks for integer/half-integer entries.
                # If the solution is close to half-integers, we can snap them.
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify snapped solution
                T_snap = tensor_contraction(U_snap, V_snap, W_snap)
                if np.allclose(T_snap, TARGET_TENSOR, atol=1e-9):
                    best_rank = R
                    best_solution = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    break # Found a better solution, stop searching lower ranks? 
                         # Actually, we want the MINIMAL rank. 
                         # We are iterating downwards, so the first one we find is the best (lowest) in this range.
                         # If we want to be sure, we could continue, but time is limited.
                         # Since we start from 47, if we find 47, it's better than 48.
                         # If we find 46, it's even better.
                         # We break because we found a valid one and we are going down. 
                         # Wait, if we break, we might miss 46 if we started at 47 and found it? 
                         # No, if we find 47, we return it. 
                         # If we want the absolute minimum, we should not break until we are sure 46 doesn't exist? 
                         # But finding 46 is harder. 
                         # Given the time, finding any R < 48 is a win. 
                         # Let's just return the first one we find.
                         
        except Exception:
            pass

    if best_solution:
        return best_solution
    else:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

# EVOLVE-BLOCK-END
