import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    num_A = n * m
    num_B = m * p
    num_C = p * n
    
    # Pre-compute the indices for non-zero entries of the tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # Else T[a, b, c] = 0
    # We need sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    
    # List of (a, b, c, target_val) for all non-zero and some zero entries?
    # Actually, we need to check ALL entries.
    # There are num_A * num_B * num_C = 6 * 12 * 8 = 576 entries.
    # 24 of them are 1, 552 are 0.
    
    # Naive decomposition
    rank_naive = num_A * num_B * num_C # No, rank is n*m*p = 24.
    # Wait, the number of terms in the naive decomposition is n*m*p = 24.
    # Each term corresponds to a (i,j,k).
    # U[i*m+j, r] = 1 if r corresponds to (i,j,k)
    # V[j*p+k, r] = 1 if r corresponds to (i,j,k)
    # W[k*n+i, r] = 1 if r corresponds to (i,j,k)
    
    U_naive = np.zeros((num_A, 24))
    V_naive = np.zeros((num_B, 24))
    W_naive = np.zeros((num_C, 24))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m+j, idx] = 1
                V_naive[j*p+k, idx] = 1
                W_naive[k*n+i, idx] = 1
                idx += 1
                
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is valid for the full tensor."""
        # Check all 576 entries
        for a in range(num_A):
            for b in range(num_B):
                # Determine if (a,b) is a valid pair for some k?
                # a = i*m+j, b = j'*p+k
                # If j != j', then T[a,b,c] must be 0 for all c.
                # If j == j', then c must be k*n+i.
                # Let's just iterate all c.
                for c in range(num_C):
                    val = np.dot(U[a], V[b] * W[c]) # This is sum_r U[a,r] V[b,r] W[c,r]
                    # Determine target
                    # a = i*m+j
                    i = a // m
                    j = a % m
                    # b = j'*p+k
                    jp = b // p
                    k = b % p
                    # c = k'*n+i'
                    kp = c // n
                    i_c = c % n
                    
                    if j == jp and k == kp and i == i_c:
                        target = 1.0
                    else:
                        target = 0.0
                        
                    if abs(val - target) > 1e-9:
                        return False
        return True

    # Verify naive
    if not check_decomposition(U_naive, V_naive, W_naive, 24):
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()} # Should be valid

    best_U, best_V, best_W = U_naive, V_naive, W_naive
    best_rank = 24
    
    # Try to find lower rank
    # Strategy: Random initialization + Gradient Descent on the full tensor error
    # Error = sum_{a,b,c} (sum_r U[a,r] V[b,r] W[c,r] - T[a,b,c])^2
    
    # Pre-compute T as a vector of length 576
    T_flat = np.zeros(num_A * num_B * num_C)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx = a * (num_B * num_C) + b * num_C + c
                T_flat[idx] = 1.0

    def tensor_error(params, rank):
        # params is flattened U, V, W
        # U: num_A * rank, V: num_B * rank, W: num_C * rank
        size_U = num_A * rank
        size_V = num_B * rank
        size_W = num_C * rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(num_A, rank)
        V = V_flat.reshape(num_B, rank)
        W = W_flat.reshape(num_C, rank)
        
        # Compute tensor
        # T_calc[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is a 3-way tensor product.
        # We can compute it efficiently.
        # For each r, we add outer product U[:,r] V[:,r] W[:,r]
        # But we need to sum over r.
        # Let's just loop over a,b,c for simplicity in error calc, it's 576 * rank ops.
        # 576 * 24 is small.
        
        err = 0.0
        # We can vectorize the dot product part?
        # T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a 6x12x8 tensor. 576 elements.
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        T_calc_flat = T_calc.flatten()
        
        diff = T_calc_flat - T_flat
        return np.sum(diff**2)
    
    def tensor_grad(params, rank):
        size_U = num_A * rank
        size_V = num_B * rank
        size_W = num_C * rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(num_A, rank)
        V = V_flat.reshape(num_B, rank)
        W = W_flat.reshape(num_C, rank)
        
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_calc - T_flat.reshape(num_A, num_B, num_C)
        
        # Gradient of sum(diff^2) w.r.t U[a,r]
        # dE/dU[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
        # This can be done with einsum
        grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    np.random.seed(42)
    
    for target_rank in range(23, 10, -1):
        if time.time() > deadline - 20:
            break
            
        # Try a few random starts
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random small values
            size_U = num_A * target_rank
            size_V = num_B * target_rank
            size_W = num_C * target_rank
            total_size = size_U + size_V + size_W
            
            params0 = np.random.randn(total_size) * 0.1
            
            # Use L-BFGS-B
            try:
                res = minimize(tensor_error, params0, args=(target_rank,), method='L-BFGS-B', jac=tensor_grad, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            except Exception:
                continue
                
            if res.success and res.fun < 1e-9:
                # Check exactness
                U_final = res.x[:size_U].reshape(num_A, target_rank)
                V_final = res.x[size_U:size_U+size_V].reshape(num_B, target_rank)
                W_final = res.x[size_U+size_V:].reshape(num_C, target_rank)
                
                if check_decomposition(U_final, V_final, W_final, target_rank):
                    best_U, best_V, best_W = U_final, V_final, W_final
                    best_rank = target_rank
                    break # Found a lower rank, try even lower
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
