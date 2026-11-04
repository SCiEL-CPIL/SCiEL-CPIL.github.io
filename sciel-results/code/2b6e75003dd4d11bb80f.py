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
    
    # Naive decomposition
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
        for a in range(num_A):
            for b in range(num_B):
                for c in range(num_C):
                    val = np.dot(U[a], V[b] * W[c])
                    i = a // m
                    j = a % m
                    jp = b // p
                    k = b % p
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
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

    best_U, best_V, best_W = U_naive, V_naive, W_naive
    best_rank = 24
    
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
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    np.random.seed(42)
    
    def try_rank(target_rank, num_attempts=30):
        """Try to find a valid decomposition at the given rank."""
        size_U = num_A * target_rank
        size_V = num_B * target_rank
        size_W = num_C * target_rank
        total_size = size_U + size_V + size_W
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Different initialization strategies
            if attempt % 3 == 0:
                # Sparse random initialization
                params0 = np.zeros(total_size)
                for _ in range(total_size // 3):
                    idx = np.random.randint(0, total_size)
                    params0[idx] = np.random.choice([-1, 0.5, 1])
            elif attempt % 3 == 1:
                # Dense random with small values
                params0 = np.random.randn(total_size) * 0.5
            else:
                # Structured initialization: start from naive and randomly perturb
                params0 = np.random.randn(total_size) * 0.1
                # Add some structure
                for r in range(target_rank):
                    params0[r] = 1.0  # First column of U
                    params0[size_U + r] = 1.0  # First column of V
                    params0[size_U + size_V + r] = 1.0  # First column of W
            
            try:
                res = minimize(tensor_error, params0, args=(target_rank,), method='L-BFGS-B', 
                              jac=tensor_grad, options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
            except Exception:
                continue
                
            if res.fun < 1e-10:
                U_final = res.x[:size_U].reshape(num_A, target_rank)
                V_final = res.x[size_U:size_U+size_V].reshape(num_B, target_rank)
                W_final = res.x[size_U+size_V:].reshape(num_C, target_rank)
                
                if check_decomposition(U_final, V_final, W_final, target_rank):
                    return (U_final, V_final, W_final)
        
        return None
    
    # Try ranks from 23 down to 15
    for target_rank in range(23, 14, -1):
        if time.time() > deadline - 30:
            break
        
        result = try_rank(target_rank, num_attempts=40)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
