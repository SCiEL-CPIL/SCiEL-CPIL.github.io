import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Build the target tensor T
    # T is sparse. We store non-zero indices and values.
    # T[alpha, beta, gamma] = 1
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    non_zero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                non_zero.append((alpha, beta, gamma, 1.0))
    
    # We will search for rank r
    # Known lower bound is 7, but 10 is a strong candidate.
    # We will try r = 10, 11, 12... up to 18
    # If we find an exact solution, we return it.
    
    best_result = None
    best_rank = 18
    
    # Fallback: Schoolbook
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a lower rank solution
    # We use a simple gradient descent or random search with local optimization.
    # Given the small size, we can use scipy.optimize.
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        minimize = None

    def error_func(params, r):
        # params: flattened U, V, W
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        
        U = params[:size_u].reshape(n*m, r)
        V = params[size_u:size_u+size_v].reshape(m*p, r)
        W = params[size_u+size_v:].reshape(p*n, r)
        
        err = 0.0
        for (alpha, beta, gamma, val) in non_zero:
            # Sum over c
            s = 0.0
            for c in range(r):
                s += U[alpha, c] * V[beta, c] * W[gamma, c]
            err += (s - val)**2
        
        # We also need to ensure that for zero entries, the sum is 0?
        # The tensor is defined by these non-zero entries.
        # The decomposition must match the tensor exactly.
        # The tensor is 0 everywhere else.
        # So we must also minimize error on zero entries.
        # There are (n*m * m*p * p*n) - 18 zero entries.
        # This is 6*9*6 - 18 = 324 - 18 = 306 entries.
        # We can't iterate all in Python loop efficiently inside gradient descent if called many times.
        # We can sample zero entries or use a batch.
        
        # For now, let's just minimize non-zero error and hope the zero errors stay small?
        # No, that's dangerous.
        
        # Let's add a penalty for large values to keep it stable.
        return err

    # Better approach: Use the fact that the tensor is very sparse.
    # We can use a specific solver or random restarts with local search.
    
    # Let's try a few random initializations for rank 10.
    rng = np.random.default_rng(42)
    
    for r in range(10, 18):
        if time.time() > deadline - 10:
            break
            
        found = False
        for attempt in range(50): # 50 attempts per rank
            if time.time() > deadline - 5:
                break
                
            # Initialize
            U = rng.normal(size=(n*m, r))
            V = rng.normal(size=(m*p, r))
            W = rng.normal(size=(p*n, r))
            
            # Use simple gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                    
                # Compute gradients
                # dE/dU[alpha, c] = 2 * sum_{beta,gamma} (S - T) V[beta,c] W[gamma,c]
                # This is expensive to compute exactly for all zero entries.
                # We will only enforce the 18 non-zero constraints strictly and 
                # add a regularization term to keep U,V,W small, hoping zeros are satisfied.
                # Or we can check the zero constraints periodically.
                
                # Let's compute error on non-zero entries only for gradient.
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # Precompute V and W parts for efficiency?
                # For each non-zero (alpha, beta, gamma):
                # term = U[alpha, :] * V[beta, :] * W[gamma, :]
                # This is not quite right.
                # S = sum_c U[alpha,c] V[beta,c] W[gamma,c]
                
                # Gradient wrt U[alpha, c]:
                # dS/dc = V[beta,c] W[gamma,c]
                
                for (alpha, beta, gamma, val) in non_zero:
                    S = np.dot(U[alpha, :], np.multiply(V[beta, :], W[gamma, :]))
                    diff = S - val
                    # grad_U[alpha, :] += 2 * diff * (V[beta, :] * W[gamma, :])
                    grad_U[alpha, :] += 2 * diff * (V[beta, :] * W[gamma, :])
                    grad_V[beta, :] += 2 * diff * (U[alpha, :] * W[gamma, :])
                    grad_W[gamma, :] += 2 * diff * (U[alpha, :] * V[beta, :])
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Periodic check on zero entries
                if step % 20 == 0:
                    # Check a random subset of zero entries
                    # Total indices: 6*9*6 = 324
                    # Non-zero: 18
                    # We check 10 random zeros
                    for _ in range(10):
                        a = rng.integers(0, n*m)
                        b = rng.integers(0, m*p)
                        g = rng.integers(0, p*n)
                        # Check if this is a non-zero entry
                        is_nz = False
                        for (na, nb, ng, nv) in non_zero:
                            if na == a and nb == b and ng == g:
                                is_nz = True
                                break
                        if not is_nz:
                            S = np.dot(U[a, :], np.multiply(V[b, :], W[g, :]))
                            if abs(S) > 0.1:
                                # Penalty
                                # Add gradient to push S to 0
                                # dS/dU[a,c] = V[b,c] W[g,c]
                                grad_U[a, :] += 2 * S * (V[b, :] * W[g, :])
                                grad_V[b, :] += 2 * S * (U[a, :] * W[g, :])
                                grad_W[g, :] += 2 * S * (U[a, :] * V[b, :])
                                U -= lr * grad_U
                                V -= lr * grad_V
                                W -= lr * grad_W
                                break # Break inner loop? No, just one penalty step.
                
                # Check convergence
                S_nonzero = 0.0
                for (alpha, beta, gamma, val) in non_zero:
                    S = np.dot(U[alpha, :], np.multiply(V[beta, :], W[gamma, :]))
                    S_nonzero += (S - val)**2
                
                if S_nonzero < 1e-6:
                    # Check all zero entries
                    all_zero_ok = True
                    for a in range(n*m):
                        for b in range(m*p):
                            for g in range(p*n):
                                # Skip non-zero
                                is_nz = False
                                for (na, nb, ng, nv) in non_zero:
                                    if na == a and nb == b and ng == g:
                                        is_nz = True
                                        break
                                if is_nz:
                                    continue
                                S = np.dot(U[a, :], np.multiply(V[b, :], W[g, :]))
                                if abs(S) > 1e-4:
                                    all_zero_ok = False
                                    break
                            if not all_zero_ok:
                                break
                        if not all_zero_ok:
                            break
                    
                    if all_zero_ok:
                        # Round to half-integers
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        
                        # Verify exactness with rounded values
                        valid = True
                        for (alpha, beta, gamma, val) in non_zero:
                            S = np.dot(U_r[alpha, :], np.multiply(V_r[beta, :], W_r[gamma, :]))
                            if abs(S - val) > 1e-9:
                                valid = False
                                break
                        if valid:
                            for a in range(n*m):
                                for b in range(m*p):
                                    for g in range(p*n):
                                        is_nz = False
                                        for (na, nb, ng, nv) in non_zero:
                                            if na == a and nb == b and ng == g:
                                                is_nz = True
                                                break
                                        if is_nz:
                                            continue
                                        S = np.dot(U_r[a, :], np.multiply(V_r[b, :], W_r[g, :]))
                                        if abs(S) > 1e-9:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            
                            if valid:
                                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                                best_rank = r
                                found = True
                                break # Break attempt loop
                            
            if found:
                break
        
        if found:
            break

    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
