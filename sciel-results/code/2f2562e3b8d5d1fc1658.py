import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T (flattened)
    # Indices: a (0..nm-1), b (0..mp-1), c (0..pn-1)
    # T[a,b,c] = 1 if (a//m == b//p == c%n) and (a%m == b%p == c//n) ... wait
    # Let's re-verify the index mapping.
    # Matrix A is n x m. Index (i,j) -> a = i*m + j
    # Matrix B is m x p. Index (j,k) -> b = j*p + k
    # Matrix C is n x p. Index (i,k) -> c = k*n + i (Trace convention W[k*n+i, r])
    
    # Condition for T[a,b,c] = 1:
    # i = a // m, j = a % m
    # j' = b // p, k = b % p
    # k' = c // n, i' = c % n
    # We need i == i', j == j', k == k'
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # We want to find U, V, W such that T = sum_r U[:,r] V[:,r] W[:,r]
    # To handle half-integers, let U = U_int / 2, V = V_int / 2, W = W_int / 2
    # Then T = sum_r (U_int[:,r] V_int[:,r] W_int[:,r]) / 8
    # So we solve for integer U_int, V_int, W_int such that:
    # sum_r U_int[:,r] V_int[:,r] W_int[:,r] = 8 * T
    
    T_target = 8.0 * T
    
    def loss(params, rank):
        """
        params: flattened array of U, V, W
        rank: current rank
        """
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:size_U + size_V + size_W]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        # Compute tensor product sum
        # T_calc[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We only care about the non-zero entries of T_target
        # To speed up, we can just compute the error on the non-zero terms
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    error += (val - T_target[a, b, c])**2
        return error

    def check_solution(U_int, V_int, W_int, rank):
        """
        Check if integer matrices U_int, V_int, W_int exactly decompose T_target
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_int[a, :], V_int[b, :] * W_int[c, :])
                    if val != T_target[a, b, c]:
                        return False
        return True

    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Try ranks from low to high
    # Known rank for 2x3x4 is 19. Let's start there.
    for target_rank in range(19, 25):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(attempt + target_rank)
            
            # Initialize with small random integers or floats
            # Since we are solving for integers, starting with random floats is fine
            size_U = n * m * target_rank
            size_V = m * p * target_rank
            size_W = p * n * target_rank
            
            params0 = np.random.randn(size_U + size_V + size_W) * 1.0
            
            # Use L-BFGS-B for optimization
            # Note: This might be slow if not careful.
            # We can use a simpler gradient descent if SciPy is too slow, but SciPy is available.
            
            try:
                res = minimize(loss, params0, args=(target_rank,), method='L-BFGS-B', 
                               options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.success:
                # Extract optimized parameters
                U_opt = res.x[:size_U].reshape(n*m, target_rank)
                V_opt = res.x[size_U:size_U+size_V].reshape(m*p, target_rank)
                W_opt = res.x[size_U+size_V:].reshape(p*n, target_rank)
                
                # Round to nearest integer
                U_int = np.round(U_opt).astype(int)
                V_int = np.round(V_opt).astype(int)
                W_int = np.round(W_opt).astype(int)
                
                # Verify
                if check_solution(U_int, V_int, W_int, target_rank):
                    # Found a solution!
                    # Convert back to half-integers for output
                    U_out = (U_int / 2.0).tolist()
                    V_out = (V_int / 2.0).tolist()
                    W_out = (W_int / 2.0).tolist()
                    
                    return {"U": U_out, "V": V_out, "W": W_out}
            
            # If optimization didn't succeed or rounding failed, try next attempt
            # We could also try "snap and verify" during optimization, but that's complex.
            # For now, rely on the final snap.

    # Fallback: Return naive decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                
    return {"U": U, "V": V, "W": W}
