"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    
    # Build the true tensor T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0

    def verify(U, V, W):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T_true[a,b,c] for all a,b,c."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(T_recon, T_true, atol=1e-9)

    def round_to_half_int(x):
        """Round to nearest multiple of 0.5"""
        return np.round(2 * x) / 2.0

    def try_rank(rank, max_iters=50):
        """Attempt to find a valid decomposition of given rank."""
        if time.time() > deadline:
            return None
            
        # Initialize with random small values
        np.random.seed(42)
        U = np.random.randn(size_a, rank) * 0.1
        V = np.random.randn(size_b, rank) * 0.1
        W = np.random.randn(size_c, rank) * 0.1
        
        # Objective function: minimize sum of squared errors
        def objective(params):
            if time.time() > deadline:
                return float('inf')
            U_flat = params[:size_a * rank].reshape(size_a, rank)
            V_flat = params[size_a * rank : size_a * rank + size_b * rank].reshape(size_b, rank)
            W_flat = params[size_a * rank + size_b * rank :].reshape(size_c, rank)
            T_recon = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat, optimize=True)
            err = T_recon - T_true
            return np.sum(err ** 2)
        
        # We will try a few random restarts if the first one fails
        best_result = None
        best_score = float('inf')
        
        for restart in range(max_iters):
            if time.time() > deadline:
                break
                
            # Random initialization
            U_init = np.random.randn(size_a, rank) * 0.5
            V_init = np.random.randn(size_b, rank) * 0.5
            W_init = np.random.randn(size_c, rank) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use L-BFGS-B for large problems, it's efficient
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            except Exception:
                continue
                
            if res.fun < best_score:
                best_score = res.fun
                best_result = res
                
            # Check if we found a very good solution
            if best_score < 1e-6:
                break
                
        if best_result is None:
            return None
            
        # Extract best parameters
        U_opt = best_result.x[:size_a * rank].reshape(size_a, rank)
        V_opt = best_result.x[size_a * rank : size_a * rank + size_b * rank].reshape(size_b, rank)
        W_opt = best_result.x[size_a * rank + size_b * rank :].reshape(size_c, rank)
        
        # Try to snap to half-integers
        U_rounded = round_to_half_int(U_opt)
        V_rounded = round_to_half_int(V_opt)
        W_rounded = round_to_half_int(W_opt)
        
        if verify(U_rounded, V_rounded, W_rounded):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
            
        # If rounding didn't work, try standard integers
        U_int = np.round(U_opt)
        V_int = np.round(V_opt)
        W_int = np.round(W_opt)
        if verify(U_int, V_int, W_int):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
            
        return None

    # Try rank 28 first (theoretical lower bound for this structure might be higher, but let's try)
    # Actually, for 2x2x14, rank 28 is the "strassen-like" bound if it existed.
    # Let's try rank 28, then 32, 36, 40, 44, 48, 52, 56.
    # Given the time constraint, we should focus on the most promising ranks.
    # Rank 28 is the most likely candidate for improvement over 56 if a solution exists.
    
    result = try_rank(28, max_iters=10)
    if result:
        return {"U": result[0], "V": result[1], "W": result[2]}
        
    # If rank 28 fails, try slightly higher ranks in case the numerical landscape is tricky
    # but usually if 28 fails, 56 is the safe bet. 
    # Let's try 32 just in case, but keep it quick.
    if time.time() < deadline - 10:
        result = try_rank(32, max_iters=5)
        if result:
            return {"U": result[0], "V": result[1], "W": result[2]}

    # Fallback to Schoolbook Rank 56
    rank = n * m * p
    U = np.zeros((size_a, rank), dtype=int)
    V = np.zeros((size_b, rank), dtype=int)
    W = np.zeros((size_c, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
