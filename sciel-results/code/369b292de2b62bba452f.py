import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication.
    Returns True if decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _try_rank_decomposition(rank, max_time=10):
    """Try to find a rank-r decomposition using vectorized numerical optimization.
    Returns (U, V, W) as lists of lists if found, else None."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    np.random.seed(42 + rank)
    
    best_residual = float('inf')
    best_solution = None
    
    # We will try a few random restarts
    num_restarts = 5
    
    for attempt in range(num_restarts):
        if time.time() - start_time > max_time:
            break
        
        # Initialize
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Reshape for einsum
        # U: (n, m, rank)
        # V: (m, p, rank)
        # W: (p, n, rank) -> indices k, i
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        lr = 0.01
        prev_res = float('inf')
        
        for iteration in range(1000):
            if time.time() - start_time > max_time:
                break
            
            # Compute S_ijk = sum_r U_ijk V_jkr W_kir
            # U_3d: (i, j, r)
            # V_3d: (j, k, r)
            # W_3d: (k, i, r)
            # S_ijk = einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
            
            Err = S - 1.0
            residual = np.sum(Err * Err)
            
            if residual < 1e-12:
                best_residual = residual
                best_solution = (U.copy(), V.copy(), W.copy())
                break
                
            if residual < best_residual:
                best_residual = residual
                best_solution = (U.copy(), V.copy(), W.copy())
                
            # Compute Gradients
            # dU_ijk = 2 * sum_k Err_ijk * V_jkr * W_kir
            # dV_jkr = 2 * sum_i Err_ijk * U_ijk * W_kir
            # dW_kir = 2 * sum_j Err_ijk * U_ijk * V_jkr
            
            dU = 2.0 * np.einsum('ijk,jkr,kir->ijr', Err, V_3d, W_3d, optimize=True)
            dV = 2.0 * np.einsum('ijk,ijr,kir->jkr', Err, U_3d, W_3d, optimize=True)
            dW = 2.0 * np.einsum('ijk,ijr,jkr->kir', Err, U_3d, V_3d, optimize=True)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Update 3d views (no copy needed if we just reshape again? 
            # U is flat, U_3d is a view. Modifying U modifies U_3d.
            # But we created U_3d from U. So U_3d is a view of U.
            # So U_3d is updated automatically.
            
            # Simple learning rate decay
            if iteration % 100 == 0 and iteration > 0:
                lr *= 0.9
                
            if residual > prev_res * 1.5: # If stuck/diverging, break early
                 # Try a smaller step or break
                 if iteration > 50:
                     break
            prev_res = residual
            
        if best_residual < 1e-12:
            break
    
    if best_solution is not None and best_residual < 1e-10:
        U, V, W = best_solution
        # Convert to lists
        U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
        V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
        W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
        
        # Verify with exact arithmetic check
        if _verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try rounding to nearest 0.5
        U_r = [[round(U[i, c] * 2) / 2 for c in range(rank)] for i in range(n*m)]
        V_r = [[round(V[i, c] * 2) / 2 for c in range(rank)] for i in range(m*p)]
        W_r = [[round(W[i, c] * 2) / 2 for c in range(rank)] for i in range(p*n)]
        
        if _verify(U_r, V_r, W_r):
            return {"U": U_r, "V": V_r, "W": W_r}
            
        # Try rounding to nearest 0.25
        U_r = [[round(U[i, c] * 4) / 4 for c in range(rank)] for i in range(n*m)]
        V_r = [[round(V[i, c] * 4) / 4 for c in range(rank)] for i in range(m*p)]
        W_r = [[round(W[i, c] * 4) / 4 for c in range(rank)] for i in range(p*n)]
        
        if _verify(U_r, V_r, W_r):
            return {"U": U_r, "V": V_r, "W": W_r}
            
        # Try rounding to nearest integer
        U_r = [[round(U[i, c]) for c in range(rank)] for i in range(n*m)]
        V_r = [[round(V[i, c]) for c in range(rank)] for i in range(m*p)]
        W_r = [[round(W[i, c]) for c in range(rank)] for i in range(p*n)]
        
        if _verify(U_r, V_r, W_r):
            return {"U": U_r, "V": V_r, "W": W_r}

    return None

def construct():
    overall_deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Try numerical search for various ranks, starting from low
    # Rank 23 is the theoretical minimum over reals. 
    # It might not have a half-integer solution, but we'll try.
    ranks_to_try = [23, 24, 25, 26]
    
    for rank in ranks_to_try:
        if time.time() > overall_deadline - 5:
            break
        remaining = overall_deadline - time.time() - 5
        if remaining < 2:
            break
        max_time = min(10, remaining)
        result = _try_rank_decomposition(rank, max_time=max_time)
        if result is not None:
            return result
    
    # Fallback to schoolbook (Rank 27)
    return _construct_schoolbook()
