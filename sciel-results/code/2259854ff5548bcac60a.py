import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n×m×p matrix multiplication tensor."""
    rank = U.shape[1]
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Compute decomposition tensor
    T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_dec, T, atol=tol)

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _try_rank_search(n, m, p, target_rank, time_limit=10.0):
    """
    Try to find a decomposition of the given rank using random restarts and gradient descent.
    """
    start_time = time.time()
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Build target tensor
    T = np.zeros((n_m, m_p, p_n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    best_loss = float('inf')
    best_params = None
    
    # Try multiple restarts
    max_restarts = 50
    steps_per_restart = 200
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        U = np.random.randn(n_m, target_rank) * 0.5
        V = np.random.randn(m_p, target_rank) * 0.5
        W = np.random.randn(p_n, target_rank) * 0.5
        
        # Adam optimizer
        lr = 0.05
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        t = 0
        for step in range(steps_per_restart):
            T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = T_dec - T
            loss = np.sum(diff ** 2)
            
            if loss < 1e-10:
                return U, V, W, target_rank
            
            # Gradients
            dU = np.einsum('ijk,jr,kr->ir', diff, V, W)
            dV = np.einsum('ijk,ir,kr->jr', diff, U, W)
            dW = np.einsum('ijk,ir,jr->kr', diff, U, V)
            
            # Adam update
            t += 1
            b1 = 0.9
            b2 = 0.999
            eps = 1e-8
            
            m_U = b1 * m_U + (1 - b1) * dU
            v_U = b2 * v_U + (1 - b2) * (dU ** 2)
            m_U_hat = m_U / (1 - b1 ** t)
            v_U_hat = v_U / (1 - b2 ** t)
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            
            m_V = b1 * m_V + (1 - b1) * dV
            v_V = b2 * v_V + (1 - b2) * (dV ** 2)
            m_V_hat = m_V / (1 - b1 ** t)
            v_V_hat = v_V / (1 - b2 ** t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            
            m_W = b1 * m_W + (1 - b1) * dW
            v_W = b2 * v_W + (1 - b2) * (dW ** 2)
            m_W_hat = m_W / (1 - b1 ** t)
            v_W_hat = v_W / (1 - b2 ** t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            if loss < best_loss:
                best_loss = loss
                best_params = (U.copy(), V.copy(), W.copy())
        
        if best_loss < 1e-10:
            return best_params[0], best_params[1], best_params[2], target_rank

    if best_params and best_loss < 1e-10:
        return best_params[0], best_params[1], best_params[2], target_rank
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    best_solution = None
    best_rank = n * m * p  # 36
    
    # Try lower ranks
    # Ranks to try: 12, 14, 16, 18, 20, 24
    ranks_to_try = [12, 14, 16, 18, 20, 24]
    
    for R in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        # Allocate time based on rank (lower rank gets more time? or equal?)
        # Let's give each rank a fixed time slice
        time_slice = 10.0
        if R == 12:
            time_slice = 20.0 # Give more time to the most promising one
            
        result = _try_rank_search(n, m, p, R, time_limit=time_slice)
        
        if result is not None:
            U, V, W, rank = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_solution = (U, V, W, rank)
                best_rank = rank
                break # Found a valid one, stop searching for lower ranks? 
                      # Actually, we want the MINIMAL rank. 
                      # Since we iterate from low to high, the first valid one is the minimal found.
                      # But maybe a lower rank exists? We tried 12 first. If 12 works, it's likely minimal (or close).
                      # If 12 fails, we try 14, etc.
    
    # If no improvement found, fall back to schoolbook
    if best_solution is None:
        U, V, W, rank = _schoolbook(n, m, p)
        best_solution = (U, V, W, rank)
        best_rank = rank
    
    U, V, W, R = best_solution
    
    # Convert to lists for return
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
