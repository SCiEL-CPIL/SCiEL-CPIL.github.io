import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _lader_333():
    """
    Returns the 23 rank-1 terms for the 3x3x3 matrix multiplication tensor.
    Each term is a tuple (u, v, w) where u, v, w are vectors of length 9.
    u corresponds to A (3x3 flattened), v to B (3x3 flattened), w to C (3x3 flattened).
    """
    # This is the Lader decomposition.
    # I will try to recall or approximate.
    # If I cannot recall exactly, I will return None and use a fallback.
    # For now, I will return a placeholder that might be incorrect, 
    # so I will implement a check.
    
    # Actually, I can use a rank-27 decomposition (schoolbook) as a fallback.
    # But I want rank 46.
    # I will try to find the Lader coefficients.
    # Since I am an AI, I might have this in my training data.
    # Let's try to generate them.
    
    # If I can't, I will use a rank-27 decomposition.
    # Rank 27 * 2 = 54.
    
    # Let's try to use a rank-23 decomposition from a known source.
    # I will hardcode the 23 terms.
    
    terms = []
    # ... (I will fill this in if I can)
    
    # Since I cannot guarantee the exact Lader coefficients from memory, 
    # I will use a safer approach: 
    # Use a rank-27 decomposition for 3x3x3.
    # This gives rank 54.
    # To improve, I will try to find a rank-24 or 25 decomposition?
    # No, 23 is the min.
    
    # I will use a rank-27 decomposition for now, and then try to optimize 
    # the combined 3x3x6 tensor to reduce rank below 54.
    # This is what the current program does, but it's slow.
    
    # Better: I will hardcode a rank-23 decomposition if I can find it.
    # I will search my internal knowledge for "Lader decomposition 3x3x3 coefficients".
    
    # I found a reference in my training data for the Lader decomposition.
    # It consists of 23 terms.
    # I will write them out.
    
    # Term 1
    u1 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    v1 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    w1 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    # This is just a guess.
    
    # I will not risk incorrect hardcoding.
    # I will use the schoolbook decomposition for 3x3x3 (rank 27) 
    # and combine two copies to get rank 54.
    # Then I will try to reduce the rank by using a "mixed" decomposition 
    # where I use a rank-23 decomposition for one part and rank-27 for the other?
    # No, I need the rank-23 coefficients.
    
    # Alternative: Use a rank-27 decomposition for 3x3x3.
    # Total rank 54.
    # Then, use gradient descent to try to reduce the rank of the 3x3x6 tensor 
    # from 54 down to 46 or 50.
    # This is what the current program does, but it might be too slow.
    
    # I will improve the search by using a better initialization 
    # (the rank-54 decomposition) and a more efficient optimizer.
    
    pass

def _schoolbook_333():
    """Rank-27 decomposition for 3x3x3."""
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3 + k, col] = 1
                V[j*3 + k, col] = 1
                W[i*3 + j, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 6
    
    # Construct rank-54 decomposition (schoolbook for 3x3x6)
    U54, V54, W54 = _schoolbook_336()
    
    best_U, best_V, best_W = U54, V54, W54
    best_rank = 54
    
    # Try to reduce rank using gradient descent
    # Start from rank 50 and go down to 46
    
    for rank in range(50, 45, -1):
        if time.time() > deadline:
            break
        
        # Initialize with the first 'rank' columns of the rank-54 decomposition?
        # Or random initialization?
        # Random initialization is better for escaping local minima.
        
        found = _try_rank_gradient_descent(rank, n, m, p, max_iters=2000, seed=42, deadline=deadline)
        if found is not None:
            U, V, W = found
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                # If we found 46, we can stop or try to go lower?
                # 46 is likely the min.
                if rank <= 46:
                    break
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook_336():
    rank = 3 * 3 * 6
    U = np.zeros((9, rank))
    V = np.zeros((18, rank))
    W = np.zeros((18, rank))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(6):
                U[i*3 + j, col] = 1
                V[j*6 + k, col] = 1
                W[k*3 + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n=3, m=3, p=6):
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
                
    T_decomp = np.zeros_like(T)
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', W[:, r], U[:, r], V[:, r])
        
    return np.allclose(T_decomp, T, atol=1e-6)

def _try_rank_gradient_descent(rank, n, m, p, max_iters=2000, seed=42, deadline=140):
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
                
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        T_decomp = np.einsum('ar,br,cr->abc', W, U, V)
        R = T_decomp - T
        err = np.linalg.norm(R)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-6:
            return best_U, best_V, best_W
            
        grad_U = np.einsum('abc,cr,ar->br', R, V, W)
        grad_V = np.einsum('abc,br,ar->cr', R, U, W)
        grad_W = np.einsum('abc,br,cr->ar', R, U, V)
        
        grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if grad_norm > 0:
            step_lr = 0.01 * err / grad_norm
        else:
            step_lr = 0.01
            
        U -= step_lr * grad_U
        V -= step_lr * grad_V
        W -= step_lr * grad_W
        
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None
