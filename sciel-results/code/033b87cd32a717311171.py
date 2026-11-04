import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def _build_target_tensor(n, m, p):
    """
    Construct the target multiplication tensor.
    T[i, j, k, l, k2, i2] is not quite right.
    The tensor lives in V* (dim nm) x W* (dim mp) x Z (dim pn).
    Indices:
    u = i * m + j  (0..nm-1)
    v = j * p + k  (0..mp-1)
    w = k * n + i  (0..pn-1)
    
    The component T[u, v, w] is 1 if:
    i from u == i from w
    j from u == j from v
    k from v == k from w
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """
    Verify the decomposition.
    U: (nm, R)
    V: (mp, R)
    W: (pn, R)
    """
    R = U.shape[1]
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    # T = sum_s U[:, s] outer V[:, s] outer W[:, s]
    # This is a sum of rank-1 tensors.
    # We can use einsum: 'ar, br, cr -> abc'
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check against target
    # Target has 1s at specific positions, 0 elsewhere.
    # We can just check the max absolute error.
    # But we need to ensure exactness for the 1s and 0s.
    # Since we might have numerical noise, we check if the entries are close to 0 or 1.
    
    # The target tensor is sparse.
    # Let's construct the target again or check specific indices.
    # It's faster to check the specific non-zero indices and a sample of zeros.
    # But for safety, check all.
    
    # Actually, the prompt's _verify only checks the 1s.
    # I will implement a strict check.
    
    # Create target
    Target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                Target[u, v, w] = 1.0
                
    err = np.max(np.abs(T - Target))
    return err < 1e-6

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_school[u, col] = 1
                V_school[v, col] = 1
                W_school[w, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Target tensor for error calculation
    T_target = _build_target_tensor(n, m, p)
    
    # Try to find lower rank
    # Use random initialization and gradient descent
    # We will try ranks from 41 down to 35
    
    for target_rank in range(41, 34, -1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random restarts for this rank
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + restart)
            
            # Initialize with random values
            # Scale to roughly match the magnitude
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Simple gradient descent
            # Error E = || T_recon - T_target ||_F^2
            # dE/dU = 2 * sum_s (T_recon - T_target) * V_s * W_s (contracted)
            # This is expensive.
            
            # Alternative: Use a fixed-point iteration or simpler search?
            # Given the time limit, a full gradient descent might be too slow for 45x15x15 tensor.
            # 2025 elements.
            # Each step:
            # T_recon = einsum(U, V, W) -> O(R * nm * mp * pn) = 41 * 9 * 15 * 15 = 84150 ops.
            # Gradient calculation is similar.
            # 1000 steps * 84k ops = 84M ops.
            # NumPy is fast, this should be doable in < 1s per restart.
            
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                    
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_recon - T_target
                
                # Gradient w.r.t U
                # dE/dU[a, r] = sum_{b,c} 2 * diff[a,b,c] * V[b,r] * W[c,r]
                # We can compute this using einsum
                # G_U = 2 * np.einsum('abc, br, cr -> ar', diff, V, W)
                G_U = np.einsum('abc,br,cr->ar', diff, V, W)
                G_V = np.einsum('abc,ar,cr->br', diff, U, W)
                G_W = np.einsum('abc,ar,br->cr', diff, U, V)
                
                U -= lr * G_U
                V -= lr * G_V
                W -= lr * G_W
                
                if step % 100 == 0:
                    err = np.linalg.norm(diff)
                    if err < 1e-4:
                        break
                        
            # Check if converged
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T_target)
            
            if err < 1e-4:
                # Try to round to half-integers
                # Scale? The solution might be scaled.
                # Check if we can find a scaling factor such that entries are close to half-integers.
                # Or just round and verify.
                
                # Simple rounding
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                if _verify(U_round, V_round, W_round, n, m, p):
                    best_U, best_V, best_W = U_round, V_round, W_round
                    best_rank = target_rank
                    break # Found a valid one for this rank, try lower?
                    # Actually, if we found one, we can break the restart loop and try lower rank.
        
        if best_rank == target_rank:
            # We found a decomposition for this rank, try to go lower
            continue
        else:
            # If we didn't find one for this rank, maybe we should try lower?
            # But if we fail at 41, we might fail at 40.
            # However, the search is stochastic.
            # Let's just proceed to the next lower rank in the loop.
            pass

    # Convert to lists for output
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
