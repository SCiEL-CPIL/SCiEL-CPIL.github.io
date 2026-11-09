import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # 10-second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid rank-r decomposition."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_rec, T, atol=tol)
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try to find a lower rank using gradient descent
    # We will try a few ranks below 63
    # Given the complexity, we might not find one, but it's worth a shot.
    # We use a time-boxed search.
    
    def try_gradient_descent(r, max_iters=500, lr=0.01, seed=42):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute the gradient of the loss
            # Loss = 0.5 * sum_{a,b,c} (T_rec[a,b,c] - T[a,b,c])^2
            
            # Compute T_rec
            T_rec = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            
            error = T_rec - T
            err_norm = np.max(np.abs(error))
            
            if err_norm < 1e-8:
                return (U, V, W)
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r_idx in range(r):
                # grad_U[:, r_idx] = sum_{b,c} error[:, b, c] * V[b, r_idx] * W[c, r_idx]
                grad_U[:, r_idx] = np.einsum('abc,b,c->a', error, V[:, r_idx], W[:, r_idx])
                grad_V[:, r_idx] = np.einsum('abc,a,c->b', error, U[:, r_idx], W[:, r_idx])
                grad_W[:, r_idx] = np.einsum('abc,a,b->c', error, U[:, r_idx], V[:, r_idx])
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
        # Final check
        if check_decomposition(U, V, W):
            return (U, V, W)
        
        return None
    
    # Try ranks from 62 down to 50
    # This is a heuristic; it may not find anything.
    # We limit the number of attempts to stay within time limits.
    ranks_to_try = [62, 60, 58, 56, 55]
    for r in ranks_to_try:
        if time.time() > deadline - 20: # Leave 20s for other checks
            break
        result = try_gradient_descent(r, max_iters=200, lr=0.05, seed=42)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
    
    U, V, W = best_result
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
