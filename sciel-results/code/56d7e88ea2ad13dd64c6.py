import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The rank is the number of columns in these matrices.
    Uses a randomized search to find a low-rank decomposition, falling back to schoolbook.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find lower rank using block decomposition
    # Split p into 3+2+2 and use Strassen-like decompositions
    
    # For M_{3,3,3}, use a rank-19 decomposition (known to exist)
    # For M_{3,3,2}, use a rank-10 decomposition (schoolbook is 18, but we can do better)
    
    # Actually, let's just use schoolbook for each block
    # M_{3,3,3}: rank 27
    # M_{3,3,2}: rank 18
    # M_{3,3,2}: rank 18
    # Total: 63, same as before
    
    # So no improvement from block decomposition with schoolbook
    
    # Try a randomized search for rank 58-62
    def compute_loss(U, V, W):
        """Compute the sum of squared errors."""
        loss = 0.0
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    loss += (val - 1.0) ** 2
        return loss
    
    def try_rank(rank, n_restarts=3, n_steps=20):
        """Try to find a rank-`rank` decomposition using random search."""
        if time.time() > deadline:
            return None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization with small values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Simple random walk
            for step in range(n_steps):
                if time.time() > deadline:
                    return None
                
                loss_old = compute_loss(U, V, W)
                
                # Random perturbation
                U_new = U + np.random.randn(n * m, rank) * 0.05
                V_new = V + np.random.randn(m * p, rank) * 0.05
                W_new = W + np.random.randn(p * n, rank) * 0.05
                
                loss_new = compute_loss(U_new, V_new, W_new)
                
                if loss_new < loss_old:
                    U, V, W = U_new, V_new, W_new
            
            # Check if the solution is exact
            loss_final = compute_loss(U, V, W)
            if loss_final < 1e-10:
                return (U, V, W)
        
        return None
    
    # Try ranks from 58 to 62
    for rank in range(58, 63):
        if time.time() > deadline:
            break
        
        result = try_rank(rank, n_restarts=2, n_steps=10)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank
            break
    
    return best_result
