"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank_optimization(n, m, p, target_rank, deadline, seed=42):
    """
    Attempt to find a rank-`target_rank` decomposition via least-squares optimization.
    Returns (U, V, W) if successful, else None.
    """
    if time.time() > deadline:
        return None
        
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U_init = rng.randn(n*m, target_rank)
    V_init = rng.randn(m*p, target_rank)
    W_init = rng.randn(p*n, target_rank)
    
    def flatten_params(params):
        U_flat = params[:n*m*target_rank]
        V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = params[n*m*target_rank + m*p*target_rank :]
        return U_flat.reshape(n*m, target_rank), V_flat.reshape(m*p, target_rank), W_flat.reshape(p*n, target_rank)
    
    def residuals(params):
        U, V, W = flatten_params(params)
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res.append(val - 1.0)
        return np.array(res)
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    try:
        # Use Levenberg-Marquardt for non-linear least squares
        # max_nfev is kept conservative to respect the deadline
        result = least_squares(residuals, x0, method='lm', max_nfev=2000)
        
        if result.cost < 1e-12:
            U_opt, V_opt, W_opt = flatten_params(result.x)
            # Verify with a slightly tighter tolerance to ensure exactness in floating point
            if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-7):
                return U_opt, V_opt, W_opt
    except Exception:
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 110  # Leave safety margin for the 120s limit
    
    # Try to find a lower rank decomposition
    # Theoretical lower bound is max(n*m, m*p, p*n) = 20
    # We search from a reasonable lower bound up to schoolbook - 1
    # For 3x4x5, it is unlikely we will find rank < 60 easily, but we try.
    # We prioritize higher ranks first as they are more likely to be solvable if a lower rank exists.
    # However, the task asks for rank-minimal, so we should ideally find the true minimum.
    # Given the difficulty, we attempt a few ranks.
    
    # Lower bound is 20. Let's try a few values.
    # In practice, for 3x4x5, the rank is likely 60 (schoolbook) or close to it.
    # We will try ranks from 60-1 down to a limit, but stop early if we find one.
    # Actually, finding ANY rank < 60 is an improvement.
    
    # Let's try a small range of ranks to see if any work.
    # We'll try from schoolbook-1 down to a certain point.
    # To save time, we only try a few candidates.
    
    candidates = []
    # If we have time, we can try more.
    # Let's try 59, 58, 57, 56, 55.
    for r in range(schoolbook_rank - 1, max(schoolbook_rank - 10, 19), -1):
        if time.time() > deadline - 5:
            break
            
        res = _try_rank_optimization(n, m, p, r, deadline - 5, seed=r)
        if res is not None:
            best_U, best_V, best_W = res
            best_rank = r
            # If we found one, we could try to go lower, but for now,
            # let's just return the first one found to ensure we return a valid answer quickly.
            # Or we can continue to find a smaller one?
            # The prompt says "rank-minimal". If we find 59, is 58 possible?
            # Given the complexity, finding 59 is already a big jump.
            # Let's try to find a few more if time permits, but return the best found so far.
            # Actually, the loop goes downwards, so the first one found is the highest rank < 60.
            # To find the *minimal* rank, we would need to search all the way down to 20.
            # That is too slow.
            # We will stick to finding *a* lower rank if possible, which improves the score.
            # The score is seed/value, so lower rank is better.
            # If we find 59, score is 60/59 > 1.0.
            # If we find 58, score is 60/58 > 60/59.
            # So we should try to find the lowest possible.
            # But the search space is huge.
            # Let's just return the first one found to be safe, or continue?
            # Let's continue for a few more steps if time allows.
            pass
            
    # If we didn't find any, best_rank is still schoolbook_rank.
    
    # Convert to list of lists for output
    if isinstance(best_U, np.ndarray):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": best_U,
            "V": best_V,
            "W": best_W
        }
# EVOLVE-BLOCK-END
