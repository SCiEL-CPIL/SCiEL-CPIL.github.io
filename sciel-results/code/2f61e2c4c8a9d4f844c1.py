"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k] = 1."""
    # The tensor for matrix mult (n x m) * (m x p) -> (n x p)
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    return n, m, p

def _residual_max(U, V, W, n, m, p):
    """Compute max |residual| over all (i,j,k) entries."""
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k])
                # Wait, that's not right. The tensor contraction is:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # This is an element-wise product summed over r, not a dot product of two vectors.
                # Let me fix: it's sum_r (U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r])
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                res = abs(val - 1.0)
                if res > max_res:
                    max_res = res
    return max_res

def _residual_max_grad(U, V, W, n, m, p):
    """Compute max |residual| and its gradient w.r.t. U, V, W."""
    rank = U.shape[1]
    max_res = 0.0
    max_i, max_j, max_k = 0, 0, 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                res = abs(val - 1.0)
                if res > max_res:
                    max_res = res
                    max_i, max_j, max_k = i, j, k
    
    i, j, k = max_i, max_j, max_k
    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
    sign = 1.0 if val >= 1.0 else -1.0
    
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    dU[i*m+j] = sign * V[j*p+k] * W[k*n+i]
    dV[j*p+k] = sign * U[i*m+j] * W[k*n+i]
    dW[k*n+i] = sign * U[i*m+j] * V[j*p+k]
    
    return max_res, dU, dV, dW

def _try_rank(rank, n, m, p, max_iter=5000, lr_init=0.01, seed=0, timeout=None):
    """Try to find an exact decomposition of given rank via gradient descent."""
    rng = np.random.RandomState(seed)
    
    # Initialize: start from schoolbook and randomly project to lower rank
    # Or just random init
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    # Scale to get closer to target
    # Each term contributes ~ U*V*W, we want sum to be 1
    # If each entry is ~0.5, product is ~0.125, sum of rank terms ~ rank*0.125
    # Scale so that typical residual is small
    scale = (1.0 / (rank * 0.125)) ** (1.0/3.0)
    U *= scale
    V *= scale
    W *= scale
    
    lr = lr_init
    best_res = _residual_max(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start = time.time()
    for it in range(max_iter):
        if timeout is not None and time.time() - start > timeout:
            break
        
        res, dU, dV, dW = _residual_max_grad(U, V, W, n, m, p)
        
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Gradient descent
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        if it % 500 == 0 and it > 0:
            lr *= 0.5
        
        if best_res < 1e-10:
            break
    
    return best_res, best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    deadline = 140  # safety margin from 150s limit
    
    start_time = time.time()
    
    # Try to find lower rank decompositions
    # Start from rank-1 down to some minimum
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    min_possible = max(n*m, m*p, p*n)
    
    best_rank = schoolbook_rank
    best_U = best_V = best_W = None
    
    # First, get schoolbook as fallback
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from schoolbook-1 down to min_possible
    # But that's too many. Let's try a few key ranks.
    ranks_to_try = list(range(schoolbook_rank - 1, min_possible - 1, -1))
    
    for rank in ranks_to_try:
        if time.time() - start_time > deadline:
            break
        
        # Try multiple seeds for this rank
        for seed in range(5):
            if time.time() - start_time > deadline:
                break
            
            timeout_for_this = min(15.0, deadline - (time.time() - start_time))
            if timeout_for_this < 2:
                break
            
            res, U, V, W = _try_rank(rank, n, m, p, max_iter=3000, lr_init=0.05, seed=seed, timeout=timeout_for_this)
            
            if res < 1e-8:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break  # Found exact for this rank, try lower
        
        if best_rank < schoolbook_rank:
            # Found improvement, keep trying lower
            continue
    
    # Convert to list format
    rank_out = best_U.shape[1]
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
