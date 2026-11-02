"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition exactly."""
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Generate schoolbook decomposition of rank n*m*p."""
    R = n * m * p
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_rank(R, n, m, p, max_time=10.0, seed=42):
    """Try to find a rank-R decomposition using structured initialization + refinement."""
    start = time.time()
    random.seed(seed)
    np.random.seed(seed)
    
    # Strategy: use the cyclic structure
    # For rank R, try to find U (n*m x R), V (m*p x R), W (p*n x R)
    # such that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Approach: exploit symmetry. Try R = n*m = 12 first.
    # Initialize with a structured pattern.
    
    if R < 1:
        return None
    
    # Try multiple random initializations
    best_loss = float('inf')
    best_U = best_V = best_W = None
    
    for attempt in range(50):
        if time.time() - start > max_time:
            break
        
        # Random initialization with some structure
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        # Add some structure: ensure each row has at least one non-zero
        for r in range(R):
            idx_u = random.randint(0, n*m-1)
            idx_v = random.randint(0, m*p-1)
            idx_w = random.randint(0, p*n-1)
            U[idx_u, r] = 1.0
            V[idx_v, r] = 1.0
            W[idx_w, r] = 1.0
        
        # Gradient descent
        lr = 0.05
        for it in range(200):
            if time.time() - start > max_time:
                break
            
            # Compute loss and gradients
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ui = i*m+j
                        vj = j*p+k
                        wk = k*n+i
                        s = np.dot(U[ui], V[vj] * W[wk])
                        err = s - 1.0
                        loss += err*err
                        grad_U[ui] += 2*err*V[vj]*W[wk]
                        grad_V[vj] += 2*err*U[ui]*W[wk]
                        grad_W[wk] += 2*err*U[ui]*V[vj]
            
            # Update with learning rate decay
            lr_eff = lr * (0.99 ** it)
            U -= lr_eff * grad_U
            V -= lr_eff * grad_V
            W -= lr_eff * grad_W
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if best_loss < 1e-10:
                break
        
        if best_loss < 1e-8:
            break
    
    if best_loss > 1e-6:
        return None
    
    # Round to nearest half-integer
    U_r = np.round(best_U * 2) / 2
    V_r = np.round(best_V * 2) / 2
    W_r = np.round(best_W * 2) / 2
    
    # Verify rounded version
    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # Try without rounding
    if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 4 matrix.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize with schoolbook
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = rank_schoolbook
    
    # Try to find lower-rank decompositions
    # Start from plausible low ranks
    for target_rank in range(12, rank_schoolbook + 1):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank(target_rank, n, m, p, max_time=8.0, seed=target_rank)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
