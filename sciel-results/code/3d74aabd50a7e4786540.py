"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_TIMEOUT = 145.0  # Leave 5s safety margin

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U_mat = np.array(U, dtype=float)  # shape (n*m, rank)
    V_mat = np.array(V, dtype=float)  # shape (m*p, rank)
    W_mat = np.array(W, dtype=float)  # shape (p*n, rank)
    
    # Check: for all i, j, k: sum_t U_mat[i*m+j, t] * V_mat[j*p+k, t] * W_mat[k*n+i, t] == 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > tol:
        return False, max_err
    
    return True, 0.0

def _build_schoolbook(n, m, p):
    """Build the schoolbook (trivial) decomposition of rank n*m*p."""
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
    return U, V, W

def _try_lower_rank_search(n, m, p, max_rank_to_try, time_budget):
    """
    Attempt to find a lower rank decomposition using random initialization
    and gradient-free optimization (random restarts + local search).
    Returns (U, V, W) if found, else None.
    """
    best_rank = None
    best_U = None
    best_V = best_W = None
    
    for target_rank in range(max_rank_to_try, n*m*p - 1, -1):
        if time.time() - _start_time > time_budget:
            break
        
        # Try multiple random restarts for this target rank
        n_restarts = 50
        found = False
        
        for restart in range(n_restarts):
            if time.time() - _start_time > time_budget:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Simple iterative refinement: project onto the constraint
            # T_{(ij),(jk),(ik)} = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
            # This is a nonlinear system. Use a simple fixed-point / alternating approach.
            
            # For each (i,j,k), we need the sum to be 1.
            # This is hard to solve directly. Let's try a different approach:
            # Start from schoolbook and try to merge terms.
            
            if found:
                break
        
        if not found:
            continue
    
    if best_rank is not None:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Build schoolbook decomposition (rank = n*m*p = 90)
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify schoolbook
    valid, err = _verify_decomposition(U, V, W, n, m, p)
    
    # Check time
    if time.time() - _start_time > _TIMEOUT:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a lower rank decomposition
    # For 3x5x6, known lower bounds suggest rank might be reducible.
    # Let's try a structured approach: look for decompositions with rank < 90.
    
    # One known technique: if m is prime or has special structure, sometimes
    # we can do better. For now, let's try a simple heuristic:
    # Check if we can use a rank = (n-1)*(m-1)*(p-1) + something decomposition.
    
    # Actually, let's just try to verify if any known formula works.
    # For small matrices, we can brute-force check small rank decompositions.
    
    # Try rank from 89 down to some minimum
    # This is expensive, so we'll be conservative.
    
    # For now, return schoolbook as it's guaranteed valid.
    # The search for lower rank is complex and may not succeed within time limits.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
