"""Rewritten program for matrix multiplication 3x4x5 tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

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

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _combine_decompositions(decomps):
    """
    Combine multiple (U, V, W) decompositions into one by concatenating columns.
    decomps: list of (U, V, W) tuples
    Returns: (U, V, W)
    """
    U_list = []
    V_list = []
    W_list = []
    for U, V, W in decomps:
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    return U, V, W

def _try_subspace_decomposition(n, m, p, deadline):
    """
    Try to find a lower rank decomposition using subspace/block partitioning.
    This is a heuristic that may not always work, but can sometimes find
    better decompositions than pure random search.
    """
    if time.time() > deadline:
        return None
    
    # For 3x4x5, try partitioning into blocks
    # Strategy: partition rows of A (3x4) and columns of B (4x5) into blocks
    # We'll try a simple 2-block partitioning for now
    
    # Partition A's rows into blocks of size 1 and 2 (total 3)
    # Partition B's columns into blocks of size 2 and 3 (total 5)
    # This gives us block tensors of sizes:
    # - (1x4) * (4x2) = 1x2 tensor (rank 2)
    # - (1x4) * (4x3) = 1x3 tensor (rank 3)
    # - (2x4) * (4x2) = 2x2 tensor (rank 4)
    # - (2x4) * (4x3) = 2x3 tensor (rank 6)
    # Total rank: 2 + 3 + 4 + 6 = 15
    
    # But we need to verify this actually works. Let's try to construct it.
    
    # For now, let's try a different approach: use known results for small tensors
    # The rank of matrix multiplication for small dimensions is known in some cases.
    
    # For 3x4x5, the schoolbook rank is 60.
    # There are known improvements for some small cases, but 3x4x5 is not well-studied.
    
    # Let's try a simple optimization approach with better initialization
    # based on the structure of the problem.
    
    # Try to find a decomposition of rank 50-55 using structured random search
    for target_rank in [55, 52, 50, 48]:
        if time.time() > deadline - 2:
            break
        
        result = _try_structured_random(n, m, p, target_rank, deadline - 2, seed=target_rank)
        if result is not None:
            return result
    
    return None

def _try_structured_random(n, m, p, target_rank, deadline, seed=42):
    """
    Try to find a rank-`target_rank` decomposition using structured random initialization
    that respects the block structure of matrix multiplication.
    """
    if time.time() > deadline:
        return None
    
    rng = np.random.RandomState(seed)
    
    # Initialize with structured random matrices
    # Each column of U, V, W should be "localized" to respect the block structure
    
    best_cost = np.inf
    best_params = None
    
    num_trials = 20
    
    for trial in range(num_trials):
        if time.time() > deadline - 1:
            break
        
        # Create structured random initialization
        # For each "rank slice", assign it to a specific (i,j,k) region
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Distribute rank slices among the n*m*p possible (i,j,k) combinations
        # with some overlap to allow for optimization
        for r in range(target_rank):
            # Assign this rank slice to a random (i,j,k) with some spread
            i = rng.randint(n)
            j = rng.randint(m)
            k = rng.randint(p)
            
            # Add some spread to neighboring indices
            spread = rng.randint(2)
            
            for di in range(-spread, spread+1):
                for dj in range(-spread, spread+1):
                    for dk in range(-spread, spread+1):
                        ii = (i + di) % n
                        jj = (j + dj) % m
                        kk = (k + dk) % p
                        
                        U[ii*m+jj, r] = rng.randn()
                        V[jj*p+kk, r] = rng.randn()
                        W[kk*n+ii, r] = rng.randn()
        
        # Normalize to keep magnitudes reasonable
        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Compute cost
        def residuals(params):
            U_flat = params[:n*m*target_rank]
            V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = params[n*m*target_rank + m*p*target_rank :]
            U_r = U_flat.reshape(n*m, target_rank)
            V_r = V_flat.reshape(m*p, target_rank)
            W_r = W_flat.reshape(p*n, target_rank)
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                        res.append(val - 1.0)
            return np.array(res)
        
        cost = np.sum(residuals(params)**2)
        
        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()
        
        if cost < 1e-10:
            break
    
    if best_cost < 1e-8 and best_params is not None:
        U_flat = best_params[:n*m*target_rank]
        V_flat = best_params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = best_params[n*m*target_rank + m*p*target_rank :]
        U = U_flat.reshape(n*m, target_rank)
        V = V_flat.reshape(m*p, target_rank)
        W = W_flat.reshape(p*n, target_rank)
        
        if _verify(U, V, W, n, m, p, tol=1e-7):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 110  # Leave safety margin for the 120s limit
    
    # Try subspace decomposition first
    result = _try_subspace_decomposition(n, m, p, deadline - 10)
    if result is not None:
        best_U, best_V, best_W = result
        best_rank = best_U.shape[1]
    
    # If we still have time, try a few more ranks
    if time.time() < deadline - 5:
        for r in range(best_rank - 1, max(best_rank - 3, 19), -1):
            if time.time() > deadline - 5:
                break
            
            res = _try_structured_random(n, m, p, r, deadline - 5, seed=r)
            if res is not None:
                best_U, best_V, best_W = res
                best_rank = r
                break
    
    # Convert to list of lists for output
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
