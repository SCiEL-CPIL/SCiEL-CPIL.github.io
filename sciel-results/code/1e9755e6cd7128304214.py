import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that U, V, W form a valid tensor decomposition using vectorized operations."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check if T[i,j,k] = 1 for all i,j,k
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline with margin
    
    # Start with the trivial decomposition (rank = n*m*p = 24)
    rank = n * m * p
    U0 = np.zeros((n * m, rank))
    V0 = np.zeros((m * p, rank))
    W0 = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Strategy: Try to find lower-rank decompositions by merging factors
    # For each pair of trivial factors, try to see if they can be combined
    
    # First, try rank 23, 22, etc. using a smarter search
    
    # Approach: Use the fact that we can try random linear combinations
    # of the 24 trivial factors and check if we get a valid decomposition
    
    # For target_rank < 24, we need to find a (24 - target_rank)-dimensional
    # subspace of linear dependencies
    
    # Let's try a structured approach:
    # 1. Generate random vectors in the space of possible decompositions
    # 2. Check if they satisfy the tensor equation
    
    # Better approach: Use a greedy merging strategy
    # Start with all 24 factors, try to merge pairs by finding linear dependencies
    
    # For now, let's try a more efficient random search with structured values
    
    # Try ranks from 23 down to 20
    for target_rank in range(23, 19, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random attempts for this rank
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random U, V, W with the target rank
            # Use values from a structured set to increase chance of exact solutions
            U_rand = np.zeros((n * m, target_rank))
            V_rand = np.zeros((m * p, target_rank))
            W_rand = np.zeros((p * n, target_rank))
            
            # Fill with random values from {-1, -0.5, 0, 0.5, 1}
            # But use a seeded random for reproducibility within the attempt
            rng = np.random.RandomState(attempt * 1000 + target_rank)
            vals = np.array([-1, -0.5, 0, 0.5, 1])
            
            for r in range(target_rank):
                U_rand[:, r] = vals[rng.randint(0, len(vals), n * m)]
                V_rand[:, r] = vals[rng.randint(0, len(vals), m * p)]
                W_rand[:, r] = vals[rng.randint(0, len(vals), p * n)]
            
            if _verify(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                break
        
        # If we found a valid decomposition at this rank, we can stop
        if best_rank <= target_rank:
            break
    
    # If we didn't find anything better, try a different strategy:
    # Use a local search starting from the trivial decomposition
    if best_rank == rank:
        if time.time() < deadline - 20:
            # Try to perturb the trivial decomposition and see if we can reduce rank
            # This is a heuristic approach
            
            # For each pair of columns, try to see if one can be expressed as a linear
            # combination of the others (with some tolerance for the tensor equation)
            
            # This is complex, so let's just return the trivial decomposition
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
