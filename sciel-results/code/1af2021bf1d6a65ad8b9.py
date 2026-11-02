"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140  # Leave 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Trivial schoolbook decomposition (rank = n*m*p = 105)
    rank_trivial = n * m * p
    U_trivial = np.zeros((nm, rank_trivial), dtype=np.float64)
    V_trivial = np.zeros((mp, rank_trivial), dtype=np.float64)
    W_trivial = np.zeros((pn, rank_trivial), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, col] = 1
                V_trivial[b, col] = 1
                W_trivial[c, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    best_rank = rank_trivial
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_computed, T_target, atol=1e-9)
    
    def try_structured_reduction(U, V, W, current_rank, rng, max_trials=50):
        """Try to reduce rank by exploiting structure: find terms that can be merged."""
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        
        # Strategy: Try to find pairs of columns where one is a linear combination
        # of others, or where the tensor contribution can be cancelled
        
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                break
            if best_rank <= 2:
                break
            
            # Try to remove one column and see if the remaining columns
            # can be adjusted to compensate
            cols_to_try = rng.choice(best_rank, size=min(10, best_rank), replace=False)
            
            for col_idx in cols_to_try:
                if time.time() > deadline - 20:
                    break
                
                # Remove this column and try to solve for adjustments
                # This is a linear system problem
                U_sub = best_U[:, list(range(best_rank))[:col_idx] + list(range(best_rank))[col_idx+1:]]
                V_sub = best_V[:, list(range(best_rank))[:col_idx] + list(range(best_rank))[col_idx+1:]]
                W_sub = best_W[:, list(range(best_rank))[:col_idx] + list(range(best_rank))[col_idx+1:]]
                
                # The removed term contributed: U[:,col_idx] * V[:,col_idx] * W[:,col_idx]
                # We need to see if this can be expressed as a combination of remaining terms
                # This requires solving a large linear system, so skip for now
                pass
        
        return best_U, best_V, best_W, best_rank
    
    # Better strategy: Use structured decomposition based on block structure
    # For matrix multiplication C = A * B where A is n x m and B is m x p
    # We can try to use decompositions that exploit the structure
    
    # Approach: Try to find a decomposition with rank < n*m*p by using
    # the fact that some entries might be shared or cancellable
    
    # Let's try a more systematic approach: start from the trivial decomposition
    # and try to find linear dependencies among the rank-1 terms
    
    rng = np.random.default_rng(42)
    
    # Try to improve by finding if any terms are redundant
    # For each term, check if it can be expressed as a combination of others
    
    # A more practical approach: try different structured patterns
    # that might give lower rank
    
    # Pattern 1: Try to use the fact that for certain index patterns,
    # multiple terms might contribute to the same output entry
    
    # Let's try a greedy approach: start with trivial, try to remove terms
    # and adjust others to compensate
    
    # For efficiency, let's try a different strategy: use known results
    # or try structured patterns that are known to work for small cases
    
    # For 3x5 x 5x7, the rank is at least max(nm, mp, pn) = 35
    # and at most n*m*p = 105
    
    # Let's try to find a decomposition with rank around 50-70 using
    # a structured approach
    
    # Strategy: Use the fact that the tensor has a specific structure
    # and try to find a low-rank approximation that is exact
    
    # For now, let's try to improve the trivial decomposition by
    # finding if any terms can be merged or cancelled
    
    # Try to find linear dependencies among the rank-1 terms
    # This is equivalent to finding if the tensor has a lower rank
    
    # A practical approach: try different random seeds and patterns
    # and see if we can find a valid decomposition with lower rank
    
    # Let's try a more focused search: for each possible rank from 104 down,
    # try to find a valid decomposition using structured initialization
    
    lower_bound = max(nm, mp, pn)  # 35
    
    # Try ranks from 104 down to lower_bound + some margin
    for target_rank in range(104, max(lower_bound, 50) - 1, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few structured patterns for this rank
        max_trials = 20
        for trial in range(max_trials):
            if time.time() > deadline - 25:
                break
            
            # Pattern: Use entries from {-1, 0, 1} with structured sparsity
            # that respects the matrix multiplication structure
            
            # Initialize with the trivial decomposition structure
            U = np.zeros((nm, target_rank), dtype=np.float64)
            V = np.zeros((mp, target_rank), dtype=np.float64)
            W = np.zeros((pn, target_rank), dtype=np.float64)
            
            # Try to assign terms in a way that might allow cancellation
            # For each (i,j,k) in the trivial decomposition, try to
            # assign it to one of the target_rank terms
            
            # Use a greedy assignment with some randomization
            assignments = rng.integers(0, target_rank, size=(n, m, p))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        r = assignments[i, j, k]
                        
                        # Add contribution with random sign
                        sign = rng.choice([-1, 1])
                        U[a, r] += sign
                        V[b, r] += sign
                        W[c, r] += sign
            
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        if best_rank < 105:
            break
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
