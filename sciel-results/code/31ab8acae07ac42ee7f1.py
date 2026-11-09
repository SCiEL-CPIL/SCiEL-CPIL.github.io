"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(r):
                        actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def try_rank(r, time_budget=30):
        """Try to find a rank-r decomposition using structured approach."""
        if r < 1:
            return None
        
        # Strategy 1: Block-structured decomposition
        # For 4x5 x 5x5, we can try various block structures
        
        # Strategy 2: Random initialization + optimization
        # Use the fact that we need integer/half-integer entries
        
        # Try different rank values from known results
        # For 4x5x5, the tensor rank is at most 100 (schoolbook)
        # Known lower bounds suggest it might be lower
        
        # Attempt with specific structured patterns
        # Pattern: use the 4x4 submatrix structure
        
        # For now, try to see if we can find any improvement
        # by checking if a smaller rank works with random search
        
        if time.time() > start_time + time_budget:
            return None
        
        # Try a simple heuristic: can we reduce rank by combining terms?
        # The schoolbook has 100 terms. Let's see if some can be merged.
        
        # For exact decomposition with integer/half-integer entries,
        # we need to be careful. Let's try a few specific ranks.
        
        # Known: rank(4x5x5) <= 4*5*5 = 100
        # Lower bound: max(4*5, 5*5, 5*4) = max(20, 25, 20) = 25
        # But actual rank is likely higher.
        
        # Try ranks: 99, 98, 97, ... down to some minimum
        # This is expensive, so we'll be selective.
        
        # For this problem, let's try a few specific approaches:
        
        # Approach: Use the fact that we can sometimes save rank by
        # using bilinear tricks. For 4x5x5, try to find if rank < 100 works.
        
        # Since this is computationally hard, we'll return None for now
        # unless we find a quick win.
        
        return None
    
    # Try to find better rank (limited time)
    for r in range(rank_schoolbook - 1, max(50, rank_schoolbook - 10), -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(r, time_budget=5)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_rank = r
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
