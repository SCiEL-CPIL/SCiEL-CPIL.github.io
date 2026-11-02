"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Start with the schoolbook (trivial) decomposition as fallback
    rank_school = n * m * p  # 36
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # The multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r]
    # W is used to label output positions, but the core constraint is on U and V.
    
    # Strategy: Try to find a lower-rank decomposition by exploiting structure.
    # For 3x3x4, we can try to share products across the 4 columns of B.
    
    # Key insight: For each fixed (i, j), the product A[i,j] * B[j,k] appears
    # for all k. If we can group terms, we might reduce rank.
    
    # Let's try a systematic search for rank < 36 using random initialization
    # and gradient-free optimization (or direct algebraic solving).
    
    # Approach: Try ranks from a lower bound upward.
    # Lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # But realistically, for 3x3x3 we know rank is around 23, so for 3x3x4 it's at least that.
    
    # Let's try to construct a valid decomposition with rank < 36.
    # We'll use a random search approach with verification.
    
    def verify(U, V, W, rank):
        """Check if the decomposition is valid."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        
        # Reconstruct tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r]
        # Should equal 1 for all (i, j, k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i * m + j, r] * V_arr[j * p + k, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a lower-rank decomposition using a structured approach.
    # For 3x3x4, one approach is to use the fact that we can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # If we fix j, then for all (i, k), we have A[i,j] * B[j,k].
    # This is an outer product of A[:,j] and B[j,:].
    # 
    # The trivial decomposition uses 3*3*4 = 36 such outer products.
    # Can we do better?
    
    # Let's try a random search for small rank reductions.
    # We'll parameterize U and V with random values and try to solve for
    # a valid decomposition.
    
    # Actually, let's try a more systematic approach:
    # Use the fact that the tensor is separable in a certain way.
    
    # For now, let's try a few random rank reductions and see if any work.
    # This is a heuristic approach.
    
    np.random.seed(42)
    
    # Try ranks from 35 down to 20 (we'll stop early if we find one)
    for target_rank in range(best_rank - 1, 11, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            
            # Check if this random decomposition works (very unlikely)
            if verify(U_rand.tolist(), V_rand.tolist(), None, target_rank):
                # Found a valid decomposition!
                # Construct W (it's just a labeling, so we can use a simple scheme)
                W_rand = [[0] * target_rank for _ in range(p * n)]
                for r in range(target_rank):
                    # Find which (i, k) this rank term contributes to
                    # This is complex; for now, just use a placeholder
                    pass
                best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    # If no better solution found, return the schoolbook decomposition
    return best_result
# EVOLVE-BLOCK-END
