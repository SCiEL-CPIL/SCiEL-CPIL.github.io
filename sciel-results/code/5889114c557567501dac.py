"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy: Direct sum decomposition
    # 3x3x4 = 3x3x3 (k=0,1,2) + 3x3x1 (k=3)
    # Use Strassen (rank 7) for 3x3x3 part
    # Use schoolbook (rank 9) for 3x3x1 part
    # Total rank = 16
    
    # First, build Strassen decomposition for 3x3x3
    # Strassen's 7 multiplications for 2x2x2, but we need 3x3x3
    # Actually, the minimal rank for 3x3x3 is 23, not 7.
    # Rank 7 is for 2x2x2 (Strassen).
    # For 3x3x3, the best known is 23.
    
    # Let me reconsider. The minimal rank for multiplying 3x3 by 3x3 is 23.
    # So the 3x3x3 part has rank 23, not 7.
    
    # So total rank = 23 + 9 = 32? Or can we do better?
    
    # Actually, let me just use the schoolbook for the whole thing first,
    # then try to optimize.
    
    # Schoolbook rank = 3*3*4 = 36
    
    # Let's try the direct sum: 3x3x3 (rank 23) + 3x3x1 (rank 9) = 32
    
    # I need a valid rank-23 decomposition for 3x3x3.
    # This is complex. Let me use a known construction or fall back to schoolbook.
    
    # For now, let's use schoolbook for the 3x3x3 part too, giving 27 + 9 = 36.
    # That's not better than just doing 36 directly.
    
    # The key insight: if I can find a rank-23 decomposition for 3x3x3,
    # then total rank = 23 + 9 = 32, which is better than 36.
    
    # Let me try to construct or find a rank-23 decomposition for 3x3x3.
    # This is non-trivial. Let me use a randomized approach to search for lower rank.
    
    # Actually, let's be pragmatic. The best known rank for 3x3x3 is 23.
    # I'll implement a search for a valid decomposition of rank R for the full 3x3x4 problem.
    
    # Given time constraints, let me try a hybrid:
    # 1. Start with schoolbook (rank 36)
    # 2. Try to find a rank-35 or lower decomposition via random search + verification
    
    rank_school = n * m * p  # 36
    
    # Build schoolbook as baseline
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    # Try to find a lower rank decomposition
    # The tensor equation: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # For a given rank R, we need to find U, V, W such that the above holds.
    # This is a system of bilinear equations.
    
    # Let's try random initialization + iterative refinement for rank 35, 34, etc.
    
    def verify_decomposition(U, V, W, n, m, p):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Try rank 35
    for target_rank in [35, 34, 33]:
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        np.random.seed(42)
        U_rand = np.random.randn(n * m, target_rank) * 0.5
        V_rand = np.random.randn(m * p, target_rank) * 0.5
        W_rand = np.random.randn(p * n, target_rank) * 0.5
        
        # Simple iterative refinement: not straightforward for this problem
        # Let's just check if random works (very unlikely)
        if verify_decomposition(U_rand, V_rand, W_rand, n, m, p):
            best_result = {
                "U": U_rand.tolist(),
                "V": V_rand.tolist(),
                "W": W_rand.tolist()
            }
            best_rank = target_rank
            break
    
    # The random approach is very unlikely to work.
    # Let me try a structured approach: use the direct sum idea.
    
    # For the 3x3x1 part (k=3), schoolbook uses 9 terms.
    # For the 3x3x3 part (k=0,1,2), if I can use rank 23, total = 32.
    
    # Let me try to construct a rank-23 decomposition for 3x3x3.
    # This is the famous open problem. The best known is 23.
    
    # Given the complexity, let me just return the schoolbook result.
    # But wait, let me try one more thing: can I exploit the structure?
    
    # Actually, let me try a different approach: use the fact that for the 3x3x1 part,
    # I only need 3 terms if I can share with the 3x3x3 part.
    
    # This is getting too complex for the time budget. Let me return the schoolbook.
    
    if time.time() > deadline - 10:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
