import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Schoolbook rank-72 baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    baseline = schoolbook()
    
    # Try to find a lower-rank decomposition via random search
    # Strategy: start with schoolbook and try to merge/cancel terms
    # Or: use a different approach - try known decompositions
    
    # Let's try a Strassen-based approach for 3x4x6
    # 
    # Key insight: We can split the 4x6 matrix B into two 4x3 blocks:
    # B = [B1 | B2] where B1, B2 are 4x3
    # Then C = A*B = [A*B1 | A*B2]
    # Each A*Bi is a 3x4x3 multiplication
    # 
    # For 3x4x3: schoolbook rank = 36, total = 72
    # 
    # Can we do 3x4x3 better? Let's try a Strassen-like approach.
    # 
    # Actually, let me try a completely different approach:
    # Use the fact that we can express the 3x4x6 tensor as a sum of
    # rank-1 terms using a clever linear combination.
    #
    # Let's try a random search approach:
    # 1. Start with random U, V, W matrices of shape (nm, r), (mp, r), (pn, r)
    # 2. Check if they decompose the multiplication tensor
    # 3. Use gradient descent or similar to find valid decompositions
    #
    # But this is computationally expensive. Let me try a more structured approach.
    #
    # Known result: rank(3,4,6) <= 49 via a specific construction
    # Let me try to implement this.
    #
    # The construction uses the following idea:
    # Split A (3x4) into A1 (2x4) and A2 (1x4)
    # Split B (4x6) into B1 (4x3) and B2 (4x3)
    # Use Strassen on the 2x4x3 part (if possible)
    #
    # Actually, let me try a simpler approach: use the schoolbook decomposition
    # but with some terms combined using algebraic identities.
    #
    # For now, let's return the schoolbook rank-72 which is guaranteed correct.
    
    # Try a few random searches for lower-rank decompositions
    random.seed(42)
    
    for target_rank in [49, 50, 55, 60, 65, 70]:
        if time.time() > deadline - 5:
            break
        
        # Try to find a decomposition of this rank
        # Use a structured random approach
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Check if this decomposes the multiplication tensor
        # The multiplication tensor T[i,j,k] = 1 if i,j,k are valid indices
        # We need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
        # and 0 otherwise (but since i,j,k always form valid triples, we just need =1)
        
        # Actually, the tensor is: T_{i,j,k} = e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
        # So we need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
        
        # Let's check this condition
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Found a valid decomposition!
            # Convert to integer/half-integer if possible
            result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            return result
    
    # Fall back to schoolbook
    return baseline
# EVOLVE-BLOCK-END
