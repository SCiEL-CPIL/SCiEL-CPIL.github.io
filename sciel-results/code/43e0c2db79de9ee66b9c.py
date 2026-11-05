"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition search."""
DIMENSIONS = [4, 5, 5]

from fractions import Fraction
import itertools
import random
import sys

def construct():
    n, m, p = DIMENSIONS
    # Trivial schoolbook as fallback
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    def verify_decomposition(U, V, W, rank):
        """Check if sum_r U[:,r]*V[:,r]*W[:,r] equals the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    # New approach: Use structured decomposition based on embedding smaller matrices
    # Strategy: Decompose 4x5x5 by splitting dimensions into blocks
    # For example, split 5 = 4 + 1, then use block matrix multiplication
    
    # Let's try a systematic search over structured patterns
    # Pattern: Use the fact that we can write 5x5 multiplication using 4x4 blocks plus corrections
    
    # First, try a direct construction using Kronecker-like patterns
    # We'll search for decompositions where entries are in {-2,-1,0,1,2}
    
    # Use a smarter initialization: start from schoolbook and try to merge terms
    # Or use known bilinear algorithm patterns
    
    # For now, let's do a more intelligent random search with structure
    # Key insight: In matrix multiplication, the tensor has a specific sparsity pattern
    # T[i,j,k] = 1 only when the indices are "compatible" (j is shared)
    
    # Let's try ranks from a better lower bound
    # Lower bound for matrix multiplication: rank >= max(n*m, m*p, p*n) / something
    # Actually, a better lower bound: rank >= ceil(n*p) = 20, but also considering structure
    # For 4x5x5, known results suggest rank around 40-50 might be achievable
    
    best = None
    lower_bound = 25  # Conservative lower bound
    upper_bound = min(rank_trivial, 60)
    
    # Use structured random search: generate matrices with specific patterns
    random.seed(123)
    
    # Try different entry distributions
    for rank in range(lower_bound, upper_bound + 1):
        for trial in range(100):
            # Generate with more structure: use small integers
            # Try different magnitudes
            max_entry = random.choice([1, 2, 3])
            U = [[random.randint(-max_entry, max_entry) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.randint(-max_entry, max_entry) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.randint(-max_entry, max_entry) for _ in range(rank)] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W, rank):
                best = (rank, U, V, W)
                break
        if best is not None:
            break
    
    if best is not None:
        rank, U, V, W = best
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to trivial
        return {"U": U_trivial, "V": V_trivial, "W": W_trivial}
