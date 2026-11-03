import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor values: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The equation is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # We will try ranks starting from a lower bound
    # Lower bound is max(n+m-1, m+p-1, n+p-1) = 7
    # But for 2x3x5, we suspect 13 or 14.
    # Let's try a range of ranks.
    
    def verify(U, V, W, R):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def try_random_decomposition(R, n_restarts=20):
        """Try random initialization and verify."""
        for restart in range(n_restarts):
            # Use different random seeds
            np.random.seed(restart * 1000 + R * 7)
            
            # Random entries from a small set of half-integers
            # This increases the chance of finding a valid half-integer decomposition
            half_ints = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            U = np.random.choice(half_ints, size=(n * m, R))
            V = np.random.choice(half_ints, size=(m * p, R))
            W = np.random.choice(half_ints, size=(p * n, R))
            
            if verify(U, V, W, R):
                return U, V, W
        
        return None

    def try_structured_decomposition(R):
        """Try structured decompositions based on known patterns."""
        # For matrix multiplication, there are known tensor rank decompositions.
        # Let's try some structured approaches.
        
        # One approach: use the fact that matrix multiplication can be decomposed
        # using the standard algorithm plus some optimizations.
        
        # Standard schoolbook has rank n*m*p = 30
        # We want to find lower ranks.
        
        # Let's try to build up from simpler cases.
        
        # For 2x3x5, one approach is to use the fact that we can combine
        # some of the multiplications.
        
        # Try a few structured patterns
        for pattern in range(10):
            np.random.seed(pattern * 500 + R)
            
            # Initialize with a mix of structured and random
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Fill in some structured entries
            for r in range(R):
                # Each rank-1 term contributes to some subset of (i,j,k)
                # Try to cover all (i,j,k) pairs
                i_idx = np.random.randint(0, n)
                j_idx = np.random.randint(0, m)
                k_idx = np.random.randint(0, p)
                
                # Add contributions
                U[i_idx * m + j_idx, r] = 1.0
                V[j_idx * p + k_idx, r] = 1.0
                W[k_idx * n + i_idx, r] = 1.0
                
                # Add some random perturbations
                if pattern % 2 == 0:
                    U[i_idx * m + j_idx, r] = np.random.choice([-1.0, 1.0])
                    V[j_idx * p + k_idx, r] = np.random.choice([-1.0, 1.0])
                    W[k_idx * n + i_idx, r] = np.random.choice([-1.0, 1.0])
            
            if verify(U, V, W, R):
                return U, V, W
        
        return None

    # Try ranks from 13 to 20
    for R in range(13, 21):
        # First try random half-integer search
        result = try_random_decomposition(R, n_restarts=50)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        
        # Then try structured search
        result = try_structured_decomposition(R)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        
        # If still not found, try a more systematic search with limited values
        # Use a smaller search space
        for attempt in range(20):
            np.random.seed(attempt * 777 + R * 333)
            
            # Use values from {-1, -0.5, 0, 0.5, 1}
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            U = np.random.choice(vals, size=(n * m, R))
            V = np.random.choice(vals, size=(m * p, R))
            W = np.random.choice(vals, size=(p * n, R))
            
            if verify(U, V, W, R):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
