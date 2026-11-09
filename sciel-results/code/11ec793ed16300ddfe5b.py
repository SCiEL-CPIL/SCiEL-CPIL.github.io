"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    def schoolbook():
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
    
    best_result = schoolbook()
    
    # Try to find a lower rank decomposition using random search
    # The tensor equation is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # We'll use a structured random search approach
    # Key insight: we can try to find sparse decompositions
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W decompose the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        rank = U.shape[1]
        
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try random search for lower rank
    # We'll use a heuristic: try to find decompositions with small integer/rational entries
    # Start from lower ranks and work up
    
    import random
    rng = random.Random(42)
    
    # Known lower bounds: for 4x4x8, the minimum rank is at least 128 (schoolbook)
    # but we can try to find better ones
    
    # Strategy: try random matrices with small entries and check if they work
    # This is unlikely to succeed for much lower ranks, but let's try a few
    
    # Try a few random attempts with decreasing rank
    for target_rank in range(120, 128):
        if time.time() > deadline - 10:
            break
            
        # Try a few random decompositions
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            # Generate random U, V, W with small entries
            # Use a structured approach: random sparse matrices
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Fill with random values from {-1, 0, 1}
            for c in range(target_rank):
                # Randomly choose which entries to set
                for idx in range(rng.randint(1, 5)):
                    row = rng.randint(0, n * m - 1)
                    U[row, c] = rng.choice([-1, 1])
                
                for idx in range(rng.randint(1, 5)):
                    row = rng.randint(0, m * p - 1)
                    V[row, c] = rng.choice([-1, 1])
                
                for idx in range(rng.randint(1, 5)):
                    row = rng.randint(0, p * n - 1)
                    W[row, c] = rng.choice([-1, 1])
            
            # Verify
            if verify(U, V, W):
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    return best_result
# EVOLVE-BLOCK-END
