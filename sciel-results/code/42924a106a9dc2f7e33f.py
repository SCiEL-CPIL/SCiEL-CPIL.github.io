"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 125  # Safety margin before 150s kill
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        rank = n * m * p  # 72
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
    
    best_result = schoolbook()
    best_rank = n * m * p  # 72
    
    # Verify a candidate decomposition
    def verify(U, V, W, rank):
        """Check if U,V,W with given rank exactly decompose the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try structured rank reduction using slice-and-dice approach
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We can partition the output into blocks and try to share structure
    
    # Approach 1: Try to find a decomposition where we can share some columns
    # by exploiting the fact that the tensor is all-ones
    
    # For rank r, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Let's try a systematic search with structured initialization
    
    # Strategy: Use the fact that for each (i,k) pair, we need:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all j
    
    # This is a system of linear equations in the products
    
    # Try different rank values with structured patterns
    
    for target_rank in range(71, 40, -1):
        if time.time() > deadline:
            break
        
        # Try a few structured patterns
        for attempt in range(8):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialize with small random half-integer values
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Try a structured pattern: for each column, assign to a subset of (i,j,k)
            for c in range(target_rank):
                # Choose which (i,j,k) triples this column affects
                # Use a random subset
                affected = set()
                for _ in range(np.random.randint(3, 8)):
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    affected.add((i, j, k))
                
                for (i, j, k) in affected:
                    # Assign values that will contribute to the sum
                    val = np.random.choice([-2, -1, 1, 2])
                    U[i * m + j, c] += val
                    V[j * p + k, c] += val
                    W[k * n + i, c] += val
            
            # Round to half-integers
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            # Verify
            if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break  # Found valid for this rank, try lower
        
        if best_rank < 71:
            break
    
    # Strategy 2: Try a more systematic approach using the structure of the problem
    # For each (i,k) pair, we have 3 equations (one for each j)
    # sum_j=0..2 U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # This suggests we can try to solve for the entries systematically
    
    if time.time() < deadline - 20 and best_rank == 72:
        # Try rank 60-70 with a different structured approach
        for target_rank in range(70, 55, -1):
            if time.time() > deadline:
                break
            
            for attempt in range(5):
                if time.time() > deadline:
                    break
                
                np.random.seed(attempt * 2000 + target_rank)
                
                # Initialize with a pattern that tries to share structure
                U = np.zeros((n * m, target_rank), dtype=float)
                V = np.zeros((m * p, target_rank), dtype=float)
                W = np.zeros((p * n, target_rank), dtype=float)
                
                # For each column, try to make it contribute to multiple (i,j,k) triples
                for c in range(target_rank):
                    # Choose a base (i,k) pair
                    i_base = np.random.randint(0, n)
                    k_base = np.random.randint(0, p)
                    
                    # Assign values for all j
                    for j in range(m):
                        U[i_base * m + j, c] = np.random.choice([-2, -1, 1, 2])
                        V[j * p + k_base, c] = np.random.choice([-2, -1, 1, 2])
                        W[k_base * n + i_base, c] = np.random.choice([-2, -1, 1, 2])
                    
                    # Also add some cross terms
                    if np.random.random() < 0.5:
                        i_other = np.random.randint(0, n)
                        k_other = np.random.randint(0, p)
                        for j in range(m):
                            U[i_other * m + j, c] += np.random.choice([-1, 0, 1])
                            V[j * p + k_other, c] += np.random.choice([-1, 0, 1])
                            W[k_other * n + i_other, c] += np.random.choice([-1, 0, 1])
                
                # Round to half-integers
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Verify
                if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
            
            if best_rank < 70:
                break
    
    # Strategy 3: Try a pure random search with smaller values for lower ranks
    if time.time() < deadline - 15 and best_rank == 72:
        for target_rank in range(65, 45, -1):
            if time.time() > deadline:
                break
            
            for attempt in range(3):
                if time.time() > deadline:
                    break
                
                np.random.seed(attempt * 3000 + target_rank)
                
                # Random initialization with small half-integer values
                U = (np.random.randint(-3, 4, (n * m, target_rank)) * 0.5)
                V = (np.random.randint(-3, 4, (m * p, target_rank)) * 0.5)
                W = (np.random.randint(-3, 4, (p * n, target_rank)) * 0.5)
                
                if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
            
            if best_rank < 65:
                break
    
    return best_result
# EVOLVE-BLOCK-END
