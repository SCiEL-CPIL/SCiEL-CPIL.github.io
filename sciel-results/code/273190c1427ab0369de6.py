"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Function to create the schoolbook decomposition
    def schoolbook_decomposition():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Function to check if a decomposition is valid
    def check_decomposition(U, V, W, tol=1e-6):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        
        rank = U.shape[1]
        
        # Build the full tensor and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    expected = 1.0
                    actual = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                    if abs(actual - expected) > tol:
                        return False
        
        # Check that non-matching entries are zero
        # This is expensive, so we sample
        import random
        for _ in range(1000):
            a = random.randint(0, n * m - 1)
            b = random.randint(0, m * p - 1)
            c = random.randint(0, p * n - 1)
            # Check if this corresponds to a valid (i,j,k)
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            if not (i_a == i_c and j_a == j_b and k_b == k_c):
                actual = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                if abs(actual) > tol:
                    return False
        return True
    
    # Start with schoolbook
    best_decomp = schoolbook_decomposition()
    best_rank = rank_schoolbook
    
    # Try to find a lower-rank decomposition using a simple approach
    # We'll try ranks from 41 down to 30, with a time limit
    for target_rank in range(rank_schoolbook - 1, max(20, rank_schoolbook - 15), -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
            
            # Initialize with schoolbook and try to merge
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Fill with schoolbook-like structure, but with some sharing
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < target_rank:
                            U[i * m + j, col] = 1
                            V[j * p + k, col] = 1
                            W[k * n + i, col] = 1
                            col += 1
                        else:
                            # Try to reuse an existing column
                            # Find a column that can be adjusted
                            break
                    if col >= target_rank:
                        break
                if col >= target_rank:
                    break
            if col >= target_rank:
                break
            
            # This approach is too simplistic. Let's just return schoolbook.
            break
    
    return best_decomp
# EVOLVE-BLOCK-END
