"""Matrix multiplication 3,3,7 - rank minimization with verification and time guard."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin within 150s limit

    def schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
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

    def verify(U, V, W, num_samples=30):
        """Verify tensor decomposition correctness using random integer matrices."""
        rng = random.Random(42)
        rank = len(U[0]) if U else 0
        for _ in range(num_samples):
            A = [[rng.randint(-3, 3) for _ in range(m)] for _ in range(n)]
            B = [[rng.randint(-3, 3) for _ in range(p)] for _ in range(m)]
            # True product
            C_true = [[0] * p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += A[i][j] * B[j][k]
                    C_true[i][k] = s
            # Product via factorization: C[i,k] = sum_{j,c} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            C_fact = [[0] * p for _ in range(n)]
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        uc = U[i * m + j][c]
                        if uc == 0:
                            continue
                        for k in range(p):
                            vc = V[j * p + k][c]
                            if vc == 0:
                                continue
                            wc = W[k * n + i][c]
                            if wc == 0:
                                continue
                            C_fact[i][k] += uc * vc * wc
            for i in range(n):
                for k in range(p):
                    if C_true[i][k] != C_fact[i][k]:
                        return False
        return True

    # Try to find a lower rank decomposition using structured random search
    # Strategy: start from schoolbook and try to merge/eliminate terms
    # For 3x3x7, schoolbook rank is 63. We try to find rank < 63.
    
    # Approach: Use the fact that matrix multiplication tensor can sometimes
    # be decomposed with fewer terms by exploiting linear dependencies.
    # We'll try a random perturbation + pruning approach.
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook rank
    
    # First, try a few random structured decompositions
    # The key insight: for rectangular matrices, there may be structure
    # that allows rank reduction.
    
    # Try rank from best_rank-1 down, but only a few attempts per rank due to time
    max_attempts_per_rank = 5
    
    for target_rank in range(best_rank - 1, max(20, best_rank - 15), -1):
        if time.time() > deadline:
            break
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline:
                break
            
            # Generate random factor matrices with small integer entries
            # Using a structured approach: start with random sparse matrices
            U = [[random.randint(-2, 2) if random.random() < 0.3 else 0 
                  for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) if random.random() < 0.3 else 0 
                  for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) if random.random() < 0.3 else 0 
                  for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U, V, W, num_samples=20):
                # Found a valid decomposition!
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # Move to next (lower) rank
        if best_result is not None and best_rank <= target_rank:
            # Found something at this rank or lower, continue searching lower
            pass
    
    # If we found something better, return it; otherwise return schoolbook
    if best_result is not None:
        return best_result
    
    return schoolbook()
# EVOLVE-BLOCK-END
