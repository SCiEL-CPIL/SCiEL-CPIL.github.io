"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    # Schoolbook decomposition as fallback (rank = n*m*p = 84)
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

    def verify_fast(U, V, W):
        """Vectorized verification using numpy."""
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        # Check all entries equal 1
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def verify_integer(U, V, W):
        """Check if entries are integer or half-integer."""
        for mat in [U, V, W]:
            for row in mat:
                for x in row:
                    # Check if x is integer or half-integer
                    doubled = 2 * x
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        return True

    best_result = schoolbook()
    best_rank = n * m * p

    # Known lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # The tensor rank of matrix multiplication is known to be at most n*m*p
    # and at least max(n*m, m*p, p*n)
    
    # For 3x4x7, we know from literature that the rank is likely around 30-40
    # Let's try to find a decomposition using structured search
    
    # Strategy: Use the fact that matrix multiplication can be decomposed
    # using bilinear methods. We'll try various structured approaches.
    
    # First, let's try a systematic approach based on the structure of matrix multiplication
    # The key insight: we need T[i,j,k] = 1 for all i,j,k
    
    # Approach: Try to find a low-rank decomposition by solving a system of equations
    # For each rank r, we have r terms: sum_{c=1}^{r} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of n*m*p = 84 equations in 3*r variables (r per matrix)
    # We can try to solve this using numerical methods
    
    import numpy as np
    from scipy.optimize import least_squares
    
    # Value pool for half-integer entries
    value_pool = [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5, 3, -3]
    
    # Try ranks from lower bound up to current best
    for target_rank in range(28, min(best_rank, 50)):
        if time.time() > deadline:
            break
        
        # Limit attempts per rank to stay within time budget
        max_attempts = 30 if target_rank < 40 else 15
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            strategy = attempt % 4
            
            if strategy == 0:
                # Random from value pool
                U = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(p * n)]
            elif strategy == 1:
                # Sparse random
                U = [[random.choice(value_pool) if random.random() < 0.4 else 0 for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice(value_pool) if random.random() < 0.4 else 0 for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice(value_pool) if random.random() < 0.4 else 0 for _ in range(target_rank)] for _ in range(p * n)]
            elif strategy == 2:
                # Integer-only
                int_pool = [0, 1, -1, 2, -2, 3, -3]
                U = [[random.choice(int_pool) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice(int_pool) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice(int_pool) for _ in range(target_rank)] for _ in range(p * n)]
            else:
                # Binary with some -1
                bin_pool = [0, 1, -1]
                U = [[random.choice(bin_pool) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice(bin_pool) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice(bin_pool) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Quick check
            if all(all(x == 0 for x in row) for row in U):
                continue
            if all(all(x == 0 for x in row) for row in V):
                continue
            if all(all(x == 0 for x in row) for row in W):
                continue
            
            if verify_fast(U, V, W):
                if verify_integer(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
        
        if best_rank == target_rank:
            break
    
    # If we haven't found a better solution, try a more systematic approach
    # using the structure of the problem
    
    # For matrix multiplication, there's a known decomposition using
    # the fact that (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let's try to exploit this structure more directly
    
    # One approach: use the identity that we can write the multiplication
    # as a sum of rank-1 tensors
    
    # For now, return the best result found
    return best_result
# EVOLVE-BLOCK-END
