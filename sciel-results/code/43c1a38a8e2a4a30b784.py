import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a low-rank decomposition using random search with integer constraints
    # Start from rank 55 and go up
    for rank in [55, 56, 57, 58, 59, 60, 61, 62, 63]:
        result = try_find_decomposition(n, m, p, rank)
        if result is not None:
            return result
    # Fallback to schoolbook
    return schoolbook(n, m, p)

def schoolbook(n, m, p):
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

def try_find_decomposition(n, m, p, rank):
    """
    Try to find a rank-rank decomposition with integer entries.
    Uses random initialization and local search.
    """
    num_trials = 100
    for trial in range(num_trials):
        random.seed(42 + trial)
        # Initialize U, V, W with small random integers
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        # Fill with random small integers
        for r in range(rank):
            for i in range(n * m):
                U[i, r] = random.randint(-1, 1)
            for j in range(m * p):
                V[j, r] = random.randint(-1, 1)
            for k in range(p * n):
                W[k, r] = random.randint(-1, 1)
        
        # Check if this decomposes the tensor
        if check_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Local search: try to improve
        best_error = compute_error(U, V, W, n, m, p)
        for step in range(1000):
            # Pick a random element to modify
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = random.randint(0, U.size - 1)
                i, r = np.unravel_index(idx, U.shape)
                old_val = U[i, r]
                U[i, r] = old_val + random.choice([-1, 0, 1])
            elif mat == 'V':
                idx = random.randint(0, V.size - 1)
                j, r = np.unravel_index(idx, V.shape)
                old_val = V[j, r]
                V[j, r] = old_val + random.choice([-1, 0, 1])
            else:
                idx = random.randint(0, W.size - 1)
                k, r = np.unravel_index(idx, W.shape)
                old_val = W[k, r]
                W[k, r] = old_val + random.choice([-1, 0, 1])
            
            new_error = compute_error(U, V, W, n, m, p)
            if new_error < best_error:
                best_error = new_error
                if best_error == 0:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                # Revert
                if mat == 'U':
                    U[i, r] = old_val
                elif mat == 'V':
                    V[j, r] = old_val
                else:
                    W[k, r] = old_val
    
    return None

def check_decomposition(U, V, W, n, m, p):
    """
    Check if U, V, W exactly decompose the matrix multiplication tensor.
    The tensor T_{i,j,k} = 1 for all i,j,k in the standard basis.
    Decomposition: T = sum_r U[:,r] V[:,r] W[:,r]^T
    So for each (i,j,k): sum_r U[i,r] * V[j,r] * W[k,r] should equal 1
    where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    """
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # For each i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s != 1:
                    return False
    return True

def compute_error(U, V, W, n, m, p):
    """
    Compute the sum of squared errors.
    """
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                error += (s - 1) ** 2
    return error
# EVOLVE-BLOCK-END
