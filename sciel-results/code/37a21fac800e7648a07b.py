import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145  # Leave 5s safety margin

def _check_time():
    return (time.time() - START_TIME) > TIME_LIMIT

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification using numpy matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We need sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Compute sum_r U[a,r]*V[b,r]*W[c,r]
    # This is equivalent to: for each r, compute outer(U[:,r], V[:,r]) then multiply element-wise with W[:,r]
    # More efficiently: reshape and use einsum
    # result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    result = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check if result matches T
    return np.allclose(result, T, atol=1e-10)

def _schoolbook_decomposition(n, m, p):
    """Create the schoolbook decomposition with rank n*m*p."""
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
    return U, V, W

def _strassen_like_decomposition(n, m, p):
    """
    Try a Strassen-like decomposition that may have lower rank than schoolbook.
    For 3x4x6, we can try to exploit structure.
    """
    # Start with schoolbook and try to merge terms
    U, V, W = _schoolbook_decomposition(n, m, p)
    return U, V, W

def _random_perturbation(U, V, W, n, m, p, epsilon=0.1):
    """Apply small random perturbations to see if we can find equivalent lower-rank forms."""
    rank = len(U[0])
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    # Randomly try to zero out some entries
    for r in range(rank):
        if random.random() < 0.3:
            # Try to simplify column r
            if random.random() < 0.5:
                # Zero out a random entry in U
                idx = random.randint(0, n*m-1)
                U_new[idx][r] = 0
            else:
                # Zero out a random entry in V
                idx = random.randint(0, m*p-1)
                V_new[idx][r] = 0
        if random.random() < 0.3:
            if random.random() < 0.5:
                idx = random.randint(0, p*n-1)
                W_new[idx][r] = 0
            else:
                # Scale a column
                scale = random.choice([0.5, 2, -1])
                for i in range(n*m):
                    U_new[i][r] *= scale
                for j in range(m*p):
                    V_new[j][r] *= scale
                for k in range(p*n):
                    W_new[k][r] *= scale
    
    return U_new, V_new, W_new

def _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit=10):
    """Try to reduce the rank by finding linearly dependent columns."""
    start = time.time()
    current_U, current_V, current_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    current_rank = len(current_U[0])
    
    while current_rank > target_rank and (time.time() - start) < time_limit:
        if _check_time():
            break
        # Try to find a column that can be removed
        found = False
        for r in range(current_rank):
            test_U = [row[:] for row in current_U]
            test_V = [row[:] for row in current_V]
            test_W = [row[:] for row in current_W]
            
            # Remove column r
            for row in test_U:
                del row[r]
            for row in test_V:
                del row[r]
            for row in test_W:
                del row[r]
            
            if _verify_decomposition_fast(test_U, test_V, test_W, n, m, p):
                current_U = test_U
                current_V = test_V
                current_W = test_W
                current_rank -= 1
                found = True
                break
        
        if not found:
            break
    
    return current_U, current_V, current_W, current_rank

def _try_column_merging(U, V, W, n, m, p, max_iter=100):
    """
    Try to merge pairs of columns if their sum can be represented with fewer terms.
    This is a more aggressive reduction strategy.
    """
    start = time.time()
    current_U, current_V, current_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    current_rank = len(current_U[0])
    n_iter = 0
    
    while current_rank > 2 and n_iter < max_iter and (time.time() - start) < 5:
        if _check_time():
            break
        n_iter += 1
        merged = False
        # Try merging pairs of columns
        for r1 in range(current_rank):
            if merged:
                break
            for r2 in range(r1 + 1, current_rank):
                # Try to see if columns r1 and r2 can be merged
                # by checking if their combined contribution can be represented
                # with a single rank-1 term
                test_U = [row[:] for row in current_U]
                test_V = [row[:] for row in current_V]
                test_W = [row[:] for row in current_W]
                
                # Remove columns r2 and r1 (higher index first)
                for row in test_U:
                    del row[r2]
                    del row[r1]
                for row in test_V:
                    del row[r2]
                    del row[r1]
                for row in test_W:
                    del row[r2]
                    del row[r1]
                
                # Try to find a new column that represents the sum
                # The sum of two rank-1 tensors is generally not rank 1,
                # but we can try random linear combinations
                for _ in range(5):
                    alpha = random.uniform(-2, 2)
                    beta = random.uniform(-2, 2)
                    
                    new_U_col = [alpha * current_U[i][r1] + beta * current_U[i][r2] for i in range(n*m)]
                    new_V_col = [alpha * current_V[j][r1] + beta * current_V[j][r2] for j in range(m*p)]
                    new_W_col = [alpha * current_W[k][r1] + beta * current_W[k][r2] for k in range(p*n)]
                    
                    test_U.append(new_U_col)
                    test_V.append(new_V_col)
                    test_W.append(new_W_col)
                    
                    if _verify_decomposition_fast(test_U, test_V, test_W, n, m, p):
                        current_U = test_U
                        current_V = test_V
                        current_W = test_W
                        current_rank -= 1
                        merged = True
                        break
                    else:
                        # Undo
                        test_U.pop()
                        test_V.pop()
                        test_W.pop()
                
                if merged:
                    break
    
    return current_U, current_V, current_W, current_rank

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = len(U[0])
    
    # Try to reduce rank
    for target in range(best_rank - 1, 60, -1):
        if _check_time():
            break
        U_test, V_test, W_test, rank_test = _try_reduce_rank(U, V, W, n, m, p, target, time_limit=10)
        if rank_test < best_rank:
            best_rank = rank_test
            U, V, W = U_test, V_test, W_test
    
    return {"U": U, "V": V, "W": W}
