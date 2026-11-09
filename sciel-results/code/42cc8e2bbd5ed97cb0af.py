"""Matrix multiplication 3x4x8 tensor decomposition search."""
import itertools
import random
import time
import numpy as np

DIMENSIONS = [3, 4, 8]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 96

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
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

def _verify(U, V, W, tol=1e-9):
    """Check if U,V,W exactly decompose the matrix multiplication tensor.
    The tensor T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to C[i,k].
    Convention: W index is k*n+i.
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    """
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Use einsum for efficient verification
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(result, 1.0, atol=tol)

def _verify_fast(U, V, W, tol=1e-9):
    """Faster verification using numpy."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # For each (i,j,k), sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # Use einsum
    # U_r[i,j,c], V_r[j,k,c], W_r[k,i,c] -> sum over c, result[i,j,k]
    result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(result, 1.0, atol=tol)

def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < tol

def _all_half_integers(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val, tol):
                    return False
    return True

def _random_search(start_rank, max_rank, num_trials, deadline):
    """Random search for a decomposition of given rank."""
    n, m, p = N, M, P
    best = None
    best_rank = max_rank + 1
    rng = random.Random(42)
    
    for rank in range(start_rank, max_rank + 1):
        if time.time() > deadline:
            break
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            # Generate random half-integer matrices with sparsity
            density = 0.3  # 30% non-zero entries
            U = [[rng.choice([-1, 0, 1]) if rng.random() < density else 0 for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, 0, 1]) if rng.random() < density else 0 for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, 0, 1]) if rng.random() < density else 0 for _ in range(rank)] for _ in range(p * n)]
            
            if _verify_fast(U, V, W):
                if rank < best_rank:
                    best_rank = rank
                    best = {"U": U, "V": V, "W": W}
                    return best
    return best

def _structured_search(deadline):
    """Try structured patterns that might yield lower rank."""
    n, m, p = N, M, P
    best = None
    best_rank = RANK_SCHOOLBOOK
    
    # Try rank 90-95 with some zero patterns
    for rank in range(90, 96):
        if time.time() > deadline:
            break
        # Start from schoolbook and try to merge columns
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Initialize with schoolbook-like pattern
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < rank:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
        
        if _verify_fast(U, V, W):
            if rank < best_rank:
                best_rank = rank
                best = {"U": U, "V": V, "W": W}
    
    return best

def _try_reductions_from_schoolbook(deadline):
    """Try to reduce schoolbook rank by merging similar terms."""
    n, m, p = N, M, P
    best = None
    best_rank = RANK_SCHOOLBOOK
    
    # Start with schoolbook
    result = _schoolbook()
    U = result["U"]
    V = result["V"]
    W = result["W"]
    current_rank = RANK_SCHOOLBOOK
    
    # Try to merge columns that have similar patterns
    # For each pair of columns, check if we can combine them
    for merge_round in range(20):
        if time.time() > deadline:
            break
        
        merged = False
        for c1 in range(current_rank):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, current_rank):
                if time.time() > deadline:
                    break
                
                # Try to see if column c2 can be expressed in terms of column c1
                # or if we can modify both to reduce rank
                # Simple heuristic: check if they have the same support pattern
                U_c1 = [U[i][c1] for i in range(n * m)]
                U_c2 = [U[i][c2] for i in range(n * m)]
                V_c1 = [V[i][c1] for i in range(m * p)]
                V_c2 = [V[i][c2] for i in range(m * p)]
                W_c1 = [W[i][c1] for i in range(p * n)]
                W_c2 = [W[i][c2] for i in range(p * n)]
                
                # If columns are identical, we can potentially merge
                if U_c1 == U_c2 and V_c1 == V_c2 and W_c1 == W_c2:
                    # Remove column c2
                    new_rank = current_rank - 1
                    U_new = [row[:c2] + row[c2+1:] for row in U]
                    V_new = [row[:c2] + row[c2+1:] for row in V]
                    W_new = [row[:c2] + row[c2+1:] for row in W]
                    
                    if _verify_fast(U_new, V_new, W_new):
                        U, V, W = U_new, V_new, W_new
                        current_rank = new_rank
                        merged = True
                        best = {"U": U, "V": V, "W": W}
                        best_rank = current_rank
                        break
            
            if merged:
                break
        
        if not merged:
            break
    
    return best if best is not None else result

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Start with schoolbook
    result = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Try reductions from schoolbook
    if time.time() < deadline:
        r = _try_reductions_from_schoolbook(deadline)
        if r is not None:
            result = r
            best_rank = len(result["U"][0])
    
    # Try random search for lower ranks
    if time.time() < deadline:
        r = _random_search(max(1, best_rank - 10), best_rank - 1, 5000, deadline)
        if r is not None:
            result = r
            best_rank = len(result["U"][0])
    
    return result
