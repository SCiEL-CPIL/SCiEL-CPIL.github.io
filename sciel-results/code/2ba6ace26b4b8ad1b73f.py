"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """
    Check exact decomposition:
    T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    where T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    """
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    
    # Check non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if val != 1:
                    return False
    
    # Check zero entries where j != j2
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j == j2:
                    continue
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if val != 0:
                        return False
    
    # Check zero entries where i != i2
    for i in range(n):
        for i2 in range(n):
            if i == i2:
                continue
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i2
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if val != 0:
                        return False
    
    # Check zero entries where k != k2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for k2 in range(p):
                    if k == k2:
                        continue
                    a = i * m + j
                    b = j * p + k2
                    c = k * n + i
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if val != 0:
                        return False
    
    return True


def _construct_schoolbook():
    """
    Construct a rank-30 decomposition using the schoolbook algorithm.
    """
    n, m, p = 2, 3, 5
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    
    return U, V, W


def _try_rank_search(rank, n, m, p, max_attempts=100, deadline=None):
    """
    Try to find a valid decomposition of the given rank using random search.
    Returns (U, V, W) if found, None otherwise.
    """
    start_time = time.time()
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            return None
        
        # Random matrices with entries from {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(n * m, rank))
        V = np.random.randint(-1, 2, size=(m * p, rank))
        W = np.random.randint(-1, 2, size=(p * n, rank))
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None


def _try_greedy_sparse_recovery(n, m, p, target_rank, deadline=None):
    """
    Try to find a sparse decomposition using greedy matching pursuit.
    This is a heuristic approach that may find lower rank solutions.
    """
    start_time = time.time()
    if deadline is not None and time.time() > deadline:
        return None
    
    # Build the tensor T as a 3D array
    T = np.zeros((n * m, m * p, p * n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Greedy algorithm: iteratively add rank-1 terms that best approximate the residual
    U = np.zeros((n * m, target_rank), dtype=object)
    V = np.zeros((m * p, target_rank), dtype=object)
    W = np.zeros((p * n, target_rank), dtype=object)
    residual = T.copy()
    
    for r in range(target_rank):
        if deadline is not None and time.time() > deadline:
            return None
        
        # Find the best rank-1 term to add
        best_score = -1
        best_a, best_b, best_c = None, None, None
        
        # Sample a subset of possible rank-1 terms
        n_samples = 1000
        for _ in range(n_samples):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            
            # Compute the score: how well does this rank-1 term match the residual?
            # For a rank-1 term e_a ⊗ e_b ⊗ e_c, the score is T[a,b,c]
            score = abs(residual[a, b, c])
            
            if score > best_score:
                best_score = score
                best_a, best_b, best_c = a, b, c
        
        if best_a is None or best_score == 0:
            break
        
        # Add this rank-1 term
        U[best_a, r] = 1
        V[best_b, r] = 1
        W[best_c, r] = 1
        
        # Update residual
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    residual[a, b, c] = residual[a, b, c] - U[a, r] * V[b, r] * W[c, r]
    
    # Check if we achieved an exact decomposition
    if _verify(U, V, W, n, m, p):
        return U, V, W
    
    return None


def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    U, V, W = _construct_schoolbook()
    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # Upper bound: n*m*p = 30
    
    # Try ranks from 15 upward
    for rank in range(15, best_rank):
        if time.time() > deadline - 20:
            break
        
        # Try random search
        result = _try_rank_search(rank, n, m, p, max_attempts=50, deadline=deadline - 20)
        if result is not None:
            U, V, W = result
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank
            break
        
        # Try greedy sparse recovery
        result = _try_greedy_sparse_recovery(n, m, p, rank, deadline=deadline - 20)
        if result is not None:
            U, V, W = result
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank
            break
    
    return best

# EVOLVE-BLOCK-END
