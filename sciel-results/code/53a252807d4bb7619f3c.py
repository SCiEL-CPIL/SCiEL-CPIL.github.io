import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def _verify(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if i == k else 0.0
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - expected) > tol:
                    return False
    return True

def _verify_3x3(U, V, W, tol=1e-10):
    """Verify rank-r decomposition for 3x3 x 3x3 -> 3x3 with delta_{i,k}"""
    n = 3
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * n or V.shape[0] != n * n or W.shape[0] != n * n:
        return False
    for i in range(n):
        for j in range(n):
            for k in range(n):
                expected = 1.0 if i == k else 0.0
                val = 0.0
                for c in range(rank):
                    val += U[i * n + j, c] * V[j * n + k, c] * W[k * n + i, c]
                if abs(val - expected) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _find_3x3_decomposition(target_rank, max_attempts=500, deadline=None, seed=None):
    """Find a rank-r decomposition for 3x3 x 3x3 -> 3x3 with delta_{i,k}"""
    n = 3
    nn = n * n  # 9
    rng = random.Random(seed)
    
    # Try different value sets and sparsity patterns
    val_sets = [
        [0, 1, -1],
        [0, 1, -1, 2, -2],
        [0, 0.5, 1, -0.5, -1],
        [0, 1, -1, 1.5, -1.5, 2, -2],
    ]
    
    for vs_idx, vals in enumerate(val_sets):
        for attempt in range(max_attempts // len(val_sets)):
            if deadline is not None and time.time() > deadline:
                return None
            
            # Generate sparse matrices with structured non-zero patterns
            U = np.zeros((nn, target_rank))
            V = np.zeros((nn, target_rank))
            W = np.zeros((nn, target_rank))
            
            for c in range(target_rank):
                # Each column: pick non-zero entries with some structure
                # Bias toward diagonal-like patterns
                for _ in range(rng.randint(2, 4)):
                    ri = rng.randrange(nn)
                    U[ri, c] = rng.choice(vals)
                for _ in range(rng.randint(2, 4)):
                    rj = rng.randrange(nn)
                    V[rj, c] = rng.choice(vals)
                for _ in range(rng.randint(2, 4)):
                    rk = rng.randrange(nn)
                    W[rk, c] = rng.choice(vals)
            
            if _verify_3x3(U, V, W):
                return U, V, W
    
    return None

def _combine_3x3_to_3x6(U3, V3, W3):
    """Combine two 3x3 decompositions into a 3x3 x 3x6 decomposition"""
    r = U3.shape[1]
    
    U = np.zeros((9, 2 * r))
    V = np.zeros((18, 2 * r))
    W = np.zeros((18, 2 * r))
    
    # First block: k = 0,1,2
    U[:, 0:r] = U3
    V[0:9, 0:r] = V3
    W[0:9, 0:r] = W3
    
    # Second block: k = 3,4,5
    U[:, r:2*r] = U3
    V[9:18, r:2*r] = V3
    W[9:18, r:2*r] = W3
    
    return U, V, W

def _try_direct_search(target_rank, max_attempts=100, deadline=None, seed=None):
    """Direct random search for the full 3x3 x 3x6 problem"""
    n, m, p = DIMENSIONS
    um, vm, wm = n*m, m*p, p*n
    rng = random.Random(seed)
    
    vals = [0, 1, -1, 0.5, -0.5]
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            return None
        
        U = np.zeros((um, target_rank))
        V = np.zeros((vm, target_rank))
        W = np.zeros((wm, target_rank))
        
        for c in range(target_rank):
            for _ in range(rng.randint(2, 4)):
                ri = rng.randrange(um)
                U[ri, c] = rng.choice(vals)
            for _ in range(rng.randint(2, 4)):
                rj = rng.randrange(vm)
                V[rj, c] = rng.choice(vals)
            for _ in range(rng.randint(2, 4)):
                rk = rng.randrange(wm)
                W[rk, c] = rng.choice(vals)
        
        if _verify(U, V, W):
            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    
    # Fallback: schoolbook
    best = _schoolbook()
    best_rank = 54
    
    n, m, p = DIMENSIONS
    
    # Strategy 1: Find 3x3 decomposition, combine two copies
    for target_rank_3x3 in range(19, 27):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        deadline = DEADLINE - SAFETY_MARGIN
        result = _find_3x3_decomposition(target_rank_3x3, max_attempts=300, 
                                          deadline=deadline, seed=target_rank_3x3 * 1000)
        if result is not None:
            U3, V3, W3 = result
            U, V, W = _combine_3x3_to_3x6(U3, V3, W3)
            
            if _verify(U, V, W):
                total_rank = 2 * target_rank_3x3
                if total_rank < best_rank:
                    best_rank = total_rank
                    best = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                # Don't break - try lower ranks
                break
    
    # Strategy 2: Direct search for full problem
    for target_rank in range(38, 54):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        deadline = DEADLINE - SAFETY_MARGIN
        result = _try_direct_search(target_rank, max_attempts=100,
                                     deadline=deadline, seed=target_rank * 777)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                break
    
    return best
