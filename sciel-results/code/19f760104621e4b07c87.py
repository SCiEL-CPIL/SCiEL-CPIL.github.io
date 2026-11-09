"""Matrix multiplication 2x3x5 - search for rank-minimal decomposition."""
import numpy as np
from fractions import Fraction
import time
import random

def _validate(U, V, W, n=2, m=3, p=5):
    """Validate that U, V, W form a valid decomposition."""
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # And for non-matching triples, it should be 0.
    # Actually, the tensor T has T[i,j,k] = 1 if this is a valid (i,j,k) triple, else 0.
    # The decomposition must satisfy: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is in the tensor
    # The tensor T_{n,m,p} has T[i,j,k] = 1 for all valid i,j,k (all of them in 2x3x5)
    # Wait, no. The matrix multiplication tensor is:
    # T[i,j,k] = A[i,j] * B[j,k] contributes to C[i,k]
    # So the tensor has 1s at all (i,j,k) positions where i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for all valid i,j,k.
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n=2, m=3, p=5):
    """Return schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}

def _try_random_rank(n, m, p, rank, seed=42):
    """Try to find a valid decomposition with given rank using random search."""
    rng = random.Random(seed)
    
    # The entries should be integers or half-integers.
    # Let's try small integer values: -1, 0, 1, 2, -2, etc.
    # Or half-integers: -0.5, 0, 0.5, 1, 1.5, etc.
    
    # Strategy: start from schoolbook and try to merge terms.
    # For now, try random initialization with small values.
    
    for attempt in range(100):
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Initialize with random small values
        # Use integers and half-integers
        vals = [-1, 0, 0, 0, 1, 2, -2, 0.5, -0.5, 1.5, -1.5]
        
        for i in range(n * m):
            for r in range(rank):
                if rng.random() < 0.3:
                    U[i][r] = rng.choice(vals)
        
        for j in range(m * p):
            for r in range(rank):
                if rng.random() < 0.3:
                    V[j][r] = rng.choice(vals)
        
        for k in range(p * n):
            for r in range(rank):
                if rng.random() < 0.3:
                    W[k][r] = rng.choice(vals)
        
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _merge_schoolbook_terms(n, m, p):
    """
    Try to merge schoolbook terms to reduce rank.
    
    The schoolbook has one term per (i,j,k) triple. If two terms can be combined,
    we reduce the rank.
    
    Two terms r1 and r2 can be merged if there exist coefficients such that:
    U1[r1] * V1[r1] * W1[r1] + U2[r2] * V2[r2] * W2[r2] can be expressed
    as a single rank-1 term, which is generally not possible unless they share structure.
    
    A better approach: look for linear dependencies among the terms.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor equation is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system of n*m*p = 30 equations.
    # Each equation involves products of three vectors.
    
    # Let's think of it differently. Define for each r, the "term" is the outer product
    # of U[:,r], V[:,r], W[:,r]. The sum of these outer products must equal the all-ones tensor.
    
    # The all-ones tensor T has T[i,j,k] = 1 for all i,j,k.
    # We need sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
    
    # This is a tensor decomposition problem. The rank is the minimum r such that
    # such U, V, W exist.
    
    # For now, let's just return the schoolbook.
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return the matrix multiplication tensor for 2x3 * 3x5 as a rank decomposition.
    U: (n*m, rank) = (6, rank)
    V: (m*p, rank) = (15, rank)
    W: (p*n, rank) = (10, rank)
    
    Indexing conventions:
    U[i*m + j, r] corresponds to A[i,j]
    V[j*p + k, r] corresponds to B[j,k]
    W[k*n + i, r] corresponds to C[i,k]
    """
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 30
    
    # Try to find lower rank decompositions
    # Lower bound is at least max(n*p, m*n, m*p) = max(10, 6, 15) = 15
    # Try ranks from 15 to 29
    
    for rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        # Try random search for this rank
        result = _try_random_rank(n, m, p, rank, seed=rank * 1000)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    return best
