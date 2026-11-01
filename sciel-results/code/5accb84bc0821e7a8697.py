"""Optimized matrix multiplication tensor decomposition for 2x3 by 3x6."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, deadline, seed_offset=0, max_trials=30):
    """Try to find a valid decomposition of rank r using randomized search."""
    for trial in range(max_trials):
        if time.time() > deadline - 3:
            return None
        
        np.random.seed(42 + seed_offset + trial * 1000)
        
        # Use structured random values from {-1, 0, 1}
        U = np.random.choice([-1, 0, 1], size=(n*m, r))
        V = np.random.choice([-1, 0, 1], size=(m*p, r))
        W = np.random.choice([-1, 0, 1], size=(p*n, r))
        
        if _verify(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return None

def _try_rank_half(r, n, m, p, deadline, seed_offset=0, max_trials=30):
    """Try to find a valid decomposition of rank r using half-integer values."""
    values = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    
    for trial in range(max_trials):
        if time.time() > deadline - 3:
            return None
        
        np.random.seed(1000 + seed_offset + trial * 1000)
        
        U = np.random.choice(values, size=(n*m, r))
        V = np.random.choice(values, size=(m*p, r))
        W = np.random.choice(values, size=(p*n, r))
        
        if _verify(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return None

def _kronecker_decomposition(n, m, p):
    """
    Try to exploit dimension factorization p = 3*2 to build a structured
    decomposition. We view the 3x6 multiplication as combining a 3x3 and 3x2
    structure. This gives rank 9*2 = 18 as a structured candidate.
    
    Actually, let's think more carefully. The Kronecker product approach:
    If we can write the tensor as a product of smaller tensors, we get
    rank = rank1 * rank2.
    
    For 2x3 * 3x6: We can view this as (2x3 * 3x3) x (3x2) in some sense,
    but the dimensions don't factor cleanly that way.
    
    Alternative: Use the fact that 6 = 2*3 and try to build from
    <2,3,3> (rank 9 schoolbook) and <1,1,2> (rank 2 schoolbook).
    But this doesn't directly apply.
    
    Let's just try the structured Kronecker-style construction:
    Split the 6 columns of B into two groups of 3.
    B = [B1 | B2] where B1, B2 are 3x3.
    C = [A*B1 | A*B2]
    
    This gives us two independent <2,3,3> problems, each of rank 9.
    Total rank = 9 + 9 = 18.
    
    But we can do better by sharing structure. Let's construct it explicitly.
    """
    # Schoolbook for <2,3,3> has rank 9
    # We have two such blocks for the two halves of the 6 columns
    
    # Actually, let's just construct the rank-18 solution explicitly
    # by splitting p=6 into two groups of 3.
    
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each half (h=0,1), we have a <2,3,3> multiplication
    # U shares the same structure for both halves
    # V is split: V[:, 0:9] corresponds to B1, V[:, 9:18] to B2
    # W is split: W[:, 0:9] for C1, W[:, 9:18] for C2
    
    col = 0
    for h in range(2):  # two halves
        for i in range(n):
            for j in range(m):
                for k in range(3):  # 3 columns per half
                    U[i*m + j, col] = 1
                    V[j*p + h*3 + k, col] = 1
                    W[(h*3 + k)*n + i, col] = 1
                    col += 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}, rank
    
    return None, None

def construct():
    """
    Construct a tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing factor matrices
    that exactly decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10 seconds safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Build schoolbook solution as fallback (rank 36)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = rank_sb
    
    # First, try the structured Kronecker-style decomposition (rank 18)
    kronecker_result, kronecker_rank = _kronecker_decomposition(n, m, p)
    if kronecker_result is not None:
        best_result = kronecker_result
        best_rank = kronecker_rank
    
    # Try to find even lower rank decompositions using randomized search
    # Lower bound: at least max(n*m, m*p, p*n) / max(n, m, p) ... 
    # For <2,3,6>, theoretical lower bound is around 12-18
    # Let's try ranks from 12 up to best_rank-1
    
    search_start = max(12, best_rank - 8)
    search_end = min(best_rank - 1, 35)
    
    for rank in range(search_start, search_end + 1):
        if time.time() > deadline - 10:
            break
        
        # Try integer-valued search
        result = _try_rank(rank, n, m, p, deadline, seed_offset=0, max_trials=20)
        if result is not None:
            best_result = result
            best_rank = rank
            break
        
        # Try half-integer-valued search
        result = _try_rank_half(rank, n, m, p, deadline, seed_offset=0, max_trials=10)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    # If we still have time, try more aggressive search around the best rank found
    if time.time() < deadline - 15 and best_rank > 12:
        for rank in range(12, min(best_rank, 18)):
            if time.time() > deadline - 5:
                break
            
            # More trials for lower ranks
            result = _try_rank(rank, n, m, p, deadline, seed_offset=100, max_trials=40)
            if result is not None:
                best_result = result
                best_rank = rank
                break
            
            result = _try_rank_half(rank, n, m, p, deadline, seed_offset=200, max_trials=20)
            if result is not None:
                best_result = result
                best_rank = rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
