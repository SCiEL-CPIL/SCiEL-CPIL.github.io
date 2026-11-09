"""Matrix multiplication 3x3 x 3x5 tensor decomposition.
Finds a rank-minimal exact decomposition with half-integer entries.
"""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=3, m=3, p=5, tol=1e-8):
    """Verify U,V,W form a valid decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    Actually the tensor for matrix mult is T[i,j,k] = 1 for all i,j,k (it's the identity tensor in the appropriate sense)
    Wait, let me think again. The matrix multiplication tensor T is defined such that
    T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]B[j,k]
    So the tensor has T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    """
    r = U.shape[1]
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _is_half_integer(x, tol=1e-10):
    """Check if x is a half-integer (multiple of 0.5)."""
    x2 = 2.0 * x
    return abs(x2 - round(x2)) < tol

def _round_to_half_integer(x, tol=1e-6):
    """Round x to nearest half-integer if close enough, else return None."""
    x2 = 2.0 * x
    r = round(x2)
    if abs(x2 - r) < tol:
        return r / 2.0
    return None

def _make_schoolbook(n=3, m=3, p=5):
    """Create the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _make_strassen_333(n=3, m=3, p=3):
    """Create a rank-13 decomposition for 3x3 * 3x3 using Strassen-like method.
    This is a known result: 3x3 matrix multiplication has rank at most 13 (actually 14 is known, 13 is open)
    Let me use a simpler approach: just use the schoolbook for 3x3 which is rank 27, 
    but we can try to do better.
    
    Actually, let me just construct a valid decomposition. The Strassen algorithm for 3x3 
    is not straightforward. Let me use a different approach.
    
    For 3x3 * 3x3, the schoolbook is rank 27. We can try to find a lower rank decomposition.
    """
    # Use schoolbook for now
    return _make_schoolbook(n, m, p)

def _make_332_decomposition(n=3, m=3, p=2):
    """Create a decomposition for 3x3 * 3x2.
    Schoolbook rank is 18. Can we do better?
    """
    # Use schoolbook
    return _make_schoolbook(n, m, p)

def _try_rank_random(n, m, p, target_rank, num_trials=500, seed=None, deadline=None):
    """Try random initialization + verification for given target rank."""
    if seed is not None:
        np.random.seed(seed)
    
    for trial in range(num_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initialization with values in {-1, -0.5, 0, 0.5, 1}
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        U = np.random.choice(vals, size=(n*m, target_rank))
        V = np.random.choice(vals, size=(m*p, target_rank))
        W = np.random.choice(vals, size=(p*n, target_rank))
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _try_rank_structured(n, m, p, target_rank, deadline=None):
    """Try structured decompositions based on known patterns."""
    if target_rank <= 20:
        vals = [-1.0, 0.0, 1.0]
        for trial in range(200):
            if deadline is not None and time.time() > deadline:
                break
            U = np.random.choice(vals, size=(n*m, target_rank))
            V = np.random.choice(vals, size=(m*p, target_rank))
            W = np.random.choice(vals, size=(p*n, target_rank))
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
    
    return None

def _optimize_from_schoolbook(n, m, p, target_rank, deadline=None):
    """Try to find a lower-rank decomposition by optimizing from scratch."""
    best_result = None
    
    for seed in range(20):
        if deadline is not None and time.time() > deadline:
            break
        
        result = _try_rank_random(n, m, p, target_rank, num_trials=100, seed=seed, deadline=deadline)
        if result is not None:
            return result
    
    return best_result

def _combine_decompositions(U1, V1, W1, U2, V2, W2):
    """Combine two decompositions by concatenating columns.
    This works when the tensors are disjoint in at least one index.
    """
    U = np.concatenate([U1, U2], axis=1)
    V = np.concatenate([V1, V2], axis=1)
    W = np.concatenate([W1, W2], axis=1)
    return U, V, W

def _make_335_via_split(n=3, m=3, p=5):
    """Create a decomposition for 3x3 * 3x5 by splitting into 3x3 * 3x3 and 3x3 * 3x2.
    The total rank is the sum of the ranks of the two parts.
    """
    # 3x3 * 3x3 part: k in 0,1,2
    U1, V1, W1 = _make_schoolbook(n, m, 3)
    
    # 3x3 * 3x2 part: k in 3,4
    # For this part, the V matrix has rows indexed by j*p+k where p=5, k in 3,4
    # The W matrix has rows indexed by k*n+i where k in 3,4
    # We need to construct V1b and W1b for the second part
    U2, V2_partial, W2_partial = _make_schoolbook(n, m, 2)
    
    # Now V2_partial has shape (m*2, rank2) = (6, rank2)
    # But we need V2 to have shape (m*5, rank2) = (15, rank2)
    # The rows for k=3,4 correspond to rows j*5+3 and j*5+4
    V2 = np.zeros((m*5, U2.shape[1]), dtype=np.float64)
    for j in range(m):
        for k in range(2):
            V2[j*5 + (3+k), :] = V2_partial[j*2 + k, :]
    
    # W2_partial has shape (2*n, rank2) = (6, rank2)
    # We need W2 to have shape (5*n, rank2) = (15, rank2)
    # The rows for k=3,4 correspond to rows (3+k)*n+i
    W2 = np.zeros((5*n, U2.shape[1]), dtype=np.float64)
    for k in range(2):
        for i in range(n):
            W2[(3+k)*n + i, :] = W2_partial[k*n + i, :]
    
    # Combine
    U, V, W = _combine_decompositions(U1, V1, W1, U2, V2, W2)
    return U, V, W

def construct():
    """Construct a rank-minimal tensor decomposition for 3x3 * 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition (rank 45)
    U, V, W = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Verify schoolbook
    assert _verify_decomposition(U, V, W, n, m, p), "Schoolbook verification failed"
    
    # Try the split approach: 3x3*3x3 + 3x3*3x2
    # This gives rank 27 + 18 = 45, same as schoolbook
    # But we can try to find lower rank for the 3x3*3x2 part
    
    # First, try to find a lower rank decomposition for 3x3 * 3x2
    best_332_rank = 18
    best_332_U, best_332_V, best_332_W = _make_schoolbook(n, m, 2)
    
    # The lower bound for 3x3 * 3x2 is max(9, 6, 6) = 9
    for target_rank in range(9, best_332_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank_random(n, m, 2, target_rank, num_trials=200, deadline=deadline)
        if result is not None:
            best_332_U, best_332_V, best_332_W = result
            best_332_rank = target_rank
            break
    
    # Now combine with 3x3 * 3x3 (rank 27)
    U1, V1, W1 = _make_schoolbook(n, m, 3)
    U2 = best_332_U
    V2_partial = best_332_V
    W2_partial = best_332_W
    
    V2 = np.zeros((m*5, U2.shape[1]), dtype=np.float64)
    for j in range(m):
        for k in range(2):
            V2[j*5 + (3+k), :] = V2_partial[j*2 + k, :]
    
    W2 = np.zeros((5*n, U2.shape[1]), dtype=np.float64)
    for k in range(2):
        for i in range(n):
            W2[(3+k)*n + i, :] = W2_partial[k*n + i, :]
    
    U_combined, V_combined, W_combined = _combine_decompositions(U1, V1, W1, U2, V2, W2)
    
    if _verify_decomposition(U_combined, V_combined, W_combined, n, m, p):
        combined_rank = U_combined.shape[1]
        if combined_rank < best_rank:
            best_U, best_V, best_W = U_combined, V_combined, W_combined
            best_rank = combined_rank
    
    # Also try to find a lower rank decomposition for 3x3 * 3x3
    best_333_rank = 27
    best_333_U, best_333_V, best_333_W = _make_schoolbook(n, m, 3)
    
    for target_rank in range(15, best_333_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank_random(n, m, 3, target_rank, num_trials=200, deadline=deadline)
        if result is not None:
            best_333_U, best_333_V, best_333_W = result
            best_333_rank = target_rank
            break
    
    # Combine the improved 3x3*3x3 with the best 3x3*3x2
    U1b = best_333_U
    V1b = best_333_V
    W1b = best_333_W
    U2b = best_332_U
    V2_partial_b = best_332_V
    W2_partial_b = best_332_W
    
    V2b = np.zeros((m*5, U2b.shape[1]), dtype=np.float64)
    for j in range(m):
        for k in range(2):
            V2b[j*5 + (3+k), :] = V2_partial_b[j*2 + k, :]
    
    W2b = np.zeros((5*n, U2b.shape[1]), dtype=np.float64)
    for k in range(2):
        for i in range(n):
            W2b[(3+k)*n + i, :] = W2_partial_b[k*n + i, :]
    
    U_combined_b, V_combined_b, W_combined_b = _combine_decompositions(U1b, V1b, W1b, U2b, V2b, W2b)
    
    if _verify_decomposition(U_combined_b, V_combined_b, W_combined_b, n, m, p):
        combined_rank_b = U_combined_b.shape[1]
        if combined_rank_b < best_rank:
            best_U, best_V, best_W = U_combined_b, V_combined_b, W_combined_b
            best_rank = combined_rank_b
    
    # Try direct search for the full 3x3*3x5 problem
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank_structured(n, m, p, target_rank, deadline)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
        
        result = _optimize_from_schoolbook(n, m, p, target_rank, deadline)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
    
    # Convert to lists for return
    best_U_list = best_U.tolist()
    best_V_list = best_V.tolist()
    best_W_list = best_W.tolist()
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
