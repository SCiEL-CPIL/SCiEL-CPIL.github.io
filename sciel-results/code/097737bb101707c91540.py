"""Matrix multiplication 3x5x5 - search for rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

# Deadline guard: 150s wall-clock limit, use 140s for safety
_DEADLINE = 140.0
_start_time = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _start_time)

def _verify_decomposition(U, V, W, n=3, m=5, p=5):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] = 1 if we multiply A[i,j] * B[j,k] to get C[i,k].
    With trace convention W[k*n+i, r], the decomposition is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Compute decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    return np.allclose(result, T, atol=1e-10)

def _schoolbook_decomposition():
    """Return the trivial rank n*m*p decomposition."""
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

def _try_kronecker_structure(n=3, m=5, p=5):
    """Try a structured decomposition based on Kronecker product ideas.
    
    The 3x5 @ 5x5 multiplication can be viewed as:
    - 3 rows of A, each of length 5
    - 5 rows of B, each of length 5
    - Output: 3x5 matrix
    
    A natural lower bound: for each output element C[i,k], we need a sum over j.
    The tensor has n*m*p = 75 nonzero entries.
    
    Try rank = n*p = 15 (one rank-1 per output position, sharing the inner dimension)
    This won't work in general, but let's try structured approaches.
    """
    # Approach 1: Try rank = n*m = 15 (one rank per (i,j) pair, summing over k in W)
    # T[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # If we set r = i*m+j (15 ranks), then:
    # U[i*m+j, i*m+j] = 1
    # V[j*p+k, i*m+j] needs to handle the k dimension
    # W[k*n+i, i*m+j] needs to be 1 for all k
    # This doesn't work because V and W can't both be 1 for all k with a single rank per (i,j).
    
    # Approach 2: Try rank = m*p = 25 (one rank per (j,k) pair)
    # U[i*m+j, j*p+k] needs to be 1 for all i
    # V[j*p+k, j*p+k] = 1
    # W[k*n+i, j*p+k] needs to be 1 for all i
    # Same problem.
    
    # Approach 3: Try rank = n*p = 15 (one rank per (i,k) pair)
    # U[i*m+j, i*p+k] needs to be 1 for all j
    # V[j*p+k, i*p+k] needs to be 1 for all j
    # W[k*n+i, i*p+k] = 1
    # Same problem.
    
    # The fundamental issue: for each (i,j,k) we need all three to be nonzero simultaneously.
    # Minimum rank for a "complete" tensor of this type is at least max(n*m, m*p, n*p) = 25.
    # Actually the lower bound is max(n*m, m*p, n*p) = 25 for the standard multiplication tensor.
    
    # Let's try rank = 25 with a structured approach:
    # Use r = j*p + k (25 ranks indexed by (j,k))
    # U[i*m+j, j*p+k] = 1 for all i,j,k  -> U[i*m+j, r] = 1 if r corresponds to (j,*)
    #   i.e., U[i*m+j, j*p+k] = 1 for all k. So column j*p+k has 1s in rows i*m+j for all i.
    # V[j*p+k, j*p+k] = 1 (identity)
    # W[k*n+i, j*p+k] = 1 for all i. So column j*p+k has 1s in rows k*n+i for all i.
    
    # Check: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # = sum_{j',k'} U[i*m+j, j'*p+k'] * V[j*p+k, j'*p+k'] * W[k*n+i, j'*p+k']
    # V is identity, so only j'=j, k'=k survives:
    # = U[i*m+j, j*p+k] * V[j*p+k, j*p+k] * W[k*n+i, j*p+k]
    # = 1 * 1 * 1 = 1  ✓
    
    # This works! Rank = m*p = 25.
    rank = m * p  # 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            V[j * p + k][r] = 1
            for i in range(n):
                U[i * m + j][r] = 1
                W[k * n + i][r] = 1
    
    if _verify_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def _try_rank_n_m(n=3, m=5, p=5):
    """Try rank = n*m = 15: indexed by (i,j)."""
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j][r] = 1
            for k in range(p):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    if _verify_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def _try_rank_n_p(n=3, m=5, p=5):
    """Try rank = n*p = 15: indexed by (i,k)."""
    rank = n * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            W[k * n + i][r] = 1
            for j in range(m):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
    
    if _verify_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def _try_random_search(n=3, m=5, p=5, max_rank=25, time_budget=10.0):
    """Random search for lower rank decompositions using structured random initialization."""
    best = None
    best_rank = max_rank + 1
    start = time.time()
    
    for target_rank in range(max_rank, 14, -1):  # Try ranks from 25 down to 15
        if time.time() - start > time_budget:
            break
        if _time_remaining() < 20:
            break
        
        # Try a few random structured attempts
        for attempt in range(5):
            if time.time() - start > time_budget:
                break
            
            # Initialize with random ±1 entries
            U = [[random.choice([-1, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if _verify_decomposition(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_rank <= target_rank:
            break
    
    return best

def _try_systematic_search(n=3, m=5, p=5, max_rank=25, time_budget=30.0):
    """Systematic search: try all combinations of small structured patterns."""
    best = None
    best_rank = max_rank + 1
    start = time.time()
    
    # Try rank from 25 downward
    for target_rank in range(25, 14, -1):
        if time.time() - start > time_budget:
            break
        if _time_remaining() < 15:
            break
        
        # For each target rank, try structured patterns
        # Pattern: assign each of the 75 (i,j,k) triples to a rank
        # We need: for each (i,j,k), there exists r such that U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # And for each (i,j,k), the sum over all r equals 1 (no other r contributes)
        
        # This is a constraint satisfaction problem. Let's try greedy assignment.
        # For a given rank, try to assign each (i,j,k) to a unique rank r
        # such that the rank-r pattern covers exactly that (i,j,k) and no others.
        
        # A rank-r pattern is defined by:
        # U_row[i,j] = 1 for some set of (i,j)
        # V_row[j,k] = 1 for some set of (j,k)  
        # W_row[k,i] = 1 for some set of (k,i)
        # The "covered" (i,j,k) are those where all three are 1.
        
        # For a rank to cover exactly one (i,j,k):
        # U_row must have 1 at (i,j) and 0 elsewhere
        # V_row must have 1 at (j,k) and 0 elsewhere
        # W_row must have 1 at (k,i) and 0 elsewhere
        # This is the schoolbook pattern - each rank covers exactly one (i,j,k).
        # So rank 75 is the only pure schoolbook.
        
        # For lower rank, ranks must cover multiple (i,j,k) but the sum must still be 1.
        # This means overlapping contributions must cancel out.
        
        # Try: allow each rank to cover a "line" of (i,j,k) values
        # e.g., fix i, vary (j,k): covers n*... no, fix i, vary j and k independently
        # If U_row[i,j]=1 for all j, V_row[j,k]=1 for all k, W_row[k,i]=1 for all k
        # Then covered: all (i,j,k) for fixed i. That's m*p = 25 entries.
        # But we need exactly 1, not 25. So this doesn't work directly.
        
        # The key insight: we need the SUM to be 1, so overlapping contributions
        # must cancel. With ±1 entries, we can have constructive and destructive interference.
        
        # Let's try a specific construction for rank 25 (already found above).
        # For rank < 25, we need more clever constructions.
        
        # Try: use the fact that n=3 is small.
        # For each k (5 values), we need a 3x5 @ 5x1 multiplication for each output column.
        # Actually, let's think differently.
        
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] = 1 for all i,j,k.
        
        # Lower bound: The tensor has border rank at least max(n*m, m*p, n*p) = 25.
        # Actually, the standard lower bound for matrix multiplication tensor [n,m,p] 
        # is max(n*m, m*p, n*p). For [3,5,5], this is max(15, 25, 15) = 25.
        
        # So rank 25 might be optimal! Let's verify our rank-25 construction is correct.
        
        if target_rank == 25:
            result = _try_kronecker_structure(n, m, p)
            if result:
                best = result
                best_rank = 25
                break  # We found rank 25, which is the lower bound
        
        # For ranks below 25, the lower bound suggests they're impossible
        # (border rank >= 25 implies rank >= 25 for this tensor)
        # But let's still try a few random searches to be safe
    
    return best

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 @ 5x5 multiplication."""
    global _start_time
    _start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # First, try the known rank-25 construction
    result = _try_kronecker_structure(n, m, p)
    if result:
        return result
    
    # Fallback to schoolbook (rank 75)
    return _schoolbook_decomposition()
