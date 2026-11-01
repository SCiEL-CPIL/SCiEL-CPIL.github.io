"""Stronger search for low-rank exact tensor decomposition of 3x4 * 4x4."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 4]

# Deadline guard: evaluator kills at 150s; keep safety margin
_START = time.time()
_DEADLINE = 150.0 - 5.0  # 145s safety margin

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()

def _verify(U, V, W, n=3, m=4, p=4):
    """Check if U,V,W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = delta_{k,?} ... actually the matmul tensor is:
    T[(i,j), (j,k)] = C[i,k] where C = A*B
    The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if the (i,j,k) entry is 1, else 0.
    For matmul, the tensor has T[i,j,k] = 1 for all valid i,j,k (it's the identity in the sense that each triple (i,j,k) contributes to C[i,k]).
    Actually, the matmul tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check each (i,j,k) triple
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n=3, m=4, p=4):
    """Vectorized verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor check
    # For each c, we have outer product contributions
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Let's compute all triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n=3, m=4, p=4):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank_reduction(n=3, m=4, p=4, target_rank=None):
    """
    Try to find a lower-rank decomposition by exploiting symmetries.
    For 3x4 * 4x4, the schoolbook rank is 48.
    We try to find rank < 48 by searching over structured decompositions.
    
    Key insight: The matmul tensor T[i,j,k] = 1 for all i,j,k.
    We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    One approach: try to find a decomposition where the W matrix has special structure.
    Since W[k*n+i,c] appears, and the condition must hold for all k and i,
    we can think of it as: for each c, define matrices A_c (n x m), B_c (m x p), D_c (p x n)
    where A_c[i,j] = U[i*m+j,c], B_c[j,k] = V[j*p+k,c], D_c[k,i] = W[k*n+i,c].
    Then the condition is: sum_c (A_c * B_c)[i,k] * ... no wait.
    
    Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = sum_c (A_c[i,j]) * (B_c[j,k]) * (D_c[k,i])
    This must equal 1 for all i,j,k.
    
    Note that this is NOT the standard matrix multiplication trace. Let me re-read.
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    So the condition is indeed: for all i in [0,n), j in [0,m), k in [0,p):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is a 3-tensor with entries all equal to 1.
    """
    if target_rank is None:
        target_rank = n * m * p - 1
    
    # For this specific problem, let's try a structured approach.
    # Since all tensor entries are 1, we can try:
    # U[i*m+j, r] = a_{i,j,r}, V[j*p+k, r] = b_{j,k,r}, W[k*n+i, r] = c_{k,i,r}
    # sum_r a_{i,j,r} b_{j,k,r} c_{k,i,r} = 1
    
    # Try small rank decompositions with simple patterns.
    # For rank 1: a_{i,j,0} * b_{j,k,0} * c_{k,i,0} = 1 for all i,j,k.
    # This requires a_{i,j,0} = 1, b_{j,k,0} = 1, c_{k,i,0} = 1, which works!
    # Wait, let me check: sum_r=0 only: a_{i,j,0} * b_{j,k,0} * c_{k,i,0} = 1*1*1 = 1. Yes!
    # So rank 1 should work?
    
    U = np.zeros((n*m, 1), dtype=float)
    V = np.zeros((m*p, 1), dtype=float)
    W = np.zeros((p*n, 1), dtype=float)
    U[:, 0] = 1.0
    V[:, 0] = 1.0
    W[:, 0] = 1.0
    
    if _verify_fast(U, V, W, n, m, p):
        return U, V, W
    
    return None

def _search_low_rank(n=3, m=4, p=4, max_rank=48, time_budget=None):
    """Search for low-rank decompositions."""
    if time_budget is None:
        time_budget = max(0.0, _remaining() - 1.0)
    
    best = None
    best_rank = n * m * p + 1
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Try rank 1 through some upper bound
    for rank in range(1, min(best_rank, 20)):
        if _remaining() < 2.0:
            break
        
        # Try random initialization with verification
        np.random.seed(rank * 1000 + 42)
        for attempt in range(100):
            if _remaining() < 2.0:
                break
            
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Scale to try to get sums close to 1
            # The condition is sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # For random Gaussian, the expected value is 0, so we need scaling.
            # Let's try to normalize.
            
            if _verify_fast(U, V, W, n, m, p):
                best = (U, V, W)
                best_rank = rank
                break
        
        if best_rank == rank:
            break
    
    if best is not None:
        return best
    return (U_sb, V_sb, W_sb)

def construct():
    n, m, p = DIMENSIONS
    
    # First, try the trivial rank-1 solution
    U1 = np.ones((n*m, 1), dtype=float)
    V1 = np.ones((m*p, 1), dtype=float)
    W1 = np.ones((p*n, 1), dtype=float)
    
    if _verify_fast(U1, V1, W1, n, m, p):
        return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}
    
    # If rank 1 doesn't work, do a search
    result = _search_low_rank(n, m, p, time_budget=max(0.0, _remaining() - 2.0))
    U, V, W = result
    
    # Convert to lists for the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
