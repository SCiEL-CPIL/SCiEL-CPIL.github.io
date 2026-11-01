"""Search for rank-minimal tensor decomposition of 2x3 * 3x3 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]
DEADLINE = time.time() + 140  # safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if the (i,j) of A times (j,k) of B contributes to C[i,k].
    Actually the tensor for matmul: T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    represents: C[i,k] += A[i,j] * B[j,k]
    So the tensor element T[i, j, k] = 1 for all valid (i,j,k).
    """
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W (each rank-sized) exactly decompose the matmul tensor.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The decomposition requires: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]... 
    Wait, the standard tensor decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    But here U is indexed by (i*m+j), V by (j*p+k), W by (k*n+i).
    
    The condition is: for all (i,j,k):
        sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]... no.
    
    Actually for a rank-r decomposition of the matmul tensor:
    T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    But wait, that's not right either. The standard form is:
    The matmul tensor T has T[i,j,k] = 1. A rank-r decomposition means:
    T[i,j,k] = sum_c U_i_j_c * V_j_k_c * W_i_k_c
    
    But the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    So U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    
    The condition: for all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer."""
    x2 = 2 * x
    return abs(x2 - round(x2)) < tol

def _snap_to_half_integer(x):
    """Snap x to nearest half-integer."""
    return round(2 * x) / 2.0

def _try_rank(r, n, m, p, max_iters=2000, seed=42):
    """Try to find a rank-r decomposition using randomized search."""
    if time.time() > DEADLINE - 5:
        return None
    
    rng = random.Random(seed)
    
    # Use a structured approach: start from schoolbook and try to merge terms
    # Or use random initialization with gradient-like refinement
    
    # Strategy: random rank-r matrices with small half-integer entries, then check
    # Better: use the structure. For rank r, we need r terms.
    
    # Let's try a more systematic approach:
    # Initialize with random half-integer values in {-1, -0.5, 0, 0.5, 1}
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            return None
        
        # Random initialization with half-integers
        vals = [-1, -0.5, 0, 0.5, 1]
        
        U = [[rng.choice(vals) for _ in range(r)] for _ in range(n*m)]
        V = [[rng.choice(vals) for _ in range(r)] for _ in range(m*p)]
        W = [[rng.choice(vals) for _ in range(r)] for _ in range(p*n)]
        
        if _check_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _try_rank_structured(r, n, m, p, max_iters=500):
    """Try structured rank-r decomposition."""
    if time.time() > DEADLINE - 5:
        return None
    
    # For small ranks, we can be more systematic
    # The key equations: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Try using the known structure: most entries should be 0, ±1, ±0.5
    # Let's try a local search from a random feasible point
    
    vals = [-1, -0.5, 0, 0.5, 1]
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            return None
        
        # Initialize sparsely
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        # Place some non-zero entries
        for c in range(r):
            # Each term c should contribute to some subset of (i,j,k) triples
            # Pick a random pattern
            ui = random.randint(0, n*m-1)
            vj = random.randint(0, m*p-1)
            wk = random.randint(0, p*n-1)
            U[ui][c] = random.choice([0.5, 1])
            V[vj][c] = random.choice([0.5, 1])
            W[wk][c] = random.choice([0.5, 1])
        
        if _check_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Try ranks from 17 down to 1 (18 is schoolbook)
    # Start with 17 since that's the next below schoolbook
    for r in range(17, 0, -1):
        if time.time() > DEADLINE - 10:
            break
        
        # Try multiple strategies
        result = _try_rank(r, n, m, p, max_iters=1000, seed=r*1000+1)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        result = _try_rank_structured(r, n, m, p, max_iters=200)
        if result:
            U, V, W = result
            return {"U": U, "V": W, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
