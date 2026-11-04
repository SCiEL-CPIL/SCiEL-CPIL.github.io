"""Rewritten program: search for low-rank exact tensor decomposition of 4x4 * 4x9 matrix multiplication."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if output (i,k) uses A[i,j]*B[j,k].
    The tensor has indices (i, j, k) with i in [0,n), j in [0,m), k in [0,p).
    In the decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k)
    where the "target" is the identity mapping: for each (i,j,k), the coefficient is 1.
    """
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1.
    """
    rank = U.shape[1]
    # Reshape for efficient computation
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute sum over c for each (i, j, k)
    # U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum
    result = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    target = np.ones((n, m, p), dtype=np.float64)
    max_err = np.max(np.abs(result - target))
    return max_err < tol, max_err

def _try_rank(r, n, m, p, rng, time_limit=140.0, max_attempts_per_rank=5000):
    """Try to find a rank-r decomposition via random search.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    
    # Strategy: random initialization with structure
    # For matrix multiplication, we can use the fact that the tensor is "separable" in a specific way.
    
    # Approach: Random Gaussian initialization, then try to solve via iterative refinement
    # But for exact integer/half-integer solutions, we need a different approach.
    
    # Better approach: Use the structure of matrix multiplication.
    # The tensor T_{ijk} = 1 for all i,j,k.
    # We need U_{ij,c} * V_{jk,c} * W_{ki,c} summed over c = 1.
    
    # Let's try: for small ranks, use structured random matrices.
    # Initialize with small random values and use a simple optimization.
    
    for attempt in range(max_attempts_per_rank):
        if time.time() - start_time > (time_limit - 10.0):
            break
        
        # Random initialization with entries in {-1, 0, 1} or {-0.5, 0, 0.5, 1}
        # For exactness, use integers or half-integers
        
        # Try different initialization strategies
        if attempt % 3 == 0:
            # Binary initialization
            U = rng.integers(0, 2, size=(n*m, r)).astype(np.float64)
            V = rng.integers(0, 2, size=(m*p, r)).astype(np.float64)
            W = rng.integers(0, 2, size=(p*n, r)).astype(np.float64)
        elif attempt % 3 == 1:
            # Ternary initialization
            U = rng.integers(-1, 2, size=(n*m, r)).astype(np.float64)
            V = rng.integers(-1, 2, size=(m*p, r)).astype(np.float64)
            W = rng.integers(-1, 2, size=(p*n, r)).astype(np.float64)
        else:
            # Small integer initialization
            U = rng.integers(-2, 3, size=(n*m, r)).astype(np.float64)
            V = rng.integers(-2, 3, size=(m*p, r)).astype(np.float64)
            W = rng.integers(-2, 3, size=(p*n, r)).astype(np.float64)
        
        # Check if this random assignment works
        ok, err = _check_decomposition(U, V, W, n, m, p)
        if ok:
            return U, V, W
    
    return None

def _winograd_style_search(n, m, p, time_limit=140.0):
    """Try Winograd-style structured decompositions.
    For matrix multiplication, Winograd uses FFT-like transforms.
    For small sizes, we can try specific structured forms.
    """
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # For 4x4x9, the schoolbook rank is 144.
    # Winograd's algorithm for 4x4 matrix multiply uses rank 16 (for square case).
    # For 4x4 * 4x9, we can try to extend Winograd ideas.
    
    # Winograd for 4x4: uses DFT-4 like structure.
    # The key insight: if we can factor the computation into smaller convolutions.
    
    # Let's try a structured approach based on tensor train / matrix product state ideas.
    # Or try specific known low-rank decompositions.
    
    # Actually, for the specific problem of 4x4 * 4x9, let's think about what ranks are achievable.
    # The matrix multiplication tensor T_{n,m,p} has rank at most n*m*p.
    # For n=4, m=4, p=9, that's 144.
    
    # Known results: rank of T_{n,m,p} is bounded by various formulas.
    # For small cases, we can try to find explicit decompositions.
    
    # Let's try a systematic search for small ranks using the structure.
    # Key idea: use the fact that the tensor is "uniform" (all entries are 1).
    
    # Strategy: Try ranks from a reasonable lower bound up to schoolbook.
    # Lower bound: max(n*m, m*p, p*n) = max(16, 36, 36) = 36.
    # So rank is at least 36.
    
    # Let's try to find a decomposition with rank around 36-80.
    
    # Use a more structured search:
    # 1. Fix a pattern for U, V, W based on block structure.
    # 2. Solve for the entries.
    
    # For now, let's try random search with more attempts for lower ranks.
    
    for r in [36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 108, 120, 132, 144]:
        if time.time() - start_time > time_limit - 5:
            break
        
        result = _try_rank(r, n, m, p, rng, time_limit=time_limit, max_attempts_per_rank=2000)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # First, try the structured/Winograd search
    result = _winograd_style_search(n, m, p, time_limit=time_limit)
    
    if result is not None:
        # Verify
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        ok, err = _check_decomposition(U, V, W, n, m, p)
        if ok:
            return result
    
    # Fallback to schoolbook decomposition
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
# EVOLVE-BLOCK-END
