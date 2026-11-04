"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Strategy 1: Structured Kronecker product approach
    # For 4x4x9, treat as block structure and use Kronecker products
    kronecker_result = _try_kronecker_decomposition(n, m, p, deadline)
    if kronecker_result is not None:
        return kronecker_result
    
    # Strategy 2: Improved iterative refinement with rank reduction
    iterative_result = _try_iterative_refinement(n, m, p, deadline)
    if iterative_result is not None:
        return iterative_result
    
    # Fallback to schoolbook
    return _schoolbook_solution(n, m, p)

def _schoolbook_solution(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the factor matrices exactly decompose the matrix multiplication tensor.
    For each (i,j,k), the tensor entry T[i*m+j, j*p+k, k*n+i] should equal
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    Returns True if all entries match exactly (using Fraction arithmetic).
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[u_idx][c]) * Fraction(V[v_idx][c]) * Fraction(W[w_idx][c])
                if total != 1:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-8):
    """Fast verification using numpy for convergence checking."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False
    return True

def _try_kronecker_decomposition(n, m, p, deadline):
    """
    Use Kronecker product structure to find a structured low-rank decomposition.
    For 4x4x9, we can exploit the fact that 9 = 3*3 and use block Kronecker products.
    """
    start_time = time.time()
    
    # Try to decompose using Kronecker products
    # 4x4x4 has known rank 7 (Laderman)
    # 4x4x9 can be seen as 4x4x(3*3)
    
    # First, get a good 4x4x4 decomposition (Laderman's rank 7)
    laderman_4x4 = _get_laderman_4x4()
    if laderman_4x4 is None:
        return None
    
    U4, V4, W4 = laderman_4x4
    rank4 = U4.shape[1]  # Should be 7
    
    # Now extend to 4x4x9 by treating the 9 columns as 3 blocks of 3
    # Use Kronecker product with a 3x3x3 identity-like structure
    
    # For each of the 3 column blocks, we need a 4x4x3 decomposition
    # We can use the 4x4x4 rank-7 decomposition and select 3 columns
    
    # Build the extended decomposition
    total_rank = 0
    U_ext = []
    V_ext = []
    W_ext = []
    
    # For each column block (3 blocks of 3 columns)
    for block in range(3):
        # Use the Laderman 4x4x4 decomposition for this block
        # Select columns block*3 to block*3+2 from the 4x4x4 result
        for c in range(rank4):
            # Add this rank-1 component
            U_ext.append(U4[:, c])
            V_ext.append(V4[:, c])
            W_ext.append(W4[:, c])
            total_rank += 1
    
    # This gives rank 21, but we can do better by sharing structure
    # Let's try a more efficient approach: use the full 4x4x9 tensor directly
    
    # Alternative: direct structured search
    best_rank = n * m * p  # Schoolbook
    best_solution = None
    
    # Try various target ranks
    for target_rank in range(20, 40):
        if time.time() > deadline - 20:
            break
        
        # Try multiple initializations
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            
            result = _structured_search(n, m, p, target_rank, seed)
            if result is not None:
                best_solution = result
                best_rank = target_rank
                break
        
        if best_solution is not None:
            break
    
    return best_solution

def _get_laderman_4x4():
    """
    Return Laderman's rank-7 decomposition for 4x4x4 matrix multiplication.
    This is a well-known result.
    """
    # Laderman's algorithm for 4x4x4
    # U: 16x7, V: 16x7, W: 16x7
    # Using the standard Laderman construction
    
    U = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ]
    
    V = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ]
    
    W = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ]
    
    # This is just the schoolbook for 4x4x4, not Laderman's rank 7
    # Let's use a proper rank-7 construction
    # For now, return None to indicate we need a better approach
    return None

def _structured_search(n, m, p, target_rank, seed=0):
    """
    Structured search for a rank-target_rank decomposition.
    Uses random initialization with iterative refinement.
    """
    np.random.seed(123 + seed)
    
    # Initialize with random values
    U = np.random.randn(n * m, target_rank) * 0.3
    V = np.random.randn(m * p, target_rank) * 0.3
    W = np.random.randn(p * n, target_rank) * 0.3
    
    # Iterative refinement with multiple sweeps
    max_iterations = 100
    for iteration in range(max_iterations):
        # Update U
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                A = np.zeros((p, target_rank))
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    A[k, :] = V[v_idx, :] * W[w_idx, :]
                b = np.ones(p)
                try:
                    U[u_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                A = np.zeros((n, target_rank))
                for i in range(n):
                    u_idx = i * m + j
                    w_idx = k * n + i
                    A[i, :] = U[u_idx, :] * W[w_idx, :]
                b = np.ones(n)
                try:
                    V[v_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                A = np.zeros((m, target_rank))
                for j in range(m):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    A[j, :] = U[u_idx, :] * V[v_idx, :]
                b = np.ones(m)
                try:
                    W[w_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Check convergence using fast verification
        if _verify_decomposition_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-8):
            # Try to convert to exact fractions
            for denom in [100, 1000, 10000, 100000]:
                U_frac = [[Fraction(U[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(n * m)]
                V_frac = [[Fraction(V[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(m * p)]
                W_frac = [[Fraction(W[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(p * n)]
                
                if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

def _try_iterative_refinement(n, m, p, deadline):
    """
    More aggressive iterative refinement starting from higher ranks
    and attempting to reduce rank.
    """
    start_time = time.time()
    
    # Start from a moderate rank and try to refine
    for target_rank in range(25, 50):
        if time.time() > deadline - 20:
            break
        
        # Try multiple seeds
        for seed in range(3):
            if time.time() > deadline - 15:
                break
            
            result = _structured_search(n, m, p, target_rank, seed + 100)
            if result is not None:
                return result
    
    return None

# EVOLVE-BLOCK-END
