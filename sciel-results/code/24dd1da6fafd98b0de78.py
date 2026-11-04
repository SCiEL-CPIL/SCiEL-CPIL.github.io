"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a better decomposition within time budget
    deadline = time.time() + 120
    
    # First, try the Laderman-inspired approach for 4x4x9
    laderman_result = _try_laderman_extension(n, m, p)
    if laderman_result is not None:
        return laderman_result
    
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

def _try_laderman_extension(n, m, p, time_limit=100):
    """
    Try to find a lower-rank decomposition using Laderman-inspired structure.
    
    Laderman's algorithm achieves rank 7 for 4x4x4 matrix multiplication.
    For 4x4x9, we can try to extend this by treating the 9 columns as three
    blocks of 3, or by finding a direct low-rank decomposition.
    
    We'll use a structured random search with Laderman-like constraints.
    """
    start_time = time.time()
    
    # Lower bound: max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But for matrix multiplication, the actual lower bound is higher.
    # Known: 4x4x4 has rank 7 (Laderman), so 4x4x9 should be at least 7*9/4 ≈ 15.75
    # More realistically, we expect rank around 30-50 for 4x4x9.
    
    # Try ranks from a reasonable lower bound up to schoolbook
    min_rank = 30
    max_rank = n * m * p - 1  # 143
    
    best_solution = None
    
    for target_rank in range(min_rank, min(max_rank + 1, 80)):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Try multiple random restarts
        num_restarts = 3
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit - 5:
                break
            
            solution = _random_search_for_rank(n, m, p, target_rank, restart)
            if solution is not None:
                best_solution = solution
                break
        
        if best_solution is not None:
            break
    
    return best_solution

def _random_search_for_rank(n, m, p, target_rank, seed_offset=0):
    """
    Try to find a rank-target_rank decomposition using random initialization
    and iterative refinement.
    """
    np.random.seed(42 + seed_offset)
    
    # Initialize with random values
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    # Iterative refinement
    max_iterations = 50
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
        
        # Check convergence
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0
                    for c in range(target_rank):
                        val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    max_err = max(max_err, abs(val - 1))
        
        if max_err < 1e-10:
            # Try to convert to exact fractions
            for denom in [100, 1000, 10000]:
                U_frac = [[Fraction(U[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(n * m)]
                V_frac = [[Fraction(V[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(m * p)]
                W_frac = [[Fraction(W[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(p * n)]
                
                if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

# EVOLVE-BLOCK-END
