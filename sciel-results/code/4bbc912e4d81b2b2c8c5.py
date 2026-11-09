"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize, least_squares

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor T[a,b,c] = 1 iff consistent indices."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _validate_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = _build_target_tensor(n, m, p)
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    diff = np.max(np.abs(T_decomp - T))
    return diff < tol, diff


def _try_direct_ansatz(n, m, p, deadline=None):
    """
    Try known algebraic decompositions for 2x3x3 matrix multiplication.
    
    The key insight: we can use a Strassen-like approach where we compute
    intermediate products that combine multiple terms, then extract the
    final result.
    
    For 2x3x3, we can potentially achieve rank < 18 by grouping terms.
    """
    if deadline is None:
        deadline = time.time() + 115
    
    # Known decomposition strategies for 2x3x3:
    # Strategy 1: Use the fact that 2x3 = (1+1)x3, apply Strassen-like ideas
    # Strategy 2: Group columns/rows to reduce multiplications
    
    # Let's try a systematic search over small integer coefficient patterns
    # The tensor has 18 non-zero entries. We want to find rank-r decomposition.
    
    # For 2x3x3, the theoretical lower bound is 15 (from alpha(2,3,3) >= 15)
    # Known upper bound is 18 (schoolbook)
    
    # Try to find a rank-15 or rank-16 decomposition using structured patterns
    
    # Approach: Use the known fact that matrix multiplication rank satisfies
    # submultiplicativity and additivity properties.
    
    # For 2x3x3, one known approach:
    # Split 2x3 into two 1x3 matrices, use 1x3x3 which has rank 3 each,
    # but that gives 6, which is too low - need to handle the combination.
    
    # Actually, let's use a more direct approach: try all possible rank-r 
    # decompositions with small integer coefficients.
    
    T_target = _build_target_tensor(n, m, p)
    
    # Try ranks from 15 to 17 (skip 18 since schoolbook is already 18)
    for r in range(15, 18):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Use a structured initialization based on the tensor structure
        # Initialize U, V, W with patterns that respect the tensor's symmetry
        
        np.random.seed(42)
        
        # Try multiple structured initializations
        for attempt in range(100):
            if deadline is not None and time.time() > deadline - 3:
                break
            
            # Create structured initial matrices
            U_init = np.random.randn(n*m, r) * 0.5
            V_init = np.random.randn(m*p, r) * 0.5
            W_init = np.random.randn(p*n, r) * 0.5
            
            # Add some structure: initialize with values that might capture
            # the tensor's inherent patterns
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        # Add small structured components
                        U_init[a, :] += 0.1
                        V_init[b, :] += 0.1
                        W_init[c, :] += 0.1
            
            def residual(x):
                U_flat = x[:r*n*m].reshape(n*m, r)
                V_flat = x[r*n*m : r*n*m + r*m*p].reshape(m*p, r)
                W_flat = x[r*n*m + r*m*p :].reshape(p*n, r)
                
                T_decomp = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
                return (T_decomp - T_target).ravel()
            
            x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
            
            try:
                result = least_squares(residual, x0, method='trf', max_nfev=2000)
                
                if result.cost < 1e-10:
                    x = result.x
                    U = x[:r*n*m].reshape(n*m, r)
                    V = x[r*n*m : r*n*m + r*m*p].reshape(m*p, r)
                    W = x[r*n*m + r*m*p :].reshape(p*n, r)
                    
                    valid, diff = _validate_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                    if valid:
                        return U.tolist(), V.tolist(), W.tolist()
            except Exception:
                continue
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # First, return schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using direct ansatz
    result = _try_direct_ansatz(n, m, p, deadline=deadline)
    if result is not None:
        U, V, W = result
        best_result = {"U": U, "V": V, "W": W}
        best_rank = len(U[0])
    
    return best_result

# EVOLVE-BLOCK-END
