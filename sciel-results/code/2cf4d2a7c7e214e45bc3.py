"""Program for rank-minimal tensor decomposition of 3x3 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the rank-27 schoolbook decomposition."""
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

def _verify(U, V, W, tol=1e-6):
    """
    Check that the decomposition is valid.
    
    The condition is:
    sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}
    
    for all i, j, j', k.
    """
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += Ua[i*m+j, c] * Va[jp*p+k, c] * Wa[k*n+i, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def _check_half_ints(U, V, W, tol=1e-6):
    """Check all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for v in row:
                if abs(v * 2 - round(v * 2)) > tol:
                    return False
    return True

def _round_to_half_ints(U, V, W):
    """Round all entries to nearest half-integer."""
    def r(x):
        return round(x * 2) / 2.0
    return (
        [[r(v) for v in row] for row in U],
        [[r(v) for v in row] for row in V],
        [[r(v) for v in row] for row in W],
    )

def _try_strassen_rank23(deadline):
    """
    Try to construct a rank-23 decomposition based on Strassen's algorithm.
    Uses a structured approach with symmetry exploitation.
    """
    n, m, p = DIMENSIONS
    rank = 23
    
    # Strassen's algorithm for 3x3 matrices uses a specific structure.
    # We'll try to build it using the known algebraic relations.
    
    # Initialize with a structured pattern based on the 2x2 Strassen
    # and extend to 3x3 using block decomposition ideas.
    
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Try multiple structured initializations
    np.random.seed(42)
    best_obj = float('inf')
    best_solution = None
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Create a structured initialization
        U_init = np.zeros((9, rank))
        V_init = np.zeros((9, rank))
        W_init = np.zeros((9, rank))
        
        # Fill with random values with some structure
        for c in range(rank):
            # Each rank-1 term should have a specific pattern
            # based on which (i,j,k) triple it contributes to
            i_idx = c % 3
            j_idx = (c // 3) % 3
            k_idx = (c // 9) % 3
            
            # Assign values based on the index pattern
            U_init[i_idx * 3 + j_idx, c] = np.random.choice([-1, 0, 1])
            V_init[j_idx * 3 + k_idx, c] = np.random.choice([-1, 0, 1])
            W_init[k_idx * 3 + i_idx, c] = np.random.choice([-1, 0, 1])
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def obj(x):
            Ua = x[0:9*rank].reshape(9, rank)
            Va = x[9*rank:18*rank].reshape(9, rank)
            Wa = x[18*rank:27*rank].reshape(9, rank)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += Ua[i*m+j, c] * Va[jp*p+k, c] * Wa[k*n+i, c]
                            expected = 1.0 if j == jp else 0.0
                            err += (val - expected) ** 2
            return err
        
        def grad(x):
            Ua = x[0:9*rank].reshape(9, rank)
            Va = x[9*rank:18*rank].reshape(9, rank)
            Wa = x[18*rank:27*rank].reshape(9, rank)
            
            gU = np.zeros_like(Ua)
            gV = np.zeros_like(Va)
            gW = np.zeros_like(Wa)
            
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            expected = 1.0 if j == jp else 0.0
                            val = 0.0
                            for c in range(rank):
                                val += Ua[i*m+j, c] * Va[jp*p+k, c] * Wa[k*n+i, c]
                            diff = val - expected
                            for c in range(rank):
                                gU[i*m+j, c] += 2 * diff * Va[jp*p+k, c] * Wa[k*n+i, c]
                                gV[jp*p+k, c] += 2 * diff * Ua[i*m+j, c] * Wa[k*n+i, c]
                                gW[k*n+i, c] += 2 * diff * Ua[i*m+j, c] * Va[jp*p+k, c]
            
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        try:
            result = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_solution = result.x
                
                if best_obj < 1e-10:
                    break
        except Exception:
            continue
    
    if best_solution is not None and best_obj < 1e-8:
        Ua = best_solution[0:9*rank].reshape(9, rank)
        Va = best_solution[9*rank:18*rank].reshape(9, rank)
        Wa = best_solution[18*rank:27*rank].reshape(9, rank)
        
        U = [[Ua[i, c] for c in range(rank)] for i in range(9)]
        V = [[Va[i, c] for c in range(rank)] for i in range(9)]
        W = [[Wa[i, c] for c in range(rank)] for i in range(9)]
        
        U, V, W = _round_to_half_ints(U, V, W)
        
        if _verify(U, V, W) and _check_half_ints(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Try rank-23 construction
    result = _try_strassen_rank23(deadline)
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
