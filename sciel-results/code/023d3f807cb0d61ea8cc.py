"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of matrix multiplication."""
    n, m, p = n, m, p
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = 1 if (i,j,k) is a valid index, else 0
    # For matrix multiplication: (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor has entries T_{i,j,k} = 1 for all valid (i,j,k)
    # and we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_half_integer_list(U):
    """Convert numpy array to list of lists with half-integer entries if possible."""
    U = np.array(U, dtype=float)
    # Check if entries are close to half-integers
    doubled = U * 2
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) < 1e-6:
        # All entries are half-integers
        result = [[int(round(v)) for v in row] for row in rounded]
        # Convert to half-integers: value = entry / 2
        # But the interface expects integer/half-integer entries directly
        # So we return the doubled values as integers representing 2*value
        # Actually, let's just return the values as floats that are multiples of 0.5
        result = [[float(v) / 2.0 for v in row] for row in rounded]
        return result
    else:
        return [[float(v) for v in row] for row in U]

def _try_rank(r, n, m, p, time_budget=100.0, seed=42):
    """
    Try to find a rank-r decomposition of the n x m by m x p matrix multiplication tensor.
    Uses randomized linear algebra / Groebner-like approach.
    """
    np.random.seed(seed)
    start_time = time.time()
    
    # The tensor T has entries T[i,j,k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # 
    # This is a system of n*m*p equations in 3*r*(n*m + m*p + p*n) unknowns (but with structure).
    # 
    # Strategy: Fix U and V randomly (or with some structure), then solve for W linearly.
    # Actually, let's use a different approach:
    # 
    # For a given rank r, we have r "gates" c = 0..r-1.
    # For each gate c, U[:,c] is a vector in R^(n*m), V[:,c] in R^(m*p), W[:,c] in R^(p*n).
    # The constraint is: for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # Let's fix U and V, then the constraint becomes linear in W:
    # For each (i,j,k): sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
    # 
    # This is a linear system in the entries of W. The number of unknowns is r * p * n.
    # The number of equations is n * m * p.
    # For a solution to exist, we need r * p * n >= n * m * p, i.e., r >= m.
    # But that's a necessary condition, not sufficient.
    #
    # Actually, let's think more carefully. Let's fix U and V.
    # Define M_{(i,j,k), c} = U[i*m+j, c] * V[j*p+k, c]
    # Then we need: sum_c M_{(i,j,k), c} * W[k*n+i, c] = 1
    # 
    # This is a linear system: A * w = 1, where w is the vectorized W.
    # The matrix A has dimensions (n*m*p) x (r*p*n).
    # 
    # For this to have a solution, we need rank(A) = rank([A | 1]).
    # 
    # Let's try random U and V and see if the linear system has a solution.
    
    num_tries = 200
    for attempt in range(num_tries):
        if time.time() - start_time > time_budget:
            break
        
        # Generate random U and V with entries from {-1, 0, 1} for simplicity
        # Actually, let's use small integers or half-integers
        U = np.random.randint(-1, 2, size=(n*m, r), dtype=float)
        V = np.random.randint(-1, 2, size=(m*p, r), dtype=float)
        
        # Build the linear system for W
        # Number of equations: n*m*p
        # Number of unknowns: r*p*n
        num_eq = n * m * p
        num_unk = r * p * n
        
        A = np.zeros((num_eq, num_unk))
        b = np.ones(num_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        A[eq_idx, c * p * n + k * n + i] = U[i*m+j, c] * V[j*p+k, c]
                    eq_idx += 1
        
        # Solve A * w = b
        try:
            # Use least squares as a fallback
            w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if the solution is exact
            residual = np.max(np.abs(A @ w - b))
            if residual < 1e-8:
                # We found a solution!
                W = w.reshape(r, p, n).T  # W[k*n+i, c] -> reshape to (p*n, r)
                W = W.reshape(p*n, r)
                
                # Verify
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return U.tolist(), V.tolist(), W.tolist()
        except:
            continue
    
    return None

def _try_structured_rank(r, n, m, p, time_budget=100.0):
    """
    Try structured decompositions for specific ranks.
    For 2x3x4, the known lower bound is 19.
    Let's try ranks 19, 20, 21, ...
    """
    start_time = time.time()
    
    # Try various structured patterns
    # One idea: use the fact that 2x3x4 can be decomposed using fast matrix multiplication techniques
    # The rank of 2x2x2 is 7 (Strassen), and we can build up from there.
    
    # For now, let's just try the random approach with more attempts
    result = _try_rank(r, n, m, p, time_budget=time_budget)
    if result is not None:
        return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s
    
    # Known lower bound for 2x3x4 is 19
    # Try ranks from 19 upward
    for r in range(19, 25):  # Try ranks 19 to 24
        if time.time() - start_time > deadline:
            break
        
        result = _try_structured_rank(r, n, m, p, time_budget=min(20.0, deadline - (time.time() - start_time)))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition of rank n*m*p = 24
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
