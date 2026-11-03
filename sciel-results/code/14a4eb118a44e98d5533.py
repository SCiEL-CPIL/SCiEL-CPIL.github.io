import numpy as np
import time
from scipy.optimize import minimize
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
    T_true = np.ones((n, m, p))
    return np.allclose(T_rec, T_true, atol=1e-9)

def _build_linear_system_for_W(U, V, n, m, p):
    """
    Given U (n*m x R) and V (m*p x R), build the linear system to solve for W (p*n x R).
    
    For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is a linear system in W. For each r, W[k*n+i, r] appears in equations
    where the coefficient is U[i*m+j, r] * V[j*p+k, r].
    
    We can group by r: for each r, we have variables W[:, r] (length p*n).
    The constraint for each (i,j,k) is: sum_r [U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]] = 1
    
    This couples all r's together. We need to solve a joint linear system.
    
    Actually, let's think differently. The tensor equation is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For fixed (i,j,k), this is linear in W. But W entries are shared across different (i,j,k).
    
    Let's build a big linear system. Variables: W_flat of size R * p * n.
    For each (i,j,k) in range(n) x range(m) x range(p):
        sum over r of U[i*m+j, r] * V[j*p+k, r] * W_flat[r * p * n + k*n + i] = 1
    
    This is a linear system A * W_flat = b where A has n*m*p rows and R*p*n columns.
    """
    R = U.shape[1]
    num_vars = R * p * n
    num_eqs = n * m * p
    
    A = np.zeros((num_eqs, num_vars))
    b = np.ones(num_eqs)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(R):
                    coeff = U[i*m+j, r] * V[j*p+k, r]
                    if coeff != 0:
                        A[eq_idx, r * p * n + k*n + i] = coeff
                eq_idx += 1
    
    return A, b

def _solve_for_W(U, V, n, m, p):
    """Solve for W given U and V using least squares or exact rational solve."""
    A, b = _build_linear_system_for_W(U, V, n, m, p)
    
    # Try to solve the linear system
    # If A has full row rank, we can find a solution
    try:
        # Use least squares
        W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        
        # Check if the solution is exact
        W = W_flat.reshape(R := U.shape[1], p * n).T  # W is (p*n, R)
        if _verify_decomp(U, V, W, n, m, p):
            return W
    except Exception:
        pass
    
    return None

def _generate_binary_patterns(n, m, p, R, seed=42):
    """Generate random binary/sparsity patterns for U and V."""
    np.random.seed(seed)
    
    # U: (n*m, R) with entries in {0, 1, -1}
    # V: (m*p, R) with entries in {0, 1, -1}
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    
    for r in range(R):
        # For each column r, choose a subset of entries to be non-zero
        # Use a structured pattern based on difference sets or similar
        
        # Simple approach: random sparsity
        u_sparse = np.random.choice([0, 1, -1], size=n*m, p=[0.5, 0.25, 0.25])
        v_sparse = np.random.choice([0, 1, -1], size=m*p, p=[0.5, 0.25, 0.25])
        
        U[:, r] = u_sparse
        V[:, r] = v_sparse
    
    return U, V

def _try_rank(R, n, m, p, max_restarts=20, time_limit=None, start_time=None):
    """Try to find a rank-R decomposition using structured binary patterns."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 150.0
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Generate structured patterns
        seed = 42 + restart
        U, V = _generate_binary_patterns(n, m, p, R, seed=seed)
        
        # Solve for W
        W = _solve_for_W(U, V, n, m, p)
        if W is not None:
            return U, V, W
    
    return None

def _strassen_block(n, m, p):
    """
    Use Strassen-like block decomposition to get a better initial rank.
    For 2x5 by 5x6, we can partition into blocks and apply Strassen.
    """
    # This is a placeholder for a more sophisticated approach
    # For now, return the schoolbook rank
    return n * m * p

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 150.0
    
    # First, establish the schoolbook baseline
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find lower ranks using structured linear algebra
    # The theoretical lower bound for 2x5 by 5x6 is around 54-55
    # Try ranks from 54 down to 50
    
    for R in range(54, 49, -1):
        if time.time() - start_time > time_limit - 20:
            break
        
        result = _try_rank(R, n, m, p, max_restarts=5, time_limit=time_limit, start_time=start_time)
        if result is not None:
            U, V, W = result
            best_rank = R
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            # Found a valid decomposition, stop searching for lower ranks
            break
    
    return best_result
