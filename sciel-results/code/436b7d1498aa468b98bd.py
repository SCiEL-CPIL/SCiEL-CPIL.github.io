"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

_START_TIME = time.time()
# 150s limit. We want to be safe.
_DEADLINE = 145

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if U, V, W form a valid rank-r decomposition for n x m * m x p.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j, :]
                v_val = V[j*p + k, :]
                w_val = W[k*n + i, :]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    return U, V, W

def _try_rank_23_structured(n, m, p):
    """
    Attempt to find a rank 23 decomposition using structured ALS with multiple restarts.
    """
    if time.time() > _DEADLINE - 5:
        return None

    R = 23
    
    # Pre-compute indices for speed
    indices_U = [(i, j) for i in range(n) for j in range(m)]
    indices_V = [(j, k) for j in range(m) for k in range(p)]
    indices_W = [(k, i) for k in range(p) for i in range(n)]
    
    # Try multiple random initializations
    for seed in [42, 123, 456, 789, 1024]:
        if time.time() > _DEADLINE - 5:
            return None
            
        rng = np.random.RandomState(seed)
        
        # Random initialization
        U = rng.randn(n * m, R) * 0.1
        V = rng.randn(m * p, R) * 0.1
        W = rng.randn(p * n, R) * 0.1
        
        max_iters = 200
        
        for it in range(max_iters):
            if time.time() > _DEADLINE - 5:
                return None
                
            # Update U
            for i, j in indices_U:
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=1e-12)
                except np.linalg.LinAlgError:
                    break
                    
            # Update V
            for j, k in indices_V:
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=1e-12)
                except np.linalg.LinAlgError:
                    break
                    
            # Update W
            for k, i in indices_W:
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=1e-12)
                except np.linalg.LinAlgError:
                    break
                    
            # Check convergence
            if it % 5 == 0:
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                    return U, V, W

    return None

def _try_rank_23_scipy(n, m, p):
    """
    Attempt to find a rank 23 decomposition using scipy.optimize with variable elimination.
    Fix W to be identity-like and solve for U, V.
    """
    if time.time() > _DEADLINE - 10:
        return None
    
    R = 23
    
    # We have 24 constraints (i,j,k) and 3*R variables.
    # Let's fix one variable in each constraint to reduce the system.
    # Actually, let's just use a nonlinear solver on the full system but with regularization.
    
    # Define variables: U (6*R), V (12*R), W (8*R)
    # Total: 26*R = 598 variables
    # Constraints: 24 equations
    
    # This is underdetermined. We need to minimize ||x||^2 subject to f(x) = 0.
    # Use scipy.optimize.minimize with method='SLSQP' or 'trust-constr'
    
    # Let's try a simpler approach: Fix W to be random, solve for U, V.
    # If W is fixed, the equations are linear in U and V? No, U and V are multiplied.
    # So it's still bilinear.
    
    # Let's stick to ALS but with a better initialization strategy: 
    # Start from the rank-24 solution and project.
    
    U24, V24, W24 = _construct_schoolbook(n, m, p)
    
    # Project to rank 23 using SVD? 
    # The tensor is not a matrix.
    
    # Alternative: Use the fact that the tensor is "almost" rank 23.
    # Let's try ALS with a very specific initialization: 
    # Take the first 23 columns of the schoolbook solution, but scale them.
    
    U = U24[:, :R].copy()
    V = V24[:, :R].copy()
    W = W24[:, :R].copy()
    
    # Add a small component from the 24th column distributed among the first 23
    # to see if it can be absorbed.
    # T_24 = sum_{r=0}^{22} u_r v_r w_r + u_23 v_23 w_23
    # We want sum_{r=0}^{22} u'_r v'_r w'_r = T_24
    
    # This is generally impossible unless the tensor is rank 23.
    # If it is rank 23, ALS should find it.
    
    # Let's run ALS from this initialization.
    max_iters = 500
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            return None
            
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=1e-12)
                except np.linalg.LinAlgError:
                    return None
                    
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=1e-12)
                except np.linalg.LinAlgError:
                    return None
                    
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=1e-12)
                except np.linalg.LinAlgError:
                    return None
                    
        if it % 10 == 0:
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                return U, V, W

    return None

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Immediately construct the known valid rank 24 solution
    U24, V24, W24 = _construct_schoolbook(n, m, p)
    
    # 2. Attempt to find a lower rank (23) if time permits
    # Try structured ALS first
    result_23 = _try_rank_23_structured(n, m, p)
    
    if result_23 is not None:
        U, V, W = result_23
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Try scipy-based or other methods if time permits
    if time.time() < _DEADLINE - 15:
        result_23b = _try_rank_23_scipy(n, m, p)
        if result_23b is not None:
            U, V, W = result_23b
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }

    # 3. Fallback to rank 24
    return {
        "U": U24.tolist(),
        "V": V24.tolist(),
        "W": W24.tolist()
    }
# EVOLVE-BLOCK-END
