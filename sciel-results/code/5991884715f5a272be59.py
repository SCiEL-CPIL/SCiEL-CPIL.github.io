"""New approach: Alternating projection algorithm for tensor decomposition.
Uses random initialization and iterative refinement with half-integer projection
to find exact low-rank decompositions of the 3x4x6 matrix multiplication tensor.
"""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 4, 6]

def _build_tensor_support(n, m, p):
    """Build the list of (U_idx, V_idx, W_idx) tuples where the tensor has value 1.
    
    For matrix multiplication C = AB where A is n×m, B is m×p, C is n×p:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T satisfies: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k] = <AB, C>
    So T[i,j,k] = 1 for all valid (i,j,k).
    
    In the factorization: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    """
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                support.append((u_idx, v_idx, w_idx))
    return support

def _project_to_half_integers(arr, tol=1e-9):
    """Project a vector to nearest half-integer values."""
    # Multiply by 2, round, divide by 2
    return np.round(arr * 2.0) / 2.0

def _solve_for_factor(fixed_factors, fixed_idx, target_values, n_params):
    """Solve for one factor matrix given the other two are fixed.
    
    fixed_factors: list of 3 factor matrices
    fixed_idx: which factor to solve for (0=U, 1=V, 2=W)
    target_values: the support tuples
    n_params: shape of the factor to solve for
    """
    U, V, W = fixed_factors
    rank = U.shape[1]
    
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each (i,j,k)
    # This is a linear system in the unknown factor.
    
    if fixed_idx == 0:
        # Solve for U: U[i*m+j, :] is unknown
        # For each (i,j,k): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
        # Group by (i,j): for each (i,j), we have equations over k
        # Let A[(i,j), k, r] = V[j*p+k, r] * W[k*n+i, r]
        # Then for each (i,j): sum_k sum_r A[(i,j), k, r] * U[i*m+j, r] = p (sum of 1s over k)
        # Wait, that's not right. We need separate equations for each (i,j,k).
        
        # Actually, we have n*m*p equations and n*m*rank unknowns.
        # Let's build the full linear system.
        
        n_eq = len(target_values)
        n_unknowns = n_params[0] * rank
        
        # Build matrix M where M[eq, unknown] such that M @ x = b
        # Unknown x is U flattened: x[U_idx, r]
        
        M = np.zeros((n_eq, n_unknowns))
        b = np.ones(n_eq)
        
        for eq_idx, (u_idx, v_idx, w_idx) in enumerate(target_values):
            for r in range(rank):
                # U[u_idx, r] * V[v_idx, r] * W[w_idx, r] = 1 (contributing to this equation)
                coeff = V[v_idx, r] * W[w_idx, r]
                M[eq_idx, u_idx * rank + r] = coeff
        
        # Solve M @ x = b (least squares if overdetermined)
        try:
            x, residuals, rank_l, sv = np.linalg.lstsq(M, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        
        # Reshape to factor matrix
        U_new = x.reshape(n_params[0], rank)
        return U_new
    
    elif fixed_idx == 1:
        # Solve for V
        n_eq = len(target_values)
        n_unknowns = n_params[0] * rank
        
        M = np.zeros((n_eq, n_unknowns))
        b = np.ones(n_eq)
        
        for eq_idx, (u_idx, v_idx, w_idx) in enumerate(target_values):
            for r in range(rank):
                coeff = U[u_idx, r] * W[w_idx, r]
                M[eq_idx, v_idx * rank + r] = coeff
        
        try:
            x, residuals, rank_l, sv = np.linalg.lstsq(M, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        
        V_new = x.reshape(n_params[0], rank)
        return V_new
    
    else:
        # Solve for W
        n_eq = len(target_values)
        n_unknowns = n_params[0] * rank
        
        M = np.zeros((n_eq, n_unknowns))
        b = np.ones(n_eq)
        
        for eq_idx, (u_idx, v_idx, w_idx) in enumerate(target_values):
            for r in range(rank):
                coeff = U[u_idx, r] * V[v_idx, r]
                M[eq_idx, w_idx * rank + r] = coeff
        
        try:
            x, residuals, rank_l, sv = np.linalg.lstsq(M, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        
        W_new = x.reshape(n_params[0], rank)
        return W_new

def _verify_decomposition(U, V, W, target_values, tol=1e-8):
    """Verify that the decomposition is exact."""
    for (u_idx, v_idx, w_idx) in target_values:
        val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
        if abs(val - 1.0) > tol:
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    target_values = _build_tensor_support(n, m, p)
    
    # Try to find decomposition at various ranks, starting from just below schoolbook
    best_U = None
    best_V = None
    best_W = None
    best_rank = schoolbook_rank
    
    # First, set up schoolbook as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for (u_idx, v_idx, w_idx) in target_values:
        U_school[u_idx, col] = 1
        V_school[v_idx, col] = 1
        W_school[w_idx, col] = 1
        col += 1
    
    # Try ranks from schoolbook-1 down to some minimum
    min_rank_to_try = max(1, schoolbook_rank - 10)
    
    start_time = time.time()
    time_limit = 115  # Leave 5 seconds buffer
    
    for rank in range(schoolbook_rank - 1, min_rank_to_try - 1, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random initializations at this rank
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                    break
                    
            rng = np.random.RandomState(attempt * 12345 + rank)
            
            # Random initialization with small values
            U = rng.randn(n * m, rank) * 0.1
            V = rng.randn(m * p, rank) * 0.1
            W = rng.randn(p * n, rank) * 0.1
            
            # Project to half-integers initially
            U = _project_to_half_integers(U)
            V = _project_to_half_integers(V)
            W = _project_to_half_integers(W)
            
            # Alternating projection iterations
            max_iter = 200
            converged = False
            
            for it in range(max_iter):
                if time.time() - start_time > time_limit:
                    break
                    
                # Solve for U given V, W
                U_new = _solve_for_factor([U, V, W], 0, target_values, (n * m,))
                if U_new is None:
                    break
                U = _project_to_half_integers(U_new)
                
                # Solve for V given U, W
                V_new = _solve_for_factor([U, V, W], 1, target_values, (m * p,))
                if V_new is None:
                    break
                V = _project_to_half_integers(V_new)
                
                # Solve for W given U, V
                W_new = _solve_for_factor([U, V, W], 2, target_values, (p * n,))
                if W_new is None:
                    break
                W = _project_to_half_integers(W_new)
                
                # Check convergence
                if _verify_decomposition(U, V, W, target_values, tol=1e-10):
                    converged = True
                    break
            
            if converged:
                # Verify once more
                if _verify_decomposition(U, V, W, target_values, tol=1e-10):
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < schoolbook_rank and best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
