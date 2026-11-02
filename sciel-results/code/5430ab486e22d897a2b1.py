"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _build_decomposition(rank_terms):
    """
    Build U, V, W factor matrices from a list of term specifications.
    Each term is (U_dict, V_dict, W_dict) where each dict maps (i,j) or (j,k) or (k,i) to a scalar.
    """
    n, m, p = DIMENSIONS
    R = len(rank_terms)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for the 2x2 core (indices 0,1).
    """
    terms = []
    terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}))
    terms.append(({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}))
    terms.append(({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}))
    terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}))
    terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
    terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
    return terms

def _border_terms_standard():
    """Return standard multiplication terms for the border."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    return terms

def _try_alternating_minimization(R, n, m, p, max_iter=500, tol=1e-12, seed=42):
    """
    Try to find a rank-R decomposition using alternating minimization.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with structure: first 7 terms like Strassen, rest random
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Set up Strassen-like initialization for first 7
    strassen_terms = _strassen_2x2_terms()
    for r in range(7):
        u_dict, v_dict, w_dict = strassen_terms[r]
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
    
    # Randomly initialize remaining terms
    for r in range(7, R):
        U[:, r] = rng.randn(n * m) * 0.1
        V[:, r] = rng.randn(m * p) * 0.1
        W[:, r] = rng.randn(p * n) * 0.1
    
    # Precompute index mappings
    idx_u = [(i, j) for i in range(n) for j in range(m)]
    idx_v = [(j, k) for j in range(m) for k in range(p)]
    idx_w = [(k, i) for k in range(p) for i in range(n)]
    
    for iteration in range(max_iter):
        # Solve for W: for each (k,i), we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all (i,j,k)
        # For fixed U,V, the equation for W[k,i,r] is linear in W.
        # Actually, let's reformulate: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # For fixed U, V, this is linear in W.
        # Let's build the linear system: for each (i,j,k), sum_r (U[i,j,r] * V[j,k,r]) * W[k,i,r] = 1
        # But W[k,i,r] appears in multiple equations (for different j).
        
        # Better: iterate over r. For each r, W[:,r] is a vector of length p*n.
        # The constraint is: for all (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        # This is a system of 27 equations in R*(p*n) unknowns (all entries of W).
        # Actually it's coupled across r, so let's just do least squares.
        
        # Build A matrix for W: shape (27, R * p * n)
        # A[(i,j,k), (r, k, i)] = U[i,j,r] * V[j,k,r]
        A_W = np.zeros((n*m*p, R * p * n))
        b_W = np.ones(n*m*p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(R):
                        col_idx = r * p * n + k * n + i
                        A_W[eq_idx, col_idx] = U[i * m + j, r] * V[j * p + k, r]
        
        W_flat, residuals, rank_W, sv = np.linalg.lstsq(A_W, b_W, rcond=None)
        W = W_flat.reshape(R, p * n).T
        
        # Similarly solve for U
        A_U = np.zeros((n*m*p, R * n * m))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(R):
                        col_idx = r * n * m + i * m + j
                        A_U[eq_idx, col_idx] = V[j * p + k, r] * W[k * n + i, r]
        
        U_flat, residuals, rank_U, sv = np.linalg.lstsq(A_U, b_W, rcond=None)
        U = U_flat.reshape(R, n * m).T
        
        # Similarly solve for V
        A_V = np.zeros((n*m*p, R * m * p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(R):
                        col_idx = r * m * p + j * p + k
                        A_V[eq_idx, col_idx] = U[i * m + j, r] * W[k * n + i, r]
        
        V_flat, residuals, rank_V, sv = np.linalg.lstsq(A_V, b_W, rcond=None)
        V = V_flat.reshape(R, m * p).T
        
        # Check convergence
        U_reshaped = U.reshape((n, m, R))
        V_reshaped = V.reshape((m, p, R))
        W_reshaped = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        error = np.max(np.abs(T_recon - 1.0))
        
        if error < tol:
            return U, V, W, error
    
    return U, V, W, error

def _try_standard_decomposition():
    """Standard schoolbook: rank 27."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    U, V, W = _build_decomposition(terms)
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Strassen core + standard border (rank 26)
    if time.time() < deadline:
        terms = _strassen_2x2_terms() + _border_terms_standard()
        U, V, W = _build_decomposition(terms)
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    # Strategy 2: Try alternating minimization for lower ranks
    for R in [25, 24, 23]:
        if time.time() >= deadline:
            break
        
        # Try multiple seeds
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() >= deadline:
                break
            
            U, V, W, error = _try_alternating_minimization(R, n, m, p, max_iter=300, tol=1e-10, seed=seed)
            if error < 1e-9 and _verify_decomposition(U, V, W, n, m, p):
                if R < best_rank:
                    best_rank = R
                    best_result = (U, V, W)
                    break  # Found valid decomposition for this R, move to next lower R
    
    # Strategy 3: Pure standard (rank 27) as fallback
    if time.time() < deadline:
        U, V, W = _try_standard_decomposition()
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    if best_result is None:
        U, V, W = _try_standard_decomposition()
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
