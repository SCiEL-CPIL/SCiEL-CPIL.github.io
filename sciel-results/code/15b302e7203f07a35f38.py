import numpy as np
from itertools import product
import time

# EVOLVE-BLOCK-START
DIMENSIONS = [2, 4, 5]
DEADLINE = 120.0  # seconds

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    D = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
    return np.allclose(D, T, atol=1e-9)

def _schoolbook_decomposition():
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
    return U, V, W, rank

def _try_reduce_rank(U, V, W, target_rank, time_limit=10.0):
    """Try to find a decomposition with target_rank by setting some columns to zero
    and solving for the remaining ones."""
    n, m, p = DIMENSIONS
    current_rank = len(U[0])
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try all subsets of size target_rank from current_rank
    # For each subset, check if the corresponding rank-1 tensors span T
    # This is expensive, so use a smarter approach
    
    # Alternative: use randomized linear algebra
    # Project the tensor onto a random subspace and check rank
    
    start_time = time.time()
    
    # Try removing one term at a time and see if we can compensate
    # For each term to remove, try to express it as a combination of others
    for remove_idx in range(current_rank):
        if time.time() - start_time > time_limit:
            break
        
        # Get the tensor for the removed term
        # U[:, remove_idx], V[:, remove_idx], W[:, remove_idx]
        
        # Try to see if this rank-1 tensor is in the span of the others
        # by checking if the matrix formed by projecting is rank-deficient
        
        # Build a matrix where each column is a rank-1 tensor (flattened)
        # But this is too large. Instead, use a different approach.
        
        # Check if we can find coefficients c_1, ..., c_{r-1} such that
        # sum_{j != remove_idx} c_j * (U[:,j] ⊗ V[:,j] ⊗ W[:,j]) = U[:,remove_idx] ⊗ V[:,remove_idx] ⊗ W[:,remove_idx]
        
        # This is a system of (n*m)*(m*p)*(p*n) equations in (r-1) unknowns
        # For r=40, that's 1600 equations in 39 unknowns - likely overdetermined
        
        pass
    
    return None

def _randomized_search(target_rank, n_trials=100, time_limit=30.0):
    """Try to find a rank-target_rank decomposition using random initialization."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    start_time = time.time()
    
    for trial in range(n_trials):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Iterate: for each factor, hold others fixed and solve linear system
        for iteration in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Update U: for each a, solve sum_r V[b,r]*W[c,r]*U[a,r] = T[a,b,c]
            # This is a linear system in U[a,:] for each a
            # Actually, for fixed V and W, T[a,:,:] = U[a,:] @ (V ⊗ W)^T (reshaped)
            
            # Better: use alternating least squares
            # For U: T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
            # So for each a, T[a,:,:].reshape(-1) = U[a,:] @ M_a where M_a has columns V[:,r]⊗W[:,r]
            
            # Build M: shape (m*p * p*n, target_rank)
            M = np.einsum('br,cr->bc r', V, W).reshape(m*p*p*n, target_rank)
            
            # For each a, solve: T[a,:,:].reshape(-1) = U[a,:] @ M^T
            # i.e., M @ U[a,:].T = T[a,:,:].reshape(-1)
            for a in range(n*m):
                rhs = T[a, :, :].reshape(-1)
                # Solve M @ x = rhs where x = U[a, :].T
                # This is an overdetermined system if m*p*p*n > target_rank
                try:
                    U[a, :], residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                except:
                    pass
            
            # Update V similarly
            M_v = np.einsum('ar,cr->ac r', U, W).reshape(n*m*p*n, target_rank)
            for b in range(m*p):
                rhs = T[:, b, :].reshape(-1)
                try:
                    V[b, :], residuals, rank_m, sv = np.linalg.lstsq(M_v, rhs, rcond=None)
                except:
                    pass
            
            # Update W similarly
            M_w = np.einsum('ar,br->ab r', U, V).reshape(n*m*m*p, target_rank)
            for c in range(p*n):
                rhs = T[:, :, c].reshape(-1)
                try:
                    W[c, :], residuals, rank_m, sv = np.linalg.lstsq(M_w, rhs, rcond=None)
                except:
                    pass
        
        # Check if we found a valid decomposition
        D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        if np.allclose(D, T, atol=1e-6):
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return U_list, V_list, W_list, target_rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Get schoolbook decomposition as fallback
    U_school, V_school, W_school, schoolbook_rank = _schoolbook_decomposition()
    
    # Try to find lower rank decompositions
    # Start from schoolbook_rank - 1 and go down
    best_result = (U_school, V_school, W_school, schoolbook_rank)
    
    # Try a few lower ranks with randomized search
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 5, 0), -1):
        if time.time() - start_time > DEADLINE - 15:  # Leave 15s safety margin
            break
        
        result = _randomized_search(target_rank, n_trials=50, time_limit=10.0)
        if result is not None:
            U, V, W, rank = result
            # Verify
            if _verify(U, V, W):
                best_result = (U, V, W, rank)
                break  # Found a valid lower rank, stop
    
    U, V, W, rank = best_result
    
    # Final verification
    if not _verify(U, V, W):
        # Fallback to schoolbook
        U, V, W, rank = U_school, V_school, W_school, schoolbook_rank
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
