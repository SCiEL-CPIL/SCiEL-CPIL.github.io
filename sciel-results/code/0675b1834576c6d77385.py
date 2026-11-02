import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute the maximum absolute residual of the tensor decomposition."""
    rank = U.shape[1]
    
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    T_true = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    residual = M - T_true
    return np.max(np.abs(residual))

def _try_rank31_search(U0, V0, W0, n, m, p, time_budget):
    """Try to find a rank-31 decomposition by optimizing from rank-32 starting point."""
    start_time = time.time()
    
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_score = _compute_residual(U0, V0, W0, n, m, p)
    
    nm, mp, pn = n * m, m * p, p * n
    rank = 32
    
    # Precompute T_true
    T_true = np.zeros((nm, mp, pn))
    true_indices = []
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
                true_indices.append((i, j, k))
    
    # Precompute a set of random zero indices for faster residual check
    # We need to ensure that the "zero" constraints are well-sampled
    rng = np.random.default_rng(42)
    zero_indices = []
    all_indices = set([(i, j, k) for i in range(nm) for j in range(mp) for k in range(pn)])
    true_set = set(true_indices)
    candidate_zeros = list(all_indices - true_set)
    # Sample 500 random zero indices
    # This is a static set for the whole search to ensure consistency
    zero_indices = rng.choice(len(candidate_zeros), size=500, replace=False)
    zero_indices = [candidate_zeros[i] for i in zero_indices]

    def compute_sq_residual(U, V, W):
        """Compute sum of squares of residuals efficiently."""
        total_sq = 0.0
        
        # Check true indices (must be 1)
        for (i, j, k) in true_indices:
            val = np.dot(U[i] * V[j], W[k])
            total_sq += (val - 1.0) ** 2
        
        # Check zero indices (must be 0)
        for (i, j, k) in zero_indices:
            val = np.dot(U[i] * V[j], W[k])
            total_sq += val ** 2
            
        return total_sq
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    # We will try to drive each column to zero
    for target_col in range(rank):
        if time.time() - start_time > time_budget * 0.8:
            break
        
        # Objective function
        def objective(params, lam=1.0):
            U_flat = params[:nm * rank]
            V_flat = params[nm * rank : nm * rank + mp * rank]
            W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            sq_res = compute_sq_residual(U, V, W)
            
            # Penalty for column target_col
            col_penalty = np.sum(U[:, target_col]**2) + np.sum(V[:, target_col]**2) + np.sum(W[:, target_col]**2)
            
            return sq_res + lam * col_penalty
        
        # Try increasing lambda to force the column to zero
        # We start with small lambda and increase it
        # If we find a solution with col_norm < 0.05, we accept it
        
        # Try multiple initial perturbations
        for trial in range(5):
            if time.time() - start_time > time_budget * 0.95:
                break
            
            # Perturb the starting point
            if trial == 0:
                x_start = x0.copy()
            else:
                # Add noise to escape local minima
                # Scale noise by trial number
                noise = np.random.randn(len(x0)) * (0.05 * trial)
                x_start = x0 + noise
            
            # Use L-BFGS-B with tighter tolerances
            try:
                result = minimize(objective, x_start, args=(1.0,), method='L-BFGS-B', 
                                  options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
            except Exception:
                continue
            
            U_flat = result.x[:nm * rank]
            V_flat = result.x[nm * rank : nm * rank + mp * rank]
            W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            # Check if column is close to zero
            col_norm_sq = np.sum(U[:, target_col]**2) + np.sum(V[:, target_col]**2) + np.sum(W[:, target_col]**2)
            col_norm = np.sqrt(col_norm_sq)
            
            if col_norm < 0.01:
                # Try rank-31 by dropping this column
                U31 = np.delete(U, target_col, axis=1)
                V31 = np.delete(V, target_col, axis=1)
                W31 = np.delete(W, target_col, axis=1)
                score31 = _compute_residual(U31, V31, W31, n, m, p)
                if score31 < 1e-6:
                    return U31, V31, W31, score31
                elif score31 < best_score:
                    best_score = score31
                    best_U, best_V, best_W = U31.copy(), V31.copy(), W31.copy()
            
            # Update best with full rank solution
            cur_score = _compute_residual(U, V, W, n, m, p)
            if cur_score < best_score:
                best_score = cur_score
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # First, build the rank-32 solution
    U, V, W = _build_rank32()
    
    # Convert to lists for return
    def to_list(arr):
        return arr.tolist()
    
    # Try to find rank-31 using local search with time budget
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        search_budget = min(remaining_time - 10, 100)  # Leave 10s safety margin
        result = _try_rank31_search(U, V, W, n, m, p, search_budget)
        if result is not None:
            U31, V31, W31, score = result
            if score < 1e-6:
                return {"U": to_list(U31), "V": to_list(V31), "W": to_list(W31)}
    
    # Fall back to rank-32
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
