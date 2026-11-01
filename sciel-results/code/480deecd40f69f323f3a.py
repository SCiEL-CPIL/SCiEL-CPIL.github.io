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
    # The tensor T[i, j, k] should equal sum_r U[i, r] * V[j, r] * W[k, r]
    # where i in [0, n*m), j in [0, m*p), k in [0, p*n)
    # The matrix multiplication tensor has T[i, j, k] = 1 if:
    #   i = a*m + b, j = b*p + c, k = c*n + a for some a, b, c
    # Otherwise T[i, j, k] = 0
    
    # Compute the decomposed tensor
    # T_decomposed[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # This is equivalent to: for each r, outer(U[:, r], outer(V[:, r], W[:, r]))
    
    # Use einsum for efficiency
    # U: (nm, rank), V: (mp, rank), W: (pn, rank)
    # T_decomposed[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    
    # Compute using matrix operations
    # T_decomposed = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # But this might be large. Let's compute it differently.
    # For each pair (i, j), T_decomposed[i, j, :] = (U[i, :] * V[j, :]) @ W.T
    # So we can compute: UV = U @ V.T  (nm x mp)
    # Then T_decomposed[i, j, k] = sum_r (U[i, r] * V[j, r]) * W[k, r]
    # = (U @ V.T)[i, j] ... no, that's not right.
    
    # Actually: T_decomposed[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # Let's compute M[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # = sum_r (U[i, r] * V[j, r]) * W[k, r]
    # For fixed i, j: let x_r = U[i, r] * V[j, r], then M[i, j, :] = x @ W.T
    # So M[i, j, k] = sum_r x_r * W[k, r] = (x @ W.T)[k]
    
    # Compute all at once:
    # M[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # = sum_r (U @ V.T)[i, j, r] ... no.
    
    # Better: use einsum
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Now build the true tensor
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
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0

    # Precompute T_true flattened for faster access
    T_true_flat = T_true.flatten()
    
    # Build the true tensor indices for faster residual computation
    # The tensor is sparse: only n*m*p = 36 entries are 1
    true_indices = []
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                true_indices.append((i, j, k))
    
    # More efficient residual computation using only the non-zero entries of T_true
    def compute_sq_residual(U, V, W):
        """Compute sum of squares of residuals efficiently."""
        # For the 36 non-zero entries, compute the decomposed value
        # For the remaining entries, the decomposed value should be 0
        total_sq = 0.0
        
        # Compute contributions from all rank terms
        # M[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For efficiency, compute M at the true indices and at some random other indices
        
        # First, compute at true indices
        for (i, j, k) in true_indices:
            val = np.dot(U[i] * V[j], W[k])
            total_sq += (val - 1.0) ** 2
        
        # Also check that other entries are close to 0
        # Sample some random entries to ensure they're small
        # Use deterministic pseudo-random indices based on rank
        np.random.seed(42)
        for _ in range(20):
            i = np.random.randint(nm)
            j = np.random.randint(mp)
            k = np.random.randint(pn)
            # Skip if this is a true index
            if (i, j, k) in true_indices:
                continue
            val = np.dot(U[i] * V[j], W[k])
            total_sq += val ** 2
        
        return total_sq
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    for target_col in range(rank):
        if time.time() - start_time > time_budget * 0.75:
            break
        
        # Objective with adaptive penalty
        def objective(params, lam=1.0):
            U_flat = params[:nm * rank]
            V_flat = params[nm * rank : nm * rank + mp * rank]
            W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            sq_res = compute_sq_residual(U, V, W)
            
            # Penalty for column target_col - use L2 norm squared
            col_penalty = np.sum(U[:, target_col]**2) + np.sum(V[:, target_col]**2) + np.sum(W[:, target_col]**2)
            
            return sq_res + lam * col_penalty
        
        # Try multiple lambda values with increasing penalty
        for lam in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            if time.time() - start_time > time_budget * 0.9:
                break
            
            # Try multiple initial perturbations for this lambda
            for trial in range(3):
                if time.time() - start_time > time_budget * 0.92:
                    break
                
                # Create perturbed starting point
                if trial == 0:
                    x_start = x0.copy()
                else:
                    # Small random perturbation to escape local minima
                    noise = np.random.randn(len(x0)) * 0.01 * trial
                    x_start = x0 + noise
                
                try:
                    result = minimize(objective, x_start, args=(lam,), method='L-BFGS-B', 
                                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
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
                
                if col_norm < 0.05:
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
    if remaining_time > 25:
        search_budget = min(remaining_time - 15, 90)  # Leave 15s safety margin
        result = _try_rank31_search(U, V, W, n, m, p, search_budget)
        if result is not None:
            U31, V31, W31, score = result
            if score < 1e-6:
                return {"U": to_list(U31), "V": to_list(V31), "W": to_list(W31)}
    
    # Fall back to rank-32
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
