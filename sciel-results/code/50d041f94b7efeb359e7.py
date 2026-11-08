"""Program for 3x3x6 matrix multiplication rank minimization."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _build_tensor_constraints(n, m, p):
    """Return list of (U_idx, V_idx, W_idx) for each tensor entry that must equal 1."""
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
    return constraints

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is exact. Returns max error."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _to_half_int(x, tol=1e-6):
    """Round to nearest half-integer if close."""
    rounded = np.round(x * 2) / 2.0
    if np.max(np.abs(x - rounded)) < tol:
        return rounded
    return x

def _objective_fast(x, n, m, p, rank, precomputed_indices):
    """Compute the sum of squared errors for a given parameter vector (vectorized)."""
    U_flat = x[:n*m*rank]
    V_flat = x[n*m*rank : n*m*rank + m*p*rank]
    W_flat = x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
    
    U_mat = U_flat.reshape(n * m, rank)
    V_mat = V_flat.reshape(m * p, rank)
    W_mat = W_flat.reshape(p * n, rank)
    
    # Precomputed indices for all (i,j,k) tuples
    u_indices, v_indices, w_indices = precomputed_indices
    
    # Vectorized: for each (i,j,k), compute dot(U[u], V[v]*W[w])
    U_rows = U_mat[u_indices]  # shape (54, rank)
    V_rows = V_mat[v_indices]  # shape (54, rank)
    W_rows = W_mat[w_indices]  # shape (54, rank)
    
    # Element-wise product then dot with U
    vals = np.sum(U_rows * V_rows * W_rows, axis=1)  # shape (54,)
    err_sum = np.sum((vals - 1.0) ** 2)
    return err_sum

def _build_precomputed_indices(n, m, p):
    """Build precomputed index arrays for the objective function."""
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
    return (np.array(u_indices), np.array(v_indices), np.array(w_indices))

def _structured_init_block(n, m, p, rank, seed=0):
    """
    Create a structured initialization exploiting the 3x3x6 = 2*(3x3x3) structure.
    """
    np.random.seed(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    if rank >= 27:
        half = rank // 2
        # First half: handles columns 0-2 of B (3x3x3 sub-problem)
        # Use a base Strassen-like structure
        U[:, :half] = np.random.randn(n * m, half) * 0.1
        V[:m*3, :half] = np.random.randn(m * 3, half) * 0.1
        W[:3*n, :half] = np.random.randn(3 * n, half) * 0.1
        
        # Add structural entries for the 3x3x3 part
        for r in range(min(half, 9)):
            i = r // 3
            j = r % 3
            U[i*m+j, r] += 1.0
            V[j*3 + (r % 3), r] += 1.0
            W[(r % 3)*n + i, r] += 1.0
        
        # Second half: handles columns 3-5 of B
        U[:, half:] = np.random.randn(n * m, rank - half) * 0.1
        V[m*3:, half:] = np.random.randn(m * 3, rank - half) * 0.1
        W[3*n:, half:] = np.random.randn(3 * n, rank - half) * 0.1
        for r in range(min(rank - half, 9)):
            i = r // 3
            j = r % 3
            U[i*m+j, half + r] += 1.0
            V[j*3 + 3 + (r % 3), half + r] += 1.0
            W[(3 + (r % 3))*n + i, half + r] += 1.0
    
    return U, V, W

def _random_init(n, m, p, rank, seed=0):
    """Random initialization."""
    np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    return U, V, W

def _try_find_decomposition_fast(n, m, p, target_rank, max_time=10.0, seed=0, precomputed_indices=None):
    """
    Try to find a rank-target_rank decomposition using local optimization.
    """
    if precomputed_indices is None:
        precomputed_indices = _build_precomputed_indices(n, m, p)
    
    start_time = time.time()
    
    # Try multiple initializations
    for init_type in ['structured', 'random']:
        if time.time() - start_time > max_time - 1:
            break
        
        if init_type == 'structured':
            U_init, V_init, W_init = _structured_init_block(n, m, p, target_rank, seed=seed)
        else:
            U_init, V_init, W_init = _random_init(n, m, p, target_rank, seed=seed)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Try local optimization
        try:
            result = minimize(_objective_fast, x0, args=(n, m, p, target_rank, precomputed_indices),
                            method='L-BFGS-B', 
                            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if result.fun < 1e-10:
                U_found = result.x[:n*m*target_rank].reshape(n * m, target_rank)
                V_found = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m * p, target_rank)
                W_found = result.x[n*m*target_rank + m*p*target_rank :].reshape(p * n, target_rank)
                
                err = _check_decomposition(U_found, V_found, W_found, n, m, p)
                if err < 1e-8:
                    return U_found, V_found, W_found
        except Exception:
            pass
        
        # Try with different seeds
        for s in range(1, 3):
            if time.time() - start_time > max_time - 1:
                break
            
            if init_type == 'structured':
                U_init, V_init, W_init = _structured_init_block(n, m, p, target_rank, seed=seed + s * 100)
            else:
                U_init, V_init, W_init = _random_init(n, m, p, target_rank, seed=seed + s * 100)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(_objective_fast, x0, args=(n, m, p, target_rank, precomputed_indices),
                                method='L-BFGS-B', 
                                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if result.fun < 1e-10:
                    U_found = result.x[:n*m*target_rank].reshape(n * m, target_rank)
                    V_found = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m * p, target_rank)
                    W_found = result.x[n*m*target_rank + m*p*target_rank :].reshape(p * n, target_rank)
                    
                    err = _check_decomposition(U_found, V_found, W_found, n, m, p)
                    if err < 1e-8:
                        return U_found, V_found, W_found
            except Exception:
                pass
    
    return None, None, None

def _high_precision_polish(U, V, W, n, m, p, max_iter=1000, precomputed_indices=None):
    """
    High-precision polish step to tighten the solution.
    """
    if precomputed_indices is None:
        precomputed_indices = _build_precomputed_indices(n, m, p)
    
    rank = U.shape[1]
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = minimize(_objective_fast, x0, args=(n, m, p, rank, precomputed_indices), method='L-BFGS-B',
                        options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14})
        
        if result.fun < 1e-12:
            U_polished = result.x[:n*m*rank].reshape(n * m, rank)
            V_polished = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m * p, rank)
            W_polished = result.x[n*m*rank + m*p*rank :].reshape(p * n, rank)
            
            err = _check_decomposition(U_polished, V_polished, W_polished, n, m, p)
            if err < 1e-10:
                return U_polished, V_polished, W_polished
    except Exception:
        pass
    
    return U, V, W

def _try_rank_reduction(n, m, p, start_rank, end_rank, deadline, precomputed_indices=None):
    """Try to find decompositions with rank between start_rank and end_rank."""
    if precomputed_indices is None:
        precomputed_indices = _build_precomputed_indices(n, m, p)
    
    best_rank = None
    best_U = best_V = best_W = None
    
    for target_rank in range(start_rank, end_rank + 1):
        if time.time() > deadline - 5:
            break
        
        found = False
        for seed in range(4):
            if time.time() > deadline - 5:
                break
            
            U_r, V_r, W_r = _try_find_decomposition_fast(n, m, p, target_rank, 
                                                         max_time=8.0, seed=seed,
                                                         precomputed_indices=precomputed_indices)
            
            if U_r is not None:
                U_r, V_r, W_r = _high_precision_polish(U_r, V_r, W_r, n, m, p, 
                                                       precomputed_indices=precomputed_indices)
                
                U_rounded = _to_half_int(U_r)
                V_rounded = _to_half_int(V_r)
                W_rounded = _to_half_int(W_r)
                
                if _check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p) < 1e-6:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                else:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                
                found = True
                break
        
        if found:
            break
    
    return best_rank, best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0
    
    # Build precomputed indices once
    precomputed_indices = _build_precomputed_indices(n, m, p)
    
    # Start with schoolbook as fallback
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    err = _check_decomposition(U, V, W, n, m, p)
    if err >= 1e-10:
        U_out = [[float(x) for x in row] for row in best_U]
        V_out = [[float(x) for x in row] for row in best_V]
        W_out = [[float(x) for x in row] for row in best_W]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Try to find lower rank decompositions
    # The theoretical lower bound for 3x3x6 is at least max(3*3, 3*6, 6*3) = 18
    # But realistically, we're looking for something below 54
    # Try ranks from 20 upward, but prioritize common values
    found_rank, found_U, found_V, found_W = _try_rank_reduction(n, m, p, 20, 54, deadline, 
                                                                  precomputed_indices=precomputed_indices)
    
    if found_rank is not None:
        best_rank = found_rank
        best_U, best_V, best_W = found_U, found_V, found_W
    
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
