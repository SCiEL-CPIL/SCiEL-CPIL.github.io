"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1]))
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]))
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    terms.append(([1, 0, -1, 0], [1, 1, 0, 0], [0, 1, 0, 0]))
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]))
    return terms

def _make_schoolbook_terms(n, m, p, k_start=0, k_end=None):
    """Make schoolbook terms for columns k_start to k_end-1."""
    if k_end is None:
        k_end = p
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(k_start, k_end):
                u_vec = [0.0] * (n * m)
                u_vec[i * m + j] = 1.0
                v_vec = [0.0] * (m * p)
                v_vec[j * p + k] = 1.0
                w_vec = [0.0] * (p * n)
                w_vec[k * n + i] = 1.0
                terms.append((u_vec, v_vec, w_vec))
    return terms

def _build_from_terms(all_terms, n, m, p):
    """Build U, V, W matrices from a list of (u_vec, v_vec, w_vec) terms."""
    rank = len(all_terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r, (u_vec, v_vec, w_vec) in enumerate(all_terms):
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
    return U, V, W

def _tensor_error(U, V, W, n, m, p):
    """Compute the squared error of the tensor decomposition."""
    error = 0.0
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                error += (val - 1.0) ** 2
    return error

def _flatten_params(U, V, W):
    """Flatten U, V, W into a single vector."""
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _unflatten_params(params, n, m, p, rank):
    """Unflatten a parameter vector into U, V, W."""
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    U = params[:U_size].reshape(n * m, rank)
    V = params[U_size:U_size+V_size].reshape(m * p, rank)
    W = params[U_size+V_size:U_size+V_size+W_size].reshape(p * n, rank)
    return U, V, W

def _try_reduce_rank(U, V, W, target_rank, n, m, p, deadline, seed=0):
    """Try to reduce the rank of the decomposition to target_rank."""
    rng = np.random.RandomState(seed)
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return U, V, W
    
    # Initialize with the current decomposition, zeroing out some columns
    for attempt in range(10):
        if time.time() > deadline - 15:
            break
        
        # Start from current solution, randomly perturb and zero some columns
        U_init = np.zeros((n * m, target_rank))
        V_init = np.zeros((m * p, target_rank))
        W_init = np.zeros((p * n, target_rank))
        
        # Copy a subset of columns
        cols_to_keep = rng.choice(current_rank, target_rank, replace=False)
        U_init[:, :] = U[:, cols_to_keep] * 0.5
        V_init[:, :] = V[:, cols_to_keep] * 0.5
        W_init[:, :] = W[:, cols_to_keep] * 0.5
        
        # Add small random noise
        U_init += rng.randn(n * m, target_rank) * 0.01
        V_init += rng.randn(m * p, target_rank) * 0.01
        W_init += rng.randn(p * n, target_rank) * 0.01
        
        def objective(params):
            U_t, V_t, W_t = _unflatten_params(params, n, m, p, target_rank)
            return _tensor_error(U_t, V_t, W_t, n, m, p)
        
        x0 = _flatten_params(U_init, V_init, W_init)
        
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
        
        if result.fun < 1e-8:
            U_opt, V_opt, W_opt = _unflatten_params(result.x, n, m, p, target_rank)
            if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p, tol=1e-7):
                return U_opt, V_opt, W_opt
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Build the rank-13 decomposition: Strassen for cols 0,1 + schoolbook for cols 2,3,4
    strassen_terms = _strassen_2x2_terms()
    all_terms = []
    
    for u_vec, v_vec, w_vec in strassen_terms:
        u_full = list(u_vec)
        v_full = [0.0] * (m * p)
        for j in range(m):
            for k in range(2):
                v_full[j * p + k] = v_vec[j * 2 + k]
        w_full = [0.0] * (p * n)
        for k in range(2):
            for i in range(n):
                w_full[k * n + i] = w_vec[k * 2 + i]
        all_terms.append((u_full, v_full, w_full))
    
    schoolbook_terms = _make_schoolbook_terms(n, m, p, k_start=2, k_end=5)
    all_terms.extend(schoolbook_terms)
    
    U, V, W = _build_from_terms(all_terms, n, m, p)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to full schoolbook rank-20
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try to reduce rank
    for target_rank in [12, 11, 10]:
        if time.time() > deadline - 30:
            break
        
        result = _try_reduce_rank(best_U, best_V, best_W, target_rank, n, m, p, deadline)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = target_rank
            # Continue trying to reduce further
            continue
        else:
            # If we can't reduce to target_rank, try target_rank+1 if we haven't yet
            if target_rank < best_rank - 1:
                result = _try_reduce_rank(best_U, best_V, best_W, target_rank + 1, n, m, p, deadline)
                if result is not None:
                    U_r, V_r, W_r = result
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank + 1
            break
    
    # Convert to lists with clean values
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx, val in enumerate(row):
                if abs(val - round(val)) < 1e-10:
                    row[idx] = round(val)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
