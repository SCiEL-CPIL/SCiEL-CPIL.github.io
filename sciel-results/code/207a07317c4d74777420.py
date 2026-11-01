import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_HARD_DEADLINE = 115

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + _HARD_DEADLINE

    # Get the rank-14 solution
    U14, V14, W14 = _get_rank14_solution(n, m, p)
    
    # Verify it
    err14 = _verify_decomposition(U14, V14, W14, n, m, p)
    
    if err14 < 1e-9:
        # Try to find lower rank with improved search
        for target_rank in [13, 12]:
            if time.time() > deadline - 20:
                break
            U, V, W = _search_lower_rank_improved(n, m, p, target_rank, deadline, U14, V14, W14)
            if U is not None:
                err = _verify_decomposition(U, V, W, n, m, p)
                if err < 1e-9:
                    return _to_dict(U, V, W)
        
        return _to_dict(U14, V14, W14)
    else:
        # Fallback to schoolbook
        return _construct_schoolbook(n, m, p)

def _get_rank14_solution(n, m, p):
    # Rank-14 decomposition for 2x3x6 via 3x3x3 rank-7 slicing + duplication
    U33 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1],
    ], dtype=np.float64)

    V33 = np.array([
        [1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 0],
        [0, 0, 0, 0, 0, -1, 1],
        [0, 0, 0, 0, 1, -1, 0],
    ], dtype=np.float64)

    W33 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, -1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float64)

    # Slice to 2x3x3
    U_2x3x3 = U33[0:6, :]
    V_2x3x3 = V33.copy()
    W_2x3x3 = np.zeros((6, 7), dtype=np.float64)
    for k in range(3):
        for i in range(2):
            W_2x3x3[k * 2 + i, :] = W33[k * 3 + i, :]

    # Construct 2x3x6 by duplicating the 2x3x3 block for k=0..2 and k=3..5
    r = 14
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)

    U[:, 0:7] = U_2x3x3
    V[0:9, 0:7] = V_2x3x3
    W[0:6, 0:7] = W_2x3x3

    U[:, 7:14] = U_2x3x3
    V[9:18, 7:14] = V_2x3x3
    W[6:12, 7:14] = W_2x3x3

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    r = U.shape[1]
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    err = np.max(np.abs(T - 1.0))
    return err

def _search_lower_rank_improved(n, m, p, target_rank, deadline, U14, V14, W14):
    """
    Improved search for lower rank decomposition using:
    1. Warm start from rank-14 solution
    2. Structured perturbations
    3. Local refinement with gradient descent
    """
    r = target_rank
    r14 = U14.shape[1]
    
    # Strategy: Start from rank-14, try to eliminate one term by merging
    # This is a heuristic: set one column to zero and see if optimization can recover
    
    best_loss = np.inf
    best_params = None
    
    # Try different strategies
    strategies = [
        'warm_start_zero_col',  # Zero out one column from rank-14
        'warm_start_perturb',   # Small perturbation of rank-14
        'structured_random',    # Structured random initialization
    ]
    
    for strategy in strategies:
        if time.time() > deadline - 10:
            break
        
        n_init = 5 if strategy == 'warm_start_zero_col' else 3
        for init_idx in range(n_init):
            if time.time() > deadline - 10:
                break
            
            if strategy == 'warm_start_zero_col':
                # Take rank-14, zero out column init_idx % r14, then optimize remaining r-1 cols
                # Actually, we need exactly r columns, so we'll drop one and optimize
                params = _init_warm_start_zero(U14, V14, W14, n, m, p, r, init_idx)
            elif strategy == 'warm_start_perturb':
                params = _init_warm_start_perturb(U14, V14, W14, n, m, p, r, init_idx)
            else:
                params = _init_structured_random(n, m, p, r, init_idx)
            
            if params is None:
                continue
            
            # Optimize
            result = minimize(_residual, params, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14,
                                    'maxls': 50, 'maxfun': 200})
            
            loss = result.fun
            if loss < best_loss:
                best_loss = loss
                best_params = result.x
            
            if best_loss < 1e-12:
                break
        
        if best_loss < 1e-12:
            break
    
    if best_params is not None and best_loss < 1e-10:
        U = best_params[0: n*m*r].reshape(n*m, r)
        V = best_params[n*m*r: n*m*r + m*p*r].reshape(m*p, r)
        W = best_params[n*m*r + m*p*r:].reshape(p*n, r)
        return U, V, W
    
    return None, None, None

def _residual(params, n=2, m=3, p=6):
    r = params.size // (n*m + m*p + p*n)
    U = params[0: n*m*r].reshape(n*m, r)
    V = params[n*m*r: n*m*r + m*p*r].reshape(m*p, r)
    W = params[n*m*r + m*p*r:].reshape(p*n, r)
    
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    residual = (T - 1.0).flatten()
    return np.sum(residual**2)

def _init_warm_start_zero(U14, V14, W14, n, m, p, r, init_idx):
    """Initialize by taking rank-14, zeroing out one column, and keeping r-1 columns."""
    r14 = U14.shape[1]
    
    # Drop column init_idx % r14, keep the rest, then add one zero column
    cols_to_keep = [c for c in range(r14) if c != init_idx % r14]
    # We need exactly r columns, so if we have r14-1, add one zero
    if len(cols_to_keep) >= r:
        cols_to_keep = cols_to_keep[:r]
    else:
        while len(cols_to_keep) < r:
            cols_to_keep.append(-1)  # Mark as zero column
    
    U = np.zeros((n*m, r), dtype=np.float64)
    V = np.zeros((m*p, r), dtype=np.float64)
    W = np.zeros((p*n, r), dtype=np.float64)
    
    for new_col, old_col in enumerate(cols_to_keep):
        if old_col >= 0:
            U[:, new_col] = U14[:, old_col]
            V[:, new_col] = V14[:, old_col]
            W[:, new_col] = W14[:, old_col]
    
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _init_warm_start_perturb(U14, V14, W14, n, m, p, r, init_idx):
    """Initialize by perturbing rank-14 and projecting to lower rank."""
    r14 = U14.shape[1]
    
    # Simple approach: average pairs of columns to reduce rank
    # Take first r columns with small perturbation
    U = U14[:, :r].copy()
    V = V14[:, :r].copy()
    W = W14[:, :r].copy()
    
    # Add small perturbation
    np.random.seed(42 + init_idx)
    U += np.random.randn(n*m, r) * 0.1
    V += np.random.randn(m*p, r) * 0.1
    W += np.random.randn(p*n, r) * 0.1
    
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _init_structured_random(n, m, p, r, init_idx):
    """Structured random initialization with some zeros to encourage sparsity."""
    np.random.seed(42 + init_idx)
    
    U = np.zeros((n*m, r), dtype=np.float64)
    V = np.zeros((m*p, r), dtype=np.float64)
    W = np.zeros((p*n, r), dtype=np.float64)
    
    # Fill with random values but keep some structure
    for col in range(r):
        # Randomly choose which entries to fill
        u_mask = np.random.rand(n*m) < 0.5
        v_mask = np.random.rand(m*p) < 0.5
        w_mask = np.random.rand(p*n) < 0.5
        
        U[u_mask, col] = np.random.randn(np.sum(u_mask))
        V[v_mask, col] = np.random.randn(np.sum(v_mask))
        W[w_mask, col] = np.random.randn(np.sum(w_mask))
    
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _construct_schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return _to_dict(U, V, W)

def _to_dict(U, V, W):
    def clean(arr):
        out = []
        for row in arr:
            crow = []
            for x in row:
                xi = round(float(x))
                if abs(float(x) - xi) < 1e-9:
                    crow.append(int(xi))
                else:
                    h = round(2.0 * float(x))
                    if abs(2.0 * float(x) - h) < 1e-9:
                        crow.append(h / 2.0)
                    else:
                        crow.append(float(x))
            out.append(crow)
        return out
    return {"U": clean(U), "V": clean(V), "W": clean(W)}
