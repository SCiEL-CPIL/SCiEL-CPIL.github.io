import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 5]
DEADLINE = 140.0  # Safety margin

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give a valid decomposition of the multiplication tensor."""
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    # Vectorized check: compute all (i,j,k) entries at once
    T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _snap_to_half_integers(mat):
    """Snap values to nearest half-integer."""
    return np.round(mat * 2) / 2

def _structured_init(n, m, p, rank, seed=0):
    """
    Initialize U, V, W with structure that respects the multiplication pattern.
    Use Kronecker-like structure to give the optimizer a better starting point.
    """
    np.random.seed(seed)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # For each (i,j,k) in the multiplication, we want contributions to sum to 1.
    # Initialize with random signs and small magnitudes, but ensure each (i,j,k) 
    # has some initial "coverage" by assigning random subsets of columns.
    
    # Create a mapping: for each (i,j,k), pick a subset of columns that will contribute
    # This helps avoid the optimizer getting stuck in bad local minima.
    
    # Strategy: partition the rank into groups, one for each (i,j,k), with overlap
    # Actually, let's use a simpler approach: random initialization but with 
    # structured sparsity patterns.
    
    # For each column c, assign it to a random (i,j,k) pattern
    for c in range(rank):
        # Randomly choose which (i,j) pairs this column affects in U
        # and which (j,k) pairs in V, which (k,i) pairs in W
        # But ensure consistency: if U[i*m+j, c] != 0 and V[j*p+k, c] != 0,
        # then W[k*n+i, c] should be non-zero to contribute to T[i,j,k]
        
        # Pick a random j
        j = np.random.randint(m)
        # Pick a random i
        i = np.random.randint(n)
        # Pick a random k
        k = np.random.randint(p)
        
        # Assign with some probability to make it sparse
        if np.random.random() < 0.7:
            U[i*m+j, c] = np.random.choice([-1, 1]) * np.sqrt(1.0/rank)
            V[j*p+k, c] = np.random.choice([-1, 1]) * np.sqrt(1.0/rank)
            W[k*n+i, c] = np.random.choice([-1, 1]) * np.sqrt(1.0/rank)
    
    return U, V, W

def _try_rank(n, m, p, target_rank, time_limit=8):
    """Try to find a decomposition of target_rank using optimization."""
    start_time = time.time()
    T = _build_tensor(n, m, p)
    
    def objective(params):
        U = params[:n*m*target_rank].reshape(n*m, target_rank)
        V = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W = params[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.sum((T_recon - T) ** 2)
    
    # Try multiple initializations: structured and random
    init_strategies = []
    
    # Strategy 1: Structured initialization
    for seed in range(3):
        U0, V0, W0 = _structured_init(n, m, p, target_rank, seed=seed)
        init_strategies.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))
    
    # Strategy 2: Random initialization
    for seed in range(3):
        np.random.seed(seed + 100)
        init_params = np.random.normal(0, 0.5, size=(n*m + m*p + p*n) * target_rank)
        init_strategies.append(init_params)
    
    # Strategy 3: Near-schoolbook initialization (start from schoolbook and project)
    if target_rank < n*m*p:
        # Create a "compressed" version of schoolbook
        U0 = np.zeros((n*m, target_rank))
        V0 = np.zeros((m*p, target_rank))
        W0 = np.zeros((p*n, target_rank))
        # Distribute the 45 terms across target_rank columns
        for idx in range(n*m*p):
            i, j, k = idx // (m*p), (idx // p) % m, idx % p
            c = idx % target_rank
            U0[i*m+j, c] += 1.0 / max(1, n*m*p // target_rank)
            V0[j*p+k, c] += 1.0 / max(1, n*m*p // target_rank)
            W0[k*n+i, c] += 1.0 / max(1, n*m*p // target_rank)
        init_strategies.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))
    
    for init_params in init_strategies:
        if time.time() > start_time + time_limit:
            break
        
        # Use L-BFGS-B with tighter tolerances
        try:
            res = opt.minimize(objective, init_params, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            if res.fun < 1e-8:
                # Extract solution
                U = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W = res.x[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
                
                # Snap to half-integers
                U_snapped = _snap_to_half_integers(U)
                V_snapped = _snap_to_half_integers(V)
                W_snapped = _snap_to_half_integers(W)
                
                if _verify(U_snapped, V_snapped, W_snapped, n, m, p):
                    return U_snapped, V_snapped, W_snapped
                else:
                    # Try without snapping if it's very close
                    if _verify(U, V, W, n, m, p):
                        return U, V, W
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Try to improve: search from lower ranks upward, but prioritize known good ranks
    # For 3x3x5, known lower bounds suggest rank around 30-40 might be achievable
    # Try a range of ranks, starting from more promising ones
    
    # Ranks to try, in order of priority
    ranks_to_try = list(range(30, best_rank))
    # Also try some lower ranks in case we're lucky
    ranks_to_try = [25, 28, 30, 32, 35, 38, 40, 42, 43, 44] + [r for r in ranks_to_try if r not in [25, 28, 30, 32, 35, 38, 40, 42, 43, 44]]
    
    for target_rank in ranks_to_try:
        if target_rank >= best_rank:
            continue
        if time.time() > DEADLINE - 15:
            break
        
        result = _try_rank(n, m, p, target_rank, time_limit=6)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            # Continue searching for even lower ranks if time permits
    
    # Convert to list-of-lists
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                v = round(val * 2) / 2
                if abs(v - round(v)) < 1e-9:
                    new_row.append(int(round(v)))
                else:
                    new_row.append(v)
            result.append(new_row)
        return result
    
    U_out = to_list(best_U)
    V_out = to_list(best_V)
    W_out = to_list(best_W)
    
    # Final verification
    if not _verify(np.array(U_out, dtype=float), np.array(V_out, dtype=float), 
                   np.array(W_out, dtype=float), n, m, p):
        # Fall back to schoolbook
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        U_out = to_list(U_sb)
        V_out = to_list(V_sb)
        W_out = to_list(W_sb)
    
    return {"U": U_out, "V": V_out, "W": W_out}
