# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the schoolbook rank-42 decomposition."""
    n, m, p = DIMENSIONS
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    return U, V, W

def _verify_full(U, V, W, tol=1e-9):
    """Check that the decomposition exactly reconstructs the multiplication tensor, including zeros."""
    n, m, p = DIMENSIONS
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    r = U_np.shape[1]
    
    # Precompute the set of valid (u, v, w) indices that should be 1
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_indices.add((u, v, w))
                
    # Check all possible combinations
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                val = 0.0
                for c in range(r):
                    val += U_np[u, c] * V_np[v, c] * W_np[w, c]
                
                if (u, v, w) in valid_indices:
                    if abs(val - 1.0) > tol:
                        return False
                else:
                    if abs(val - 0.0) > tol:
                        return False
    return True

def _to_list(M):
    """Convert numpy array to list of lists with int or half-int entries."""
    result = []
    for row in M:
        r = []
        for val in row:
            hv = round(val * 2) / 2.0
            if abs(hv - round(hv)) < 1e-9:
                r.append(int(round(hv)))
            else:
                r.append(hv)
        result.append(r)
    return result

def _try_find_decomp_for_rank(r, max_trials=100, seed=42):
    """
    Try to find a rank-r decomposition.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    num_eq = n * m * p  # 42
    num_W = p * n * r   # 14*r
    
    for trial in range(max_trials):
        # Random U and V with entries in {-1, 0, 1}
        # Using sparse random matrices might help
        U = rng.randint(-1, 2, size=(n * m, r)).astype(float)
        V = rng.randint(-1, 2, size=(m * p, r)).astype(float)
        
        # Build coefficient matrix A and RHS b for W
        # We only use the non-zero constraints to solve for W
        # Then we verify the zero constraints separately
        A = np.zeros((num_eq, num_W))
        b = np.ones(num_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    for c in range(r):
                        w_idx = k * n + i
                        A[eq_idx, w_idx * r + c] = U[u_idx, c] * V[v_idx, c]
                    eq_idx += 1
        
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        
        # Check if solution is close to half-integers
        W_rounded = np.round(W_flat * 2) / 2.0
        if np.max(np.abs(W_flat - W_rounded)) > 1e-6:
            continue
        
        W = W_rounded.reshape(p * n, r)
        
        # Full verification including zeros
        if _verify_full(U, V, W, tol=1e-9):
            return U, V, W
            
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p  # 42
    
    # Try ranks from 21 to 41
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 10.0:
            break
            
        # Use more trials for lower ranks
        max_trials = 50 if target_rank < 30 else 20
        result = _try_find_decomp_for_rank(target_rank, max_trials=max_trials, seed=target_rank * 1000 + 42)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break
            
    U_final = _to_list(best_U)
    V_final = _to_list(best_V)
    W_final = _to_list(best_W)
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
