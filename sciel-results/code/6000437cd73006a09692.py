"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col

def _build_baseline(n, m, p):
    """Build the proven rank-54 Strassen block decomposition."""
    default_j_pairs = [(0, 1), (2, 3)]
    default_k_pairs = [(0, 1), (2, 3), (4, 5)]
    default_remaining = [4]
    
    rank = 2 * 3 * 7 + 1 * 2 * 6  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0, j1 in default_j_pairs:
        for k0, k1 in default_k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p)
    
    for j in default_remaining:
        for k in range(p):
            U[0*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            U[1*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
    
    return U, V, W, rank

def _try_random_uv_solve_w(U, V, W, n, m, p, rank, seed=42):
    """Try to find W that completes the decomposition given fixed U, V.
    For each (i,l,k) where U[i*m+l,r]*V[l*p+k,r] != 0 for some r,
    we need sum_r U[i*m+l,r]*V[l*p+k,r]*W[k*n+i,r] = 1.
    This is a linear system in W entries.
    """
    # Build linear system: for each (i,l,k), equation:
    # sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r] = 1
    # Unknowns: W[k*n+i, r] for all k,i,r -> p*n*rank unknowns
    # Equations: n*m*p = 60
    
    n_unknowns = p * n * rank
    n_eqs = n * m * p
    
    A_mat = np.zeros((n_eqs, n_unknowns))
    b_vec = np.zeros(n_eqs)
    
    eq_idx = 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                for r in range(rank):
                    coeff = U[i*m+l][r] * V[l*p+k][r]
                    if abs(coeff) > 1e-12:
                        # W unknown index: k*n+i, column r
                        unk_idx = (k * n + i) * rank + r
                        A_mat[eq_idx, unk_idx] += coeff
                b_vec[eq_idx] = 1.0
                eq_idx += 1
    
    # Solve least squares
    try:
        sol, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-12)
    except:
        return None
    
    # Check residual
    if len(residuals) > 0 and residuals[0] > 1e-8:
        return None
    
    # Reshape solution to W
    W_new = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W_new[k*n+i][r] = sol[(k*n+i)*rank + r]
    
    # Round to half-integers and verify
    W_rounded = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                val = W_new[k*n+i][r]
                # Try rounding to nearest half-integer
                half = round(val * 2) / 2
                if abs(val - half) < 1e-6:
                    W_rounded[k*n+i][r] = half
                else:
                    W_rounded[k*n+i][r] = val
    
    if not _verify_decomp(U, V, W_rounded, n, m, p, rank):
        return None
    
    return W_rounded

def _try_structured_random(n, m, p, target_rank, max_trials=50, seed=0):
    """Try structured random U, V with {0,±1} entries, solve for W."""
    rng = np.random.RandomState(seed)
    
    for trial in range(max_trials):
        # Generate random sparse U with {0,±1} entries
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        
        # Fill U and V with random ±1 patterns
        for i in range(n * m):
            for r in range(target_rank):
                if rng.random() < 0.5:
                    U[i][r] = rng.choice([-1, 1])
        
        for l in range(m * p):
            for r in range(target_rank):
                if rng.random() < 0.5:
                    V[l][r] = rng.choice([-1, 1])
        
        # Try to solve for W
        W_result = _try_random_uv_solve_w(U, V, None, n, m, p, target_rank, seed=seed + trial)
        if W_result is not None:
            return U, V, W_result, target_rank
    
    return None

def _try_eliminate_term(U, V, W, n, m, p, rank, term_idx):
    """Try to eliminate term term_idx by expressing it as combination of others.
    Returns (new_U, new_V, new_W, new_rank) if successful, else None."""
    n_eq = n * m * p
    n_unk = rank - 1
    
    rows = []
    rhs = []
    for i in range(n):
        for l in range(m):
            for k in range(p):
                rhs_val = U[i*m+l][term_idx] * V[l*p+k][term_idx] * W[k*n+i][term_idx]
                if abs(rhs_val) > 1e-12:
                    row = []
                    for r in range(rank):
                        if r == term_idx:
                            continue
                        row.append(U[i*m+l][r] * V[l*p+k][r] * W[k*n+i][r])
                    rows.append(row)
                    rhs.append(rhs_val)
    
    if not rows:
        return None
    
    A_mat = np.array(rows, dtype=float)
    b_vec = np.array(rhs, dtype=float)
    
    try:
        coeffs, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
    except:
        return None
    
    if len(residuals) > 0 and residuals[0] > 1e-8 * max(1.0, np.linalg.norm(b_vec)):
        return None
    
    if np.max(np.abs(coeffs)) > 100:
        return None
    
    new_rank = rank - 1
    keep = [r for r in range(rank) if r != term_idx]
    
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    for new_idx, r in enumerate(keep):
        for i in range(n):
            for l in range(m):
                new_U[i*m+l][new_idx] = U[i*m+l][r] + coeffs[new_idx] * U[i*m+l][term_idx]
        for l in range(m):
            for k in range(p):
                new_V[l*p+k][new_idx] = V[l*p+k][r] + coeffs[new_idx] * V[l*p+k][term_idx]
        for k in range(p):
            for i in range(n):
                new_W[k*n+i][new_idx] = W[k*n+i][r] + coeffs[new_idx] * W[k*n+i][term_idx]
    
    if not _verify_decomp(new_U, new_V, new_W, n, m, p, new_rank):
        return None
    
    return (new_U, new_V, new_W, new_rank)

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140
    
    # Build baseline rank-54 decomposition
    U_base, V_base, W_base, rank_base = _build_baseline(n, m, p)
    
    best_rank = rank_base
    best_U = [row[:] for row in U_base]
    best_V = [row[:] for row in V_base]
    best_W = [row[:] for row in W_base]
    
    # Try to find lower rank via structured random approach
    # Start from rank 53 down to 40
    for target_rank in range(best_rank - 1, 39, -1):
        if time.time() > _deadline - 60:
            break
        
        # Try a few random seeds
        found = False
        for seed in range(10):
            if time.time() > _deadline - 60:
                break
            
            result = _try_structured_random(n, m, p, target_rank, max_trials=5, seed=seed * 100)
            if result is not None:
                u, v, w, r = result
                if r < best_rank:
                    best_rank = r
                    best_U = [row[:] for row in u]
                    best_V = [row[:] for row in v]
                    best_W = [row[:] for row in w]
                    found = True
                    break
        
        if found:
            # Found a lower rank, continue trying to go lower
            continue
        else:
            # Didn't find for this rank, try next lower
            continue
    
    # Try to eliminate terms from the best decomposition
    if time.time() < _deadline - 30:
        current_rank = best_rank
        cur_U = [row[:] for row in best_U]
        cur_V = [row[:] for row in best_V]
        cur_W = [row[:] for row in best_W]
        
        for attempt in range(3):
            if time.time() > _deadline - 30:
                break
            improved = False
            for term_idx in range(current_rank - 1, -1, -1):
                if time.time() > _deadline - 30:
                    break
                result = _try_eliminate_term(cur_U, cur_V, cur_W, n, m, p, current_rank, term_idx)
                if result is not None:
                    cur_U, cur_V, cur_W, current_rank = result
                    improved = True
                    best_rank = current_rank
                    best_U = [row[:] for row in cur_U]
                    best_V = [row[:] for row in cur_V]
                    best_W = [row[:] for row in cur_W]
                    break
            if not improved:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
