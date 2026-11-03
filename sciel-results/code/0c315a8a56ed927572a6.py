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

def _try_eliminate_term(U, V, W, n, m, p, rank, term_idx):
    """Try to eliminate term term_idx by expressing it as combination of others.
    Returns (new_U, new_V, new_W, new_rank) if successful, else None."""
    # Build constraint matrix: for each (i,l,k), the contribution of term term_idx
    # must be matched by other terms.
    # Actually, we need to find coefficients c_r for r != term_idx such that
    # for all (i,l,k): U[i*m+l, term_idx]*V[l*p+k, term_idx]*W[k*n+i, term_idx]
    #   == sum_{r != term_idx} c_r * U[i*m+l, r]*V[l*p+k, r]*W[k*n+i, r]
    # This is a system of n*m*p equations in (rank-1) unknowns.
    
    n_eq = n * m * p  # 60 equations
    n_unk = rank - 1  # 53 unknowns
    
    # Build the system: for each (i,l,k), equation is:
    # sum_{r != term_idx} c_r * U[i*m+l, r]*V[l*p+k, r]*W[k*n+i, r] = 
    #   U[i*m+l, term_idx]*V[l*p+k, term_idx]*W[k*n+i, term_idx]
    
    # Only include equations where the RHS is nonzero
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
    
    # Solve least squares
    try:
        coeffs, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
    except:
        return None
    
    # Check if solution is good
    if len(residuals) > 0 and residuals[0] > 1e-8 * max(1.0, np.linalg.norm(b_vec)):
        return None
    
    # Check if coefficients are close to half-integers (for validity)
    # Actually, the problem allows integer/half-integer entries
    # Let's check if the solution is reasonable
    if np.max(np.abs(coeffs)) > 100:
        return None
    
    # Build new matrices with term_idx removed
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
    
    # Verify
    if not _verify_decomp(new_U, new_V, new_W, n, m, p, new_rank):
        return None
    
    return (new_U, new_V, new_W, new_rank)

def _try_swap_block_structure(n, m, p, j_pairs, k_pairs, remaining_j, U, V, W):
    """Try a different block structure and return best decomposition."""
    rank = len(j_pairs) * len(k_pairs) * 7 + len(remaining_j) * 2 * p
    U_new = [[0] * rank for _ in range(n * m)]
    V_new = [[0] * rank for _ in range(m * p)]
    W_new = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U_new, V_new, W_new, col, n, m, p)
    
    for j in remaining_j:
        for k in range(p):
            U_new[0*m + j][col] += 1
            V_new[j*p + k][col] += 1
            W_new[k*n + 0][col] += 1
            col += 1
            U_new[1*m + j][col] += 1
            V_new[j*p + k][col] += 1
            W_new[k*n + 1][col] += 1
            col += 1
    
    if col != rank:
        return None
    
    if not _verify_decomp(U_new, V_new, W_new, n, m, p, rank):
        return None
    
    return (U_new, V_new, W_new, rank)

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140
    
    # Default: 2 j-pairs + 1 remaining j, 3 k-pairs
    # Rank = 2*3*7 + 1*2*6 = 42 + 12 = 54
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
    
    if not _verify_decomp(U, V, W, n, m, p, rank):
        pass
    
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try alternative block structures
    alt_structures = [
        # 3 j-pairs (use all 5 j's: (0,1),(2,3),(4,?) - but need even, so try (0,1),(2,4),(3,?)
        # Actually with 5 j's we can do (0,1),(2,3) leaving 4, or (0,1),(2,4) leaving 3, etc.
        # Let's try (0,1),(2,4) leaving 3
        {'j_pairs': [(0, 1), (2, 4)], 'k_pairs': [(0, 1), (2, 3), (4, 5)], 'remaining': [3]},
        # (0,2),(1,3) leaving 4
        {'j_pairs': [(0, 2), (1, 3)], 'k_pairs': [(0, 1), (2, 3), (4, 5)], 'remaining': [4]},
        # Different k-pairings: (0,2),(1,3),(4,5)
        {'j_pairs': [(0, 1), (2, 3)], 'k_pairs': [(0, 2), (1, 3), (4, 5)], 'remaining': [4]},
        # (0,3),(1,2) for k
        {'j_pairs': [(0, 1), (2, 3)], 'k_pairs': [(0, 3), (1, 2), (4, 5)], 'remaining': [4]},
    ]
    
    for struct in alt_structures:
        if time.time() > _deadline - 30:
            break
        result = _try_swap_block_structure(n, m, p, struct['j_pairs'], struct['k_pairs'], struct['remaining'], U, V, W)
        if result is not None:
            u, v, w, r = result
            if r < best_rank:
                best_rank = r
                best_U = [row[:] for row in u]
                best_V = [row[:] for row in v]
                best_W = [row[:] for row in w]
    
    # Try to eliminate terms from the best decomposition
    if time.time() < _deadline - 60:
        current_rank = best_rank
        cur_U = [row[:] for row in best_U]
        cur_V = [row[:] for row in best_V]
        cur_W = [row[:] for row in best_W]
        
        # Try eliminating each term one at a time (greedy)
        for attempt in range(5):
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
