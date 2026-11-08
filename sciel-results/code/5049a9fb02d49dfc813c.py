import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _build_known_66():
    n, m, p = DIMENSIONS
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0

    def add_rank1(u_idx, u_vals, v_idx, v_vals, w_idx, w_vals):
        nonlocal current_rank
        if current_rank >= rank:
            return
        for idx, val in zip(u_idx, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_idx, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_idx, w_vals):
            W[idx][current_rank] += val
        current_rank += 1

    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])

    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])

    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)

    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)

    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    import numpy as np
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    R = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(R):
                    s += Ua[i * m + j, t] * Va[j * p + k, t] * Wa[k * n + i, t]
                if abs(s - 1) > 1e-6:
                    return False
    return True

def _try_reduce_rank_local_search(U, V, W, n, m, p, target_rank, deadline):
    """
    Attempts to reduce rank by dropping columns and re-solving for one factor.
    """
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    R0 = Ua.shape[1]
    
    if R0 <= target_rank:
        return U, V, W

    # We want to reach target_rank.
    # We will try to drop (R0 - target_rank) columns.
    # Strategy: Iteratively drop one column, and re-solve for W (or U or V) to maintain exactness.
    
    # We need to map (i,j,k) to an index in the tensor.
    # T[i,j,k] = 1.
    # Equation: sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1.
    
    # Let's fix U and V, and solve for W.
    # For a fixed t, W[k*n+i, t] is a variable.
    # But W is shared across t.
    # Actually, for each t, we have a set of variables W[:, t].
    # The equation is linear in W[:, t] if U and V are fixed.
    # sum_t (U[i*m+j, t] V[j*p+k, t]) W[k*n+i, t] = 1.
    # This is a system of n*m*p equations.
    # Variables are W[k*n+i, t] for all k,i,t.
    # Number of variables: p*n*R.
    # Number of equations: n*m*p.
    # If R is large, we have more variables than equations (for n=3,m=4,p=6, R=66, vars=1188, eqs=72).
    # So we can find a solution.
    
    # However, we want to ZERO OUT specific columns of U, V, W.
    # If we zero out column t in U, V, W, we remove that term from the sum.
    # We need to re-solve for the remaining W (or U/V) to compensate.
    
    # Let's try to drop columns one by one.
    current_U = Ua.copy()
    current_V = Va.copy()
    current_W = Wa.copy()
    current_R = R0
    
    # We will try to drop the last (R0 - target_rank) columns.
    cols_to_drop = R0 - target_rank
    
    # To make it robust, we can try different orders of dropping.
    # For now, we just drop the last ones.
    
    for drop_iter in range(cols_to_drop):
        if time.time() > deadline - 5:
            return U, V, W
            
        # Drop the last column (index current_R - 1)
        drop_idx = current_R - 1
        
        # Zero out this column in U, V, W
        current_U[:, drop_idx] = 0
        current_V[:, drop_idx] = 0
        current_W[:, drop_idx] = 0
        
        # Now we need to adjust the remaining columns (0 to current_R - 2) to satisfy the tensor.
        # We can re-solve for W, keeping U and V fixed.
        # But U and V also have zeros in the dropped column.
        # The remaining U and V are current_U[:, :current_R-1] and current_V[:, :current_R-1].
        
        # Let's set up the linear system for W.
        # We want sum_{t=0}^{R-2} U[i,j,t] V[j,k,t] W[k,i,t] = 1.
        
        # Build the matrix A and vector b.
        # Variable x corresponds to W[k*n+i, t].
        # Index in x: t * (p*n) + (k*n + i).
        # Number of variables: (current_R - 1) * p * n.
        
        n_vars = (current_R - 1) * p * n
        n_eqs = n * m * p
        
        A = np.zeros((n_eqs, n_vars))
        b = np.ones(n_eqs)
        
        # Fill A
        # For each equation (i, j, k):
        # Sum_{t=0}^{R-2} U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(current_R - 1):
                        coeff = current_U[i * m + j, t] * current_V[j * p + k, t]
                        var_idx = t * (p * n) + (k * n + i)
                        A[eq_idx, var_idx] += coeff
                    eq_idx += 1
        
        # Solve A x = b
        try:
            # Use least squares
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            if len(residuals) > 0 and residuals[0] > 1e-8:
                # Solution is not exact, revert
                current_U[:, drop_idx] = Ua[:, drop_idx] # Restore from original? No, we modified in place.
                # We should keep a backup.
                # For now, if it fails, we break and return the previous state.
                # But we modified current_U.
                # Let's just return the original if it fails.
                return U, V, W
            else:
                # Update W
                for t in range(current_R - 1):
                    for k in range(p):
                        for i in range(n):
                            current_W[k * n + i, t] = x[t * (p * n) + (k * n + i)]
                
                # Round to half-integers
                def round_half(a):
                    return np.round(2 * a) / 2.0
                    
                current_W = round_half(current_W)
                
                # Verify
                if _validate(current_U.tolist(), current_V.tolist(), current_W.tolist(), n, m, p):
                    current_R -= 1
                    # Remove the last column to keep matrices compact?
                    # Or just keep the zero column and ignore it?
                    # The rank is determined by the number of non-zero columns?
                    # "score is the rank". If we have a zero column, is it counted?
                    # "factor matrices of shapes ... (n*m, rank)".
                    # If we return a matrix with a zero column, the shape is still (n*m, R0).
                    # The evaluator likely counts the shape's second dimension as rank.
                    # So we MUST trim the matrix.
                    current_U = current_U[:, :current_R]
                    current_V = current_V[:, :current_R]
                    current_W = current_W[:, :current_R]
                else:
                    # Rounding broke it, revert
                    return U, V, W
        except Exception:
            return U, V, W

    return current_U.tolist(), current_V.tolist(), current_W.tolist()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0
    
    known = _build_known_66()
    U, V, W = known["U"], known["V"], known["W"]
    
    if not _validate(U, V, W, n, m, p):
        return known

    # Try to reduce to 60
    if time.time() < deadline - 20:
        U_new, V_new, W_new = _try_reduce_rank_local_search(U, V, W, n, m, p, 60, deadline)
        if U_new is not U:
            if _validate(U_new, V_new, W_new, n, m, p):
                return {"U": U_new, "V": V_new, "W": W_new}
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
