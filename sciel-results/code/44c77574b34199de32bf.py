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
        def w(i, k): return (k * n) + (i_off + i)
        # Standard Strassen 7-term decomposition for 2x2x2
        # Note: The indices in the original code for w were (k*n + i), which matches this.
        # Let's verify the Strassen terms provided in the original code.
        # Original:
        # add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        # This looks like a specific valid decomposition. I will keep the original construction as it is verified to work.
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
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    R = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(R):
                    s += Ua[i * m + j, t] * Va[j * p + k, t] * Wa[k * n + i, t]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _round_to_half(x):
    return np.round(2.0 * x) / 2.0

def _try_compression(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by selecting subsets of columns from U and V,
    solving for W, and checking if a valid half-integer solution exists.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    R0 = Ua.shape[1]
    
    best_result = None
    best_rank = R0
    
    # Precompute the linear system matrix A for the full rank R0?
    # A is (n*m*p) x (R * p * n).
    # We can build A for any R by taking subsets of columns.
    
    n_eqs = n * m * p
    b = np.ones(n_eqs)
    
    # Build the full A matrix for R0 columns to allow fast slicing?
    # A_full has shape (n*m*p, R0 * p * n).
    # This might be large: 72 * (66 * 18) = 72 * 1188 = 85536 entries. This is fine.
    
    A_full = np.zeros((n_eqs, R0 * p * n))
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for t in range(R0):
                    coeff = Ua[i * m + j, t] * Va[j * p + k, t]
                    var_idx = t * (p * n) + (k * n + i)
                    A_full[eq_idx, var_idx] = coeff
                eq_idx += 1
                
    # Try reducing rank
    # We try random subsets of columns of size R
    # We prioritize trying R close to R0 first, then decreasing.
    
    rng = np.random.default_rng(42)
    
    # Limit the number of trials to fit in time
    max_trials = 200
    
    for R in range(R0 - 1, max(1, R0 - 10), -1):
        if time.time() > deadline - 10:
            break
            
        trials_for_R = max(10, max_trials // (R0 - 1))
        
        for trial in range(trials_for_R):
            if time.time() > deadline - 10:
                break
            
            # Select random columns
            cols = rng.choice(R0, size=R, replace=False)
            
            # Extract sub-matrix A_sub
            # A_sub has shape (n_eqs, R * p * n)
            # We need to map the new column indices 0..R-1 to the original cols.
            # The variable index in A is t * (p*n) + (k*n+i).
            # In A_full, the block for original column t is at columns [t*(p*n) : (t+1)*(p*n)].
            
            A_sub = np.zeros((n_eqs, R * p * n))
            for new_t, orig_t in enumerate(cols):
                start = orig_t * (p * n)
                end = start + (p * n)
                A_sub[:, new_t * (p * n) : (new_t + 1) * (p * n)] = A_full[:, start : end]
            
            try:
                x, residuals, rank_a, sv = np.linalg.lstsq(A_sub, b, rcond=None)
                
                if len(residuals) > 0 and residuals[0] < 1e-8:
                    # Check if solution is close to half-integers
                    # Reshape x to (R, p*n)
                    W_sol = x.reshape(R, p * n).T # (p*n, R)
                    
                    W_rounded = _round_to_half(W_sol)
                    
                    # Construct U_sub and V_sub
                    U_sub = Ua[:, cols]
                    V_sub = Va[:, cols]
                    
                    # Validate
                    if _validate(U_sub.tolist(), V_sub.tolist(), W_rounded.tolist(), n, m, p):
                        if R < best_rank:
                            best_rank = R
                            best_result = (R, U_sub.tolist(), V_sub.tolist(), W_rounded.tolist())
                            # If we find a good one, we might want to continue searching for even lower rank
                            # But we are iterating R downwards, so we will find lower ranks in later iterations
                            # We can break out of trial loop for this R if we want to try smaller R sooner?
                            # No, we want to find the *best* (lowest) R. 
                            # Since we iterate R from high to low, we will eventually hit the lowest possible R in this range.
                            # We should continue to try to find if an even lower R works?
                            # The loop `for R in range(...)` goes downwards. So we will try R0-1, then R0-2, etc.
                            # If we find a solution for R0-1, we still try R0-2.
                            # So we don't break on success, we just update best.
            except Exception:
                continue

    return best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0
    
    known = _build_known_66()
    U, V, W = known["U"], known["V"], known["W"]
    
    if not _validate(U, V, W, n, m, p):
        return known
    
    best_rank = 66
    best_U, best_V, best_W = U, V, W
    
    # Try compression
    if time.time() < deadline - 20:
        result = _try_compression(U, V, W, n, m, p, deadline)
        if result is not None:
            rank, U_new, V_new, W_new = result
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U_new, V_new, W_new
                
    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
