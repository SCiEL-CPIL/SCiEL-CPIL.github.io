import time
import numpy as np
from itertools import combinations

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

def _try_remove_columns(U, V, W, n, m, p, cols_to_remove, deadline):
    """
    Try to set the specified columns to zero and solve for the remaining entries
    to maintain the exact decomposition. Returns (U_new, V_new, W_new) or None.
    """
    R0 = len(U[0])
    keep_cols = [c for c in range(R0) if c not in cols_to_remove]
    R_new = len(keep_cols)
    
    # Build linear system: for each (i,j,k), sum over kept t of
    # U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    # This is nonlinear in general, but we can try to solve it iteratively
    # by fixing two of the three matrices and solving for the third.
    
    # Strategy: fix V and W at their original values (for kept columns),
    # then solve for U. Then fix U and W, solve for V. Then fix U and V, solve for W.
    # Iterate a few times.
    
    U_orig = np.array(U, dtype=float)
    V_orig = np.array(V, dtype=float)
    W_orig = np.array(W, dtype=float)
    
    U_cur = np.zeros((n*m, R_new))
    V_cur = V_orig[:, keep_cols].copy()
    W_cur = W_orig[:, keep_cols].copy()
    
    # Initialize U_cur with original values
    U_cur = U_orig[:, keep_cols].copy()
    
    for iteration in range(20):
        if time.time() > deadline - 15:
            return None
        
        # Solve for U: for each (i,j,k), we need
        # sum_t U[i*m+j,t] * V_cur[j*p+k,t] * W_cur[k*n+i,t] = 1
        # For fixed V_cur and W_cur, this is a linear system in U.
        # But it's coupled across k for the same (i,j).
        # Actually, for each (i,j), we have p equations in R_new unknowns (one per t).
        # Let's build the system properly.
        
        # For each (i,j), define:
        # A[i,j] is a p x R_new matrix where A[i,j][k,t] = V_cur[j*p+k,t] * W_cur[k*n+i,t]
        # b[i,j] is a p-vector of ones
        # We want U_cur[i*m+j, :] such that A[i,j] @ U_cur[i*m+j, :].T = b[i,j]
        
        U_new = np.zeros((n*m, R_new))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R_new))
                for k in range(p):
                    A[k, :] = V_cur[j*p+k, :] * W_cur[k*n+i, :]
                b = np.ones(p)
                # Solve A @ x = b, where x = U_cur[i*m+j, :].T
                # Use least squares
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = x
                except:
                    return None
        
        # Check if solution is good
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(R_new):
                        s += U_new[i*m+j, t] * V_cur[j*p+k, t] * W_cur[k*n+i, t]
                    err += abs(s - 1)
        if err < 1e-8:
            break
        
        # Now solve for V similarly
        V_new = np.zeros((m*p, R_new))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R_new))
                for i in range(n):
                    A[i, :] = U_new[i*m+j, :] * W_cur[k*n+i, :]
                b = np.ones(n)
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = x
                except:
                    return None
        
        # And solve for W
        W_new = np.zeros((p*n, R_new))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R_new))
                for j in range(m):
                    A[j, :] = U_new[i*m+j, :] * V_new[j*p+k, :]
                b = np.ones(m)
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = x
                except:
                    return None
        
        U_cur = U_new
        V_cur = V_new
        W_cur = W_new
    
    # Final check
    err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(R_new):
                    s += U_cur[i*m+j, t] * V_cur[j*p+k, t] * W_cur[k*n+i, t]
                err += abs(s - 1)
    
    if err > 1e-6:
        return None
    
    # Try rounding to integers
    U_r = np.round(U_cur)
    V_r = np.round(V_cur)
    W_r = np.round(W_cur)
    
    if _validate(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # Try half-integers
    U_h = np.round(2*U_cur)/2
    V_h = np.round(2*V_cur)/2
    W_h = np.round(2*W_cur)/2
    
    if _validate(U_h.tolist(), V_h.tolist(), W_h.tolist(), n, m, p):
        return U_h.tolist(), V_h.tolist(), W_h.tolist()
    
    # Try mixed rounding
    for u_round in [np.round, lambda x: np.round(2*x)/2]:
        for v_round in [np.round, lambda x: np.round(2*x)/2]:
            for w_round in [np.round, lambda x: np.round(2*x)/2]:
                Ur = u_round(U_cur)
                Vr = v_round(V_cur)
                Wr = w_round(W_cur)
                if _validate(Ur.tolist(), Vr.tolist(), Wr.tolist(), n, m, p):
                    return Ur.tolist(), Vr.tolist(), Wr.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    known = _build_known_66()
    U, V, W = known["U"], known["V"], known["W"]
    
    if not _validate(U, V, W, n, m, p):
        return known
    
    R0 = len(U[0])
    
    # Try removing pairs of columns to get rank 64
    if time.time() < deadline - 30:
        for c1, c2 in combinations(range(R0), 2):
            if time.time() > deadline - 20:
                break
            result = _try_remove_columns(U, V, W, n, m, p, {c1, c2}, deadline)
            if result is not None:
                U_new, V_new, W_new = result
                if _validate(U_new, V_new, W_new, n, m, p):
                    return {"U": U_new, "V": V_new, "W": W_new}
    
    # Try removing 4 columns to get rank 62 (try a few random subsets)
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for _ in range(50):
            if time.time() > deadline - 20:
                break
            cols = set(rng.choice(R0, size=4, replace=False).tolist())
            result = _try_remove_columns(U, V, W, n, m, p, cols, deadline)
            if result is not None:
                U_new, V_new, W_new = result
                if _validate(U_new, V_new, W_new, n, m, p):
                    return {"U": U_new, "V": V_new, "W": W_new}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
