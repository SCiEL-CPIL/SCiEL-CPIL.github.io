import numpy as np
import time
from scipy.optimize import least_squares, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9

    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    # Precompute target tensor entries
    ones_indices = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
    ones_set = set(ones_indices)
    all_indices = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)]
    zero_indices = [idx for idx in all_indices if idx not in ones_set]

    # Vectorized target setup for optimization
    a_idx = []
    b_idx = []
    c_idx = []
    targets = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                a_idx.append(a)
                b_idx.append(b)
                c_idx.append(c)
                targets.append(1.0 if (a, b, c) in ones_set else 0.0)
    a_idx = np.array(a_idx)
    b_idx = np.array(b_idx)
    c_idx = np.array(c_idx)
    targets = np.array(targets)

    def verify(U, V, W, tol=1e-9):
        """
        Verify if U, V, W form a valid decomposition.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check ones
        for a, b, c in ones_indices:
            s = np.sum(W[c, :] * U[a, :] * V[b, :])
            if abs(s - 1.0) > tol:
                return False
        # Check zeros
        for a, b, c in zero_indices:
            s = np.sum(W[c, :] * U[a, :] * V[b, :])
            if abs(s) > tol:
                return False
        return True

    def snap_to_half_integers(U, V, W):
        """Snap to nearest half-integers and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        return None

    # --- Step 1: Construct Rank-26 Solution (Strassen Core + Border) ---
    R26 = 26
    U26 = np.zeros((nm, R26))
    V26 = np.zeros((mp, R26))
    W26 = np.zeros((pn, R26))
    
    # Strassen terms for 2x2 core (indices 0,1)
    # r=0: (A00+A11)(B00+B11)
    U26[idx_u(0,0), 0] = 1; U26[idx_u(1,1), 0] = 1
    V26[idx_v(0,0), 0] = 1; V26[idx_v(1,1), 0] = 1
    W26[idx_w(0,0), 0] = 1; W26[idx_w(1,1), 0] = 1
    
    # r=1: (A10+A11)B00
    U26[idx_u(1,0), 1] = 1; U26[idx_u(1,1), 1] = 1
    V26[idx_v(0,0), 1] = 1
    W26[idx_w(0,1), 1] = 1; W26[idx_w(1,1), 1] = -1
    
    # r=2: A00(B01-B11)
    U26[idx_u(0,0), 2] = 1
    V26[idx_v(0,1), 2] = 1; V26[idx_v(1,1), 2] = -1
    W26[idx_w(1,0), 2] = 1; W26[idx_w(1,1), 2] = 1
    
    # r=3: A11(B10-B00)
    U26[idx_u(1,1), 3] = 1
    V26[idx_v(1,0), 3] = 1; V26[idx_v(0,0), 3] = -1
    W26[idx_w(0,0), 3] = 1; W26[idx_w(0,1), 3] = 1
    
    # r=4: (A00+A01)B11
    U26[idx_u(0,0), 4] = 1; U26[idx_u(0,1), 4] = 1
    V26[idx_v(1,1), 4] = 1
    W26[idx_w(0,0), 4] = -1; W26[idx_w(1,0), 4] = 1
    
    # r=5: (A10-A00)(B00+B01)
    U26[idx_u(1,0), 5] = 1; U26[idx_u(0,0), 5] = -1
    V26[idx_v(0,0), 5] = 1; V26[idx_v(0,1), 5] = 1
    W26[idx_w(1,1), 5] = 1
    
    # r=6: (A01-A11)(B10+B11)
    U26[idx_u(0,1), 6] = 1; U26[idx_u(1,1), 6] = -1
    V26[idx_v(1,0), 6] = 1; V26[idx_v(1,1), 6] = 1
    W26[idx_w(0,0), 6] = 1
    
    # Border terms (indices involving 2)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[idx_u(i,j), col] = 1
                V26[idx_v(j,k), col] = 1
                W26[idx_w(k,i), col] = 1
                col += 1
                
    assert col == 26, f"Expected 26 columns, got {col}"
    
    # Verify Rank 26
    if not verify(U26, V26, W26):
        # Fallback to Schoolbook Rank 27
        R27 = 27
        U27 = np.zeros((nm, R27))
        V27 = np.zeros((mp, R27))
        W27 = np.zeros((pn, R27))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U27[idx_u(i,j), c] = 1
                    V27[idx_v(j,k), c] = 1
                    W27[idx_w(k,i), c] = 1
                    c += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    best_U, best_V, best_W = U26, V26, W26
    best_rank = 26
    
    # --- Step 2: Try to find lower rank using optimization ---
    def try_optimize_rank(R, time_budget, init_U=None, init_V=None, init_W=None):
        """
        Attempt to find a valid decomposition of rank R.
        Uses a two-stage approach: 
        1. Minimize sum of squares (least_squares)
        2. If close, minimize max absolute error (minimize with smooth surrogate)
        """
        if time.time() > start_time + time_budget:
            return None
        
        best_cost = float('inf')
        best_x = None
        
        max_restarts = 5
        np.random.seed(42)
        
        for i in range(max_restarts):
            if time.time() > start_time + time_budget:
                break
            
            if i == 0 and init_U is not None:
                # Use provided initialization
                x0 = np.concatenate([
                    init_U.flatten(),
                    init_V.flatten(),
                    init_W.flatten()
                ])
            else:
                # Random initialization
                x0 = np.concatenate([
                    np.random.randn(nm, R) * 0.3,
                    np.random.randn(mp, R) * 0.3,
                    np.random.randn(pn, R) * 0.3
                ]).flatten()
            
            def residual(x):
                U_f = x[:nm*R].reshape(nm, R)
                V_f = x[nm*R:2*nm*R].reshape(mp, R)
                W_f = x[2*nm*R:3*nm*R].reshape(pn, R)
                
                res = np.zeros(729)
                for idx in range(729):
                    a = a_idx[idx]
                    b = b_idx[idx]
                    c = c_idx[idx]
                    res[idx] = np.dot(U_f[a, :], V_f[b, :] * W_f[c, :]) - targets[idx]
                return res

            try:
                # Stage 1: Least Squares
                sol = least_squares(residual, x0, method='trf', max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_x = sol.x
                    if sol.cost < 1e-12:
                        break
            except Exception:
                pass

            # Stage 2: If close, try to polish with max-error minimization
            if best_x is not None and best_cost < 1e-4:
                U_cur = best_x[:nm*R].reshape(nm, R)
                V_cur = best_x[nm*R:2*nm*R].reshape(mp, R)
                W_cur = best_x[2*nm*R:3*nm*R].reshape(pn, R)
                
                def max_err(x):
                    U_f = x[:nm*R].reshape(nm, R)
                    V_f = x[nm*R:2*nm*R].reshape(mp, R)
                    W_f = x[2*nm*R:3*nm*R].reshape(pn, R)
                    
                    # Compute all errors
                    errs = np.zeros(729)
                    for idx in range(729):
                        a = a_idx[idx]
                        b = b_idx[idx]
                        c = c_idx[idx]
                        errs[idx] = abs(np.dot(U_f[a, :], V_f[b, :] * W_f[c, :]) - targets[idx])
                    
                    # Smooth max using log-sum-exp approximation
                    # max(x) ≈ (1/alpha) * log(sum(exp(alpha*x)))
                    alpha = 100.0
                    # Clip to avoid overflow
                    errs_clipped = np.clip(errs, -100, 100)
                    max_err_val = (1.0/alpha) * np.log(np.sum(np.exp(alpha * errs_clipped)))
                    return max_err_val
                
                try:
                    sol2 = minimize(max_err, best_x, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-14})
                    if sol2.fun < 1e-9:
                        # Check if it's actually valid
                        U_sol = sol2.x[:nm*R].reshape(nm, R)
                        V_sol = sol2.x[nm*R:2*nm*R].reshape(mp, R)
                        W_sol = sol2.x[2*nm*R:3*nm*R].reshape(pn, R)
                        if verify(U_sol, V_sol, W_sol, tol=1e-9):
                            return U_sol, V_sol, W_sol
                except Exception:
                    pass

        if best_x is not None and best_cost < 1e-10:
            U_sol = best_x[:nm*R].reshape(nm, R)
            V_sol = best_x[nm*R:2*nm*R].reshape(mp, R)
            W_sol = best_x[2*nm*R:3*nm*R].reshape(pn, R)
            snapped = snap_to_half_integers(U_sol, V_sol, W_sol)
            if snapped:
                return snapped
        return None

    # Try Rank 25 with warm start from rank-26 (drop one column)
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        for R in [25, 24]:
            if time.time() > deadline - 10:
                break
            
            found = False
            for drop_col in range(min(26, R+1)):
                if time.time() > deadline - 10:
                    break
                
                U_init = np.delete(U26, drop_col, axis=1)
                V_init = np.delete(V26, drop_col, axis=1)
                W_init = np.delete(W26, drop_col, axis=1)
                
                if U_init.shape[1] < R:
                    pad_size = R - U_init.shape[1]
                    U_init = np.hstack([U_init, np.random.randn(nm, pad_size) * 0.1])
                    V_init = np.hstack([V_init, np.random.randn(mp, pad_size) * 0.1])
                    W_init = np.hstack([W_init, np.random.randn(pn, pad_size) * 0.1])
                
                res = try_optimize_rank(R, time_budget=5.0, init_U=U_init, init_V=V_init, init_W=W_init)
                if res is not None:
                    best_U, best_V, best_W = res
                    best_rank = R
                    found = True
                    break
            
            if found:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
