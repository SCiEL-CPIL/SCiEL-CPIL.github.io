import scipy
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135

    # Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    def verify(U, V, W, tol=1e-8):
        # Use einsum for speed
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # We need sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
        # Let's check all (i,j,k)
        # This is hard to vectorize directly into a single matrix compare without building the full tensor.
        # But we can loop over i, j and use dot products.
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_vec = U[i*m+j, :] # (r,)
                for k in range(p):
                    v_vec = V[j*p+k, :] # (r,)
                    w_vec = W[k*n+i, :] # (r,)
                    val = np.dot(u_vec, v_vec * w_vec)
                    max_err = max(max_err, abs(val - 1))
        return max_err

    def make_residuals(r):
        def residuals(x):
            U = x[:r*n*m].reshape(n*m, r)
            V = x[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
            W = x[r*n*m + r*m*p:].reshape(p*n, r)
            
            res = []
            for i in range(n):
                for j in range(m):
                    u_vec = U[i*m+j, :]
                    for k in range(p):
                        v_vec = V[j*p+k, :]
                        w_vec = W[k*n+i, :]
                        val = np.dot(u_vec, v_vec * w_vec)
                        res.append(val - 1)
            return np.array(res)
        return residuals

    # Try ranks from 30 up to schoolbook
    # We know 84 works. Let's try to find something lower.
    # Lower bound is max(nm, mp, pn) = 28.
    
    for r in range(30, rank_school + 1):
        if time.time() > deadline:
            break
        
        n_vars = r * (n*m + m*p + p*n)
        if n_vars < n * m * p:
            continue
            
        found = False
        # More attempts for lower ranks
        n_attempts = 20 if r < 60 else 10
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            # Initialize
            if attempt == 0:
                x0 = np.random.randn(n_vars)
            elif attempt == 1:
                # Sparse
                x0 = np.zeros(n_vars)
                idx = np.random.choice(n_vars, min(n_vars, 100), replace=False)
                x0[idx] = np.random.choice([-1, 1], len(idx))
            else:
                x0 = np.random.randn(n_vars) * 0.5

            try:
                residuals = make_residuals(r)
                result = scipy.optimize.least_squares(residuals, x0, method='lm', max_nfev=5000, xtol=1e-10, ftol=1e-10)
                
                if result.cost < 1e-8:
                    sol = result.x
                    U_sol = sol[:r*n*m].reshape(n*m, r)
                    V_sol = sol[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
                    W_sol = sol[r*n*m + r*m*p:].reshape(p*n, r)
                    
                    err = verify(U_sol, V_sol, W_sol)
                    if err < 1e-6:
                        best_rank = r
                        best_U, best_V, best_W = U_sol.copy(), V_sol.copy(), W_sol.copy()
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    # Round to half-integers
    def to_half_integer(arr):
        return np.round(arr * 2) / 2

    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)

    max_err = verify(U_out, V_out, W_out)
    if max_err > 1e-5:
        # Try integer rounding
        U_out_int = np.round(best_U)
        V_out_int = np.round(best_V)
        W_out_int = np.round(best_W)
        max_err_int = verify(U_out_int, V_out_int, W_out_int)
        if max_err_int < 1e-5:
            U_out, V_out, W_out = U_out_int, V_out_int, W_out_int
        else:
            # If rounding fails, we must return the float solution?
            # The task says "integer/half-integer entries".
            # If we can't round, the solution is invalid?
            # "score is the rank" ... "invalid result ... scores 0".
            # So we MUST return integer/half-integer.
            # If the float solution doesn't round to a valid one, we should probably fall back to schoolbook (which is integer).
            # Or try to find a solution that *is* integer/half-integer.
            # For now, if rounding fails, use schoolbook.
            U_out, V_out, W_out = U_school, V_school, W_school

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
