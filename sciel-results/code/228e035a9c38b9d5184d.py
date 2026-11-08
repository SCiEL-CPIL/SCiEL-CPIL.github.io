import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=3
    # U: (9, R), V: (9, R), W: (9, R)
    # Indices:
    # U[i*m+j, r]
    # V[j*p+k, r]
    # W[k*n+i, r]
    
    # Target:
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = delta_{j, j'}
    # for all i, k, j, j' in {0, 1, 2}
    
    # Total equations: 3*3*3*3 = 81
    # Variables: 27 * R
    
    # Try ranks from 23 down to 19
    # Note: Rank 27 is always possible.
    
    start_time = time.time()
    deadline = start_time + 100 # Leave 20s margin
    
    # Precompute indices for the equations
    # eq_index maps (i, k, j, jp) to equation number
    # We need to map (i, k, j, jp) -> index in the flattened residual vector
    # Residual r_{ikjj'} = sum_r W[k*n+i, r] U[i*m+j, r] V[jp*p+k, r] - delta_{j, jp}
    
    # Let's create a list of tuples (i, k, j, jp, eq_idx)
    eqs = []
    eq_idx = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    eqs.append((i, k, j, jp, eq_idx))
                    eq_idx += 1
                    
    num_eqs = len(eqs)
    
    def residuals(x, R):
        # x is flattened U, V, W
        # U: x[0 : 9*R]
        # V: x[9*R : 18*R]
        # W: x[18*R : 27*R]
        U = x[0:9*R].reshape(9, R)
        V = x[9*R:18*R].reshape(9, R)
        W = x[18*R:27*R].reshape(9, R)
        
        res = np.zeros(num_eqs)
        for (i, k, j, jp, idx) in eqs:
            # sum_r W[k*n+i, r] * U[i*m+j, r] * V[jp*p+k, r]
            # W index: k*n + i
            # U index: i*m + j
            # V index: jp*p + k
            w_idx = k * n + i
            u_idx = i * m + j
            v_idx = jp * p + k
            
            # Vectorized sum over r
            # res[idx] = np.sum(W[w_idx, :] * U[u_idx, :] * V[v_idx, :])
            # To speed up, we can precompute or use einsum?
            # But R is small, loop is fine.
            res[idx] = np.dot(W[w_idx, :], U[u_idx, :] * V[v_idx, :])
            if j == jp:
                res[idx] -= 1.0
            else:
                res[idx] -= 0.0
        return res

    best_solution = None
    best_rank = 27 # Fallback
    
    # Try to find a solution for R < 27
    for R in range(23, 18, -1):
        if time.time() > deadline:
            break
            
        # Multiple restarts for robustness
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale matters. Let's try uniform [-1, 1]
            x0 = np.random.uniform(-1, 1, size=27 * R)
            
            try:
                result = least_squares(residuals, x0, args=(R,), method='lm', max_nfev=1000)
                if result.cost < 1e-10:
                    # Verify solution
                    x_opt = result.x
                    U = x_opt[0:9*R].reshape(9, R)
                    V = x_opt[9*R:18*R].reshape(9, R)
                    W = x_opt[18*R:27*R].reshape(9, R)
                    
                    # Double check with high precision logic if possible, or just accept if cost is very low
                    # The task requires "integer/half-integer entries". 
                    # Numerical solutions might not be exact integers/half-integers.
                    # However, if the system has a rational solution, least_squares might find it approximately.
                    # We need to round to nearest half-integer and verify.
                    
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    max_err = 0
                    for (i, k, j, jp, idx) in eqs:
                        w_idx = k * n + i
                        u_idx = i * m + j
                        v_idx = jp * p + k
                        val = np.dot(W_rounded[w_idx, :], U_rounded[u_idx, :] * V_rounded[v_idx, :])
                        target = 1.0 if j == jp else 0.0
                        err = abs(val - target)
                        if err > max_err:
                            max_err = err
                    
                    if max_err < 1e-6:
                        # Found a valid solution with half-integer entries
                        best_solution = {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        best_rank = R
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
            
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to schoolbook rank 27
        rank = 27
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
