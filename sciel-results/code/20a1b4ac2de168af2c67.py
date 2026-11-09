import time
import numpy as np
import scipy.optimize as so

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1) > tol:
                        return False
        return True

    def snap_to_half_int(arr, tol=1e-4):
        """Snap values to nearest half-integer if close, else return None."""
        snapped = []
        for row in arr:
            new_row = []
            for x in row:
                closest = round(x * 2) / 2
                if abs(x - closest) < tol:
                    new_row.append(closest)
                else:
                    return None
            snapped.append(new_row)
        return snapped

    def try_rank(r, max_restarts=15):
        """Attempt to find a rank r decomposition using optimization."""
        if time.time() > deadline - 10:
            return None
            
        nm = n * m
        mp = m * p
        pn = p * n
        total_size = nm * r + mp * r + pn * r
        
        def residual(x):
            U_flat = x[:nm*r]
            V_flat = x[nm*r:nm*r+mp*r]
            W_flat = x[nm*r+mp*r:]
            
            U = U_flat.reshape(nm, r)
            V = V_flat.reshape(mp, r)
            W = W_flat.reshape(pn, r)
            
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        res[idx] = val - 1
                        idx += 1
            return res

        def objective(x):
            res = residual(x)
            return np.sum(res**2)

        best_result = None
        best_cost = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Different initialization strategies
            if restart == 0:
                # Sparse random initialization
                init_vals = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=total_size)
            elif restart == 1:
                # Small random
                init_vals = np.random.randn(total_size) * 0.5
            elif restart == 2:
                # Medium random
                init_vals = np.random.randn(total_size) * 1.0
            elif restart == 3:
                # Structured: try to create some patterns
                init_vals = np.zeros(total_size)
                # Randomly set some entries
                for _ in range(total_size // 3):
                    idx = np.random.randint(0, total_size)
                    init_vals[idx] = np.random.choice([-1.0, 1.0])
            else:
                # Scale varies
                scale = 0.3 + 0.2 * (restart % 5)
                init_vals = np.random.randn(total_size) * scale
            
            try:
                # Try LM method
                result = so.least_squares(residual, init_vals, method='lm', max_nfev=1000)
                
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result.x
                    
                if result.cost < 1e-10:
                    x_opt = result.x
                    U = x_opt[:nm*r].reshape(nm, r)
                    V = x_opt[nm*r:nm*r+mp*r].reshape(mp, r)
                    W = x_opt[nm*r+mp*r:].reshape(pn, r)
                    
                    # Try to snap to half-integers
                    U_snapped = snap_to_half_int(U.tolist())
                    V_snapped = snap_to_half_int(V.tolist())
                    W_snapped = snap_to_half_int(W.tolist())
                    
                    if U_snapped and V_snapped and W_snapped:
                        if verify(U_snapped, V_snapped, W_snapped):
                            return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
                    
                    # If not close to half-integers, try a final polish
                    # Use the current solution as starting point for a tighter optimization
                    try:
                        polish = so.least_squares(residual, x_opt, method='lm', max_nfev=2000, ftol=1e-15, xtol=1e-15)
                        if polish.cost < 1e-12:
                            x_polish = polish.x
                            U_p = x_polish[:nm*r].reshape(nm, r)
                            V_p = x_polish[nm*r:nm*r+mp*r].reshape(mp, r)
                            W_p = x_polish[nm*r+mp*r:].reshape(pn, r)
                            
                            U_p_snapped = snap_to_half_int(U_p.tolist())
                            V_p_snapped = snap_to_half_int(V_p.tolist())
                            W_p_snapped = snap_to_half_int(W_p.tolist())
                            
                            if U_p_snapped and V_p_snapped and W_p_snapped:
                                if verify(U_p_snapped, V_p_snapped, W_p_snapped):
                                    return {"U": U_p_snapped, "V": V_p_snapped, "W": W_p_snapped}
                    except Exception:
                        pass
                        
            except Exception:
                continue
                
        return None

    # Try ranks from 34 down to 25
    # Schoolbook is 36. Known results suggest 32 or lower might work.
    for r in range(34, 24, -1):
        if time.time() > deadline - 20:
            break
        res = try_rank(r)
        if res:
            return res
            
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
