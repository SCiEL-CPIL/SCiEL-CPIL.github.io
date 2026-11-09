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

    def verify(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def snap_to_half_int(arr, tol=1e-6):
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

    def residual_func(x, r):
        nm = n * m
        mp = m * p
        pn = p * n
        
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

    def try_rank(r, max_restarts=15):
        """Attempt to find a rank r decomposition using optimization."""
        if time.time() > deadline - 10:
            return None
            
        nm = n * m
        mp = m * p
        pn = p * n
        total_dim = nm*r + mp*r + pn*r
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Different initialization strategies
            if restart == 0:
                # Sparse random initialization
                init_vals = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=total_dim)
            elif restart == 1:
                # Dense random with small scale
                init_vals = np.random.randn(total_dim) * 0.3
            elif restart == 2:
                # Medium scale random
                init_vals = np.random.randn(total_dim) * 0.5
            elif restart == 3:
                # Structured: try to group (i,j,k) into rank components
                init_vals = np.zeros(total_dim)
                # Assign each (i,j,k) to a random rank component
                comp = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r_idx = np.random.randint(r)
                            init_vals[i*m*j + r_idx] = 1  # U
                            init_vals[nm*r + j*p*k + r_idx] = 1  # V
                            init_vals[nm*r + mp*r + k*n*i + r_idx] = 1  # W
                # This won't work directly, so fall back to random
                init_vals = np.random.randn(total_dim) * 0.4
            else:
                # Various random scales
                scale = 0.2 + 0.1 * (restart % 5)
                init_vals = np.random.randn(total_dim) * scale
            
            # Try least_squares
            try:
                result = so.least_squares(residual_func, init_vals, args=(r,), 
                                          method='lm', max_nfev=800)
                
                if result.cost < 1e-10:
                    x_opt = result.x
                    U_flat = x_opt[:nm*r]
                    V_flat = x_opt[nm*r:nm*r+mp*r]
                    W_flat = x_opt[nm*r+mp*r:]
                    
                    U = U_flat.reshape(nm, r)
                    V = V_flat.reshape(mp, r)
                    W = W_flat.reshape(pn, r)
                    
                    # Snap to half-integers
                    U_snapped = snap_to_half_int(U.tolist())
                    V_snapped = snap_to_half_int(V.tolist())
                    W_snapped = snap_to_half_int(W.tolist())
                    
                    if U_snapped and V_snapped and W_snapped:
                        if verify(U_snapped, V_snapped, W_snapped):
                            return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
                    
                    # Try with tighter tolerance for snapping
                    U_snapped2 = snap_to_half_int(U.tolist(), tol=1e-4)
                    V_snapped2 = snap_to_half_int(V.tolist(), tol=1e-4)
                    W_snapped2 = snap_to_half_int(W.tolist(), tol=1e-4)
                    
                    if U_snapped2 and V_snapped2 and W_snapped2:
                        if verify(U_snapped2, V_snapped2, W_snapped2):
                            return {"U": U_snapped2, "V": V_snapped2, "W": W_snapped2}
                            
            except Exception:
                continue
                
        return None

    # Try ranks from 35 down to 25
    for r in range(35, 24, -1):
        if time.time() > deadline - 20:
            break
        res = try_rank(r)
        if res:
            return res
            
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
