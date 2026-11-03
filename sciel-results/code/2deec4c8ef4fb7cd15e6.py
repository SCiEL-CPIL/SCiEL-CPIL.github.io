import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s for safety

    def _verify(U, V, W):
        rank = U.shape[1]
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # Condition: sum_c W[k*n+i, c] U[i*m+j, c] V[jp*p+k, c] == delta(j, jp)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(rank):
                            s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                        target = 1.0 if j == jp else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True

    def _schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        return U_snapped, V_snapped, W_snapped

    def _residuals(params, rank):
        # params is a flat array of size rank * (nm + mp + pn)
        size_U = nm * rank
        size_V = mp * rank
        size_W = pn * rank
        
        U = params[:size_U].reshape((nm, rank))
        V = params[size_U:size_U+size_V].reshape((mp, rank))
        W = params[size_U+size_V:size_U+size_V+size_W].reshape((pn, rank))
        
        res = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(rank):
                            s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                        target = 1.0 if j == jp else 0.0
                        res.append(s - target)
        return np.array(res)

    def _solve_rank(rank, max_restarts=50):
        total_vars = rank * (nm + mp + pn)
        best_err = float('inf')
        best_params = None
        
        for r in range(max_restarts):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            # Scale initialization can help
            scale = 1.0
            params = np.random.randn(total_vars) * scale
            
            # Use least_squares
            try:
                res = opt.least_squares(_residuals, params, args=(rank,), method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if res.cost < best_err:
                best_err = res.cost
                best_params = res.x
            
            # Check for exact solution after snapping
            if best_err < 1e-8:
                size_U = nm * rank
                size_V = mp * rank
                size_W = pn * rank
                U = best_params[:size_U].reshape((nm, rank))
                V = best_params[size_U:size_U+size_V].reshape((mp, rank))
                W = best_params[size_U+size_V:size_U+size_V+size_W].reshape((pn, rank))
                
                U_snapped, V_snapped, W_snapped = _snap_to_half_integers(U, V, W)
                if _verify(U_snapped, V_snapped, W_snapped):
                    return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                    
        # Final check on best_params if it wasn't caught in loop
        if best_params is not None:
            size_U = nm * rank
            size_V = mp * rank
            size_W = pn * rank
            U = best_params[:size_U].reshape((nm, rank))
            V = best_params[size_U:size_U+size_V].reshape((mp, rank))
            W = best_params[size_U+size_V:size_U+size_V+size_W].reshape((pn, rank))
            U_snapped, V_snapped, W_snapped = _snap_to_half_integers(U, V, W)
            if _verify(U_snapped, V_snapped, W_snapped):
                return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                
        return None

    # Try ranks from low to high
    # Known lower bound is 12. Schoolbook is 24.
    # Try 17 first as it's the known minimum for this shape.
    for rank in [17, 18, 19, 20, 21, 22, 23]:
        if time.time() > deadline - 5:
            break
        res = _solve_rank(rank)
        if res is not None:
            return res

    return _schoolbook()
