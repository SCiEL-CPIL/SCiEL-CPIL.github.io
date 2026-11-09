"""Matrix multiplication 4,4,5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    time_limit = 110  # Leave margin before 120s kill
    
    # Schoolbook decomposition as guaranteed fallback
    rank_school = n * m * p  # 80
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Precompute index lists for efficient verification
    idx_u = [i * m + j for i in range(n) for j in range(m)]
    idx_v = [j * p + k for j in range(m) for i in range(n) for k in range(p)]  # will fix below
    # Actually let's just do the verification with numpy einsum which is fast enough
    
    def verify_numpy(U, V, W, r):
        """Verify decomposition using numpy einsum. Returns max error."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # U: (n*m, r) -> (n, m, r)
        # V: (m*p, r) -> (m, p, r)
        # W: (p*n, r) -> (p, n, r) where W[k*n+i, r]
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.max(np.abs(T - 1.0))
    
    def round_and_verify(U, V, W, r, tol=1e-6):
        """Round to nearest 0.5 and verify."""
        Ur = [[round(U[i][c] * 2) / 2.0 for c in range(r)] for i in range(n * m)]
        Vr = [[round(V[i][c] * 2) / 2.0 for c in range(r)] for i in range(m * p)]
        Wr = [[round(W[i][c] * 2) / 2.0 for c in range(r)] for i in range(p * n)]
        
        err = verify_numpy(Ur, Vr, Wr, r)
        if err < tol:
            return Ur, Vr, Wr
        return None
    
    def try_rank(r, max_restarts=5, time_per_restart=None):
        """Try to find a valid decomposition of rank r via optimization."""
        if time_per_restart is None:
            remaining = time_limit - (time.time() - start_time)
            time_per_restart = max(1.0, remaining / max(max_restarts, 1))
        
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        total_size = size_u + size_v + size_w
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 2:
                break
            
            # Random initialization with small values
            np.random.seed(42 + restart * 1000 + r)
            x0 = np.random.randn(total_size) * 0.3
            
            def objective(x):
                u_flat = x[:size_u]
                v_flat = x[size_u:size_u + size_v]
                w_flat = x[size_u + size_v:]
                
                U = u_flat.reshape(n * m, r)
                V = v_flat.reshape(m * p, r)
                W = w_flat.reshape(p * n, r)
                
                U3 = U.reshape(n, m, r)
                V3 = V.reshape(m, p, r)
                W3 = W.reshape(p, n, r)
                
                T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                return np.sum((T - 1.0) ** 2)
            
            try:
                from scipy.optimize import minimize
                t0 = time.time()
                res = minimize(
                    objective, x0,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                elapsed = time.time() - t0
                
                if res.fun < 1e-4:
                    x_opt = res.x
                    u_flat = x_opt[:size_u]
                    v_flat = x_opt[size_u:size_u + size_v]
                    w_flat = x_opt[size_u + size_v:]
                    
                    U = u_flat.reshape(n * m, r).tolist()
                    V = v_flat.reshape(m * p, r).tolist()
                    W = w_flat.reshape(p * n, r).tolist()
                    
                    # First try direct verification (in case it's already clean)
                    rounded = round_and_verify(U, V, W, r)
                    if rounded:
                        return rounded
                    
                    # Try a polish step: fix one matrix at a time
                    # If U and V are fixed, W is determined linearly
                    # But that's complex. Instead, do more optimization with tighter tol.
                    res2 = minimize(
                        objective, x_opt,
                        method='L-BFGS-B',
                        options={'maxiter': 5000, 'maxfun': 50000, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res2.fun < 1e-6:
                        x_opt2 = res2.x
                        u_flat2 = x_opt2[:size_u]
                        v_flat2 = x_opt2[size_u:size_u + size_v]
                        w_flat2 = x_opt2[size_u + size_v:]
                        
                        U2 = u_flat2.reshape(n * m, r).tolist()
                        V2 = v_flat2.reshape(m * p, r).tolist()
                        W2 = w_flat2.reshape(p * n, r).tolist()
                        
                        rounded2 = round_and_verify(U2, V2, W2, r)
                        if rounded2:
                            return rounded2
            except Exception:
                pass
        
        return None
    
    # Try ranks from low to high
    # Known: 4x4x4 has rank <= 49 (Strassen), 4x4x1 has rank 16 (schoolbook)
    # So 4x4x5 should have rank <= 49+16 = 65
    # But maybe we can do better by mixing.
    # Let's try from 50 up to 79.
    
    # Actually, let's be smarter. The minimum possible rank for 4x4x5:
    # It's at least max(4*4, 4*5, 5*4) = max(16,20,20) = 20 (trivial lower bound)
    # And at most 80 (schoolbook).
    # Strassen gives 49 for 4x4x4, plus 16 for 4x4x1 = 65.
    # Let's try from 50 to 65 first, then 66-79 if time permits.
    
    ranks_to_try = list(range(50, 66)) + list(range(66, 80))
    
    for r in ranks_to_try:
        if time.time() - start_time > time_limit - 5:
            break
        
        if r >= best_rank:
            continue
        
        # Try this rank
        result = try_rank(r, max_restarts=3)
        if result:
            Ur, Vr, Wr = result
            best_rank = r
            best_result = {"U": Ur, "V": Vr, "W": Wr}
            # Found a valid decomposition; try to go lower
            # But we're iterating in increasing order, so we'll continue
            # Actually we want the MINIMUM, so we should stop at the first valid one
            # But we're going from low to high, so the first valid is the best
            # Wait, we're trying 50, 51, 52... so first valid is minimum
            # But optimization might fail for a valid rank and succeed for a higher one
            # So let's not break, but record the best
            # Actually, since we go in increasing order, break at first success
            # But give it a few restarts to be sure it's not a fluke
            break
    
    # If we didn't find anything better, return schoolbook
    return best_result
