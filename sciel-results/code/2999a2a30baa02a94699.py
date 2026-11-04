import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Schoolbook decomposition
    rank_school = n * m * p  # 45
    U_fb = np.zeros((n*m, rank_school), dtype=np.float64)
    V_fb = np.zeros((m*p, rank_school), dtype=np.float64)
    W_fb = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1

    target = np.ones((n, m, p), dtype=np.float64)

    def verify(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(result, target, atol=1e-9, rtol=1e-9)

    def error_func(params, R):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = params[:size_u*R].reshape(size_u, R)
        V = params[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
        W = params[size_u*R + size_v*R :].reshape(size_w, R)
        
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        diff = result - target
        return np.sum(diff**2)

    def solve_rank_lbfgs(R, max_restarts=30):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = (size_u + size_v + size_w) * R
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Sparse random initialization
            params0 = np.zeros(total_size)
            for idx in range(total_size):
                if np.random.random() < 0.3:
                    params0[idx] = np.random.choice([-1.0, 0.5, 1.0, -0.5])
            
            try:
                res = minimize(error_func, params0, args=(R,), method='L-BFGS-B', 
                               options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-14})
                
                if res.fun < 1e-10:
                    U = res.x[:size_u*R].reshape(size_u, R)
                    V = res.x[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
                    W = res.x[size_u*R + size_v*R :].reshape(size_w, R)
                    
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if verify(U_snap, V_snap, W_snap):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
            except Exception:
                pass
                
        return None

    def solve_rank_structured(R, max_restarts=20):
        """Try structured initializations based on known patterns"""
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = (size_u + size_v + size_w) * R
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Different initialization strategies
            strategy = restart % 3
            
            if strategy == 0:
                # Binary initialization
                params0 = np.random.choice([-1.0, 1.0], size=total_size)
            elif strategy == 1:
                # Ternary with more zeros
                params0 = np.random.choice([-1.0, 0.0, 1.0], size=total_size, p=[0.25, 0.5, 0.25])
            else:
                # Small values
                params0 = np.random.uniform(-1, 1, size=total_size)
                params0[np.abs(params0) < 0.3] = 0.0
            
            try:
                res = minimize(error_func, params0, args=(R,), method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14})
                
                if res.fun < 1e-10:
                    U = res.x[:size_u*R].reshape(size_u, R)
                    V = res.x[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
                    W = res.x[size_u*R + size_v*R :].reshape(size_w, R)
                    
                    # Try different snapping strategies
                    for snap_mult in [2, 1, 0.5]:
                        U_snap = np.round(U * snap_mult) / snap_mult
                        V_snap = np.round(V * snap_mult) / snap_mult
                        W_snap = np.round(W * snap_mult) / snap_mult
                        
                        if verify(U_snap, V_snap, W_snap):
                            return {
                                "U": U_snap.tolist(),
                                "V": V_snap.tolist(),
                                "W": W_snap.tolist()
                            }
            except Exception:
                pass
                
        return None

    # Try to find lower ranks
    # Known lower bound for 3x3x5 is around 27-30
    # Start from reasonable ranks and work up
    ranks_to_try = list(range(28, 45))
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # Try LBFGS approach
        result = solve_rank_lbfgs(R, max_restarts=10)
        if result is not None:
            return result
        
        if time.time() > deadline - 10:
            break
        
        # Try structured approach
        result = solve_rank_structured(R, max_restarts=10)
        if result is not None:
            return result
    
    # Return schoolbook
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
