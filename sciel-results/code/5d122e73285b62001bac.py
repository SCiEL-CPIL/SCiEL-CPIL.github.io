import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    target_rank = n * m * p  # 24
    
    def verify(U, V, W, R):
        """
        Verify if U, V, W provide an exact decomposition.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        """
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        # Efficient einsum
        approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        # Check if all entries are 1
        return np.allclose(approx, 1.0, atol=1e-9)
    
    def compute_error(U, V, W, R):
        """Compute the squared error of the decomposition."""
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        error = approx - 1.0
        return np.sum(error * error)
    
    def try_rank(R, max_restarts=30):
        """
        Attempt to find a rank R decomposition.
        """
        num_vars = R * (n * m + m * p + p * n)
        
        def unpack(params):
            U_flat = params[:R * n * m]
            V_flat = params[R * n * m : R * (n * m + m * p)]
            W_flat = params[R * (n * m + m * p):]
            
            U = U_flat.reshape(R, n * m).T  # (n*m, R)
            V = V_flat.reshape(R, m * p).T  # (m*p, R)
            W = W_flat.reshape(R, p * n).T  # (p*n, R)
            return U, V, W
        
        def objective(params):
            U, V, W = unpack(params)
            return compute_error(U, V, W, R)
        
        # Different initialization strategies
        init_strategies = []
        
        # Strategy 1: Random uniform
        for _ in range(5):
            init_strategies.append(np.random.uniform(-1, 1, size=num_vars))
        
        # Strategy 2: Small integers
        for _ in range(5):
            init_strategies.append(np.random.choice([-1, 0, 1], size=num_vars).astype(float))
        
        # Strategy 3: Structured - based on schoolbook pattern
        for _ in range(5):
            params = np.zeros(num_vars)
            # Add some structure
            for c in range(R):
                # Randomly assign some 1s
                u_idx = np.random.choice(n * m, size=min(3, n * m), replace=False)
                v_idx = np.random.choice(m * p, size=min(4, m * p), replace=False)
                w_idx = np.random.choice(p * n, size=min(3, p * n), replace=False)
                params[c * (n * m) + u_idx] = 1.0
                params[(R * n * m) + c * (m * p) + v_idx] = 1.0
                params[(R * (n * m + m * p)) + c * (p * n) + w_idx] = 1.0
            init_strategies.append(params)
        
        # Strategy 4: Gradient-based initialization from schoolbook
        if R <= target_rank:
            # Start from schoolbook and zero out some columns
            schoolbook_params = np.zeros(num_vars)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m + j
                        vidx = j * p + k
                        widx = k * n + i
                        col = i * m * p + j * p + k
                        if col < R:
                            schoolbook_params[col * (n * m) + idx] = 1.0
                            schoolbook_params[R * n * m + col * (m * p) + vidx] = 1.0
                            schoolbook_params[R * (n * m + m * p) + col * (p * n) + widx] = 1.0
            # Perturb
            init_strategies.append(schoolbook_params + np.random.normal(0, 0.1, size=num_vars))
        
        for attempt, init_params in enumerate(init_strategies):
            if time.time() - start_time > max_time:
                return None
            
            # Try Nelder-Mead
            try:
                res = minimize(objective, init_params, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-14, 'disp': False})
                
                if res.fun < 1e-8:
                    U, V, W = unpack(res.x)
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(U * 2) / 2.0
                    V_rounded = np.round(V * 2) / 2.0
                    W_rounded = np.round(W * 2) / 2.0
                    
                    if verify(U_rounded, V_rounded, W_rounded, R):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    
                    # Try integer rounding
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if verify(U_int, V_int, W_int, R):
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        
            except Exception:
                pass
            
            # Try L-BFGS-B for some attempts
            if attempt % 3 == 0:
                try:
                    res = minimize(objective, init_params, method='L-BFGS-B',
                                   options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
                    
                    if res.fun < 1e-8:
                        U, V, W = unpack(res.x)
                        
                        # Try rounding
                        U_rounded = np.round(U * 2) / 2.0
                        V_rounded = np.round(V * 2) / 2.0
                        W_rounded = np.round(W * 2) / 2.0
                        
                        if verify(U_rounded, V_rounded, W_rounded, R):
                            return {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
                            
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        if verify(U_int, V_int, W_int, R):
                            return {
                                "U": U_int.tolist(),
                                "V": V_int.tolist(),
                                "W": W_int.tolist()
                            }
                            
                except Exception:
                    pass
        
        return None
    
    # Search for lower ranks, starting from 23 down to 20
    for R in range(23, 19, -1):
        if time.time() - start_time > max_time - 10:
            break
        
        result = try_rank(R, max_restarts=15)
        if result:
            return result
    
    # Fallback: Schoolbook decomposition
    rank = target_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
