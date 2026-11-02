"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 120  # Safety margin
    
    # Strassen coefficients for M_{2,2,2}
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Schoolbook coefficients for M_{2,2,1}
    U_school = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    V_school = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    W_school = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    def build_rank53():
        """Build rank-53 solution: 7 Strassen blocks + 1 schoolbook."""
        rank = 7 * 7 + 4
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col_offset = 0
        for block in range(7):
            k0 = 2 * block
            k1 = 2 * block + 1
            for r in range(7):
                c = col_offset + r
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, c] = U_strassen[i * m + j, r]
                for j in range(m):
                    V[j * p + k0, c] = V_strassen[j * 2 + 0, r]
                    V[j * p + k1, c] = V_strassen[j * 2 + 1, r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(n):
                        W[kk * n + i, c] = W_strassen[k * n + i, r]
            col_offset += 7
        
        k_last = 14
        for r in range(4):
            c = col_offset + r
            for i in range(n):
                for j in range(m):
                    U[i * m + j, c] = U_school[i * m + j, r]
            for j in range(m):
                V[j * p + k_last, c] = V_school[j, r]
            for i in range(n):
                W[k_last * n + i, c] = W_school[i, r]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def build_rank52(U_223, V_223, W_223):
        """Build rank-52 solution: 6 Strassen blocks + M_{2,2,3} rank 10."""
        rank = 6 * 7 + 10
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # 6 Strassen blocks for k=0..11
        for block in range(6):
            k0 = 2 * block
            k1 = 2 * block + 1
            for r in range(7):
                c = block * 7 + r
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, c] = U_strassen[i * m + j, r]
                for j in range(m):
                    V[j * p + k0, c] = V_strassen[j * 2 + 0, r]
                    V[j * p + k1, c] = V_strassen[j * 2 + 1, r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(n):
                        W[kk * n + i, c] = W_strassen[k * n + i, r]
        
        # M_{2,2,3} for k=12,13,14
        U[:, 42:] = U_223
        V[:, 42:] = V_223
        W[:, 42:] = W_223
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_223(U, V, W, tol=1e-6):
        """Verify M_{2,2,3} decomposition."""
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    val = np.dot(np.dot(U[i*2+j, :], V[j*3+k, :]), W[k*2+i, :])
                    target = 1.0 if i == k else 0.0
                    if abs(val - target) > tol:
                        return False
        return True
    
    def try_m223_rank10():
        """Try to find rank-10 decomposition for M_{2,2,3}."""
        if time.time() > deadline - 20:
            return None
        
        from scipy.optimize import least_squares
        
        def residuals(U, V, W):
            res = np.zeros(12)
            idx = 0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        val = np.dot(np.dot(U[i*2+j, :], V[j*3+k, :]), W[k*2+i, :])
                        target = 1.0 if i == k else 0.0
                        res[idx] = val - target
                        idx += 1
            return res
        
        def flatten(x):
            return (x[0:40].reshape(4, 10), 
                    x[40:100].reshape(6, 10), 
                    x[100:160].reshape(6, 10))
        
        def objective(x):
            U, V, W = flatten(x)
            return residuals(U, V, W)
        
        np.random.seed(123)
        best_sol = None
        best_res = np.inf
        
        # Multiple initialization strategies
        inits = []
        
        # Strategy 1: Small random
        for s in [0.1, 0.5, 1.0]:
            U = np.random.randn(4, 10) * s
            V = np.random.randn(6, 10) * s
            W = np.random.randn(6, 10) * s
            inits.append((U, V, W))
        
        # Strategy 2: Structured - start with partial schoolbook
        U = np.zeros((4, 10))
        V = np.zeros((6, 10))
        W = np.zeros((6, 10))
        # Schoolbook for first 4 terms
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1  # (i=0,j=0,k=0)
        U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1  # (i=0,j=1,k=0)
        U[2, 2] = 1; V[1, 2] = 1; W[2, 2] = 1  # (i=1,j=0,k=1)
        U[3, 3] = 1; V[4, 3] = 1; W[3, 3] = 1  # (i=1,j=1,k=1)
        # Perturb remaining
        U[:, 4:] += np.random.randn(4, 6) * 0.3
        V[:, 4:] += np.random.randn(6, 6) * 0.3
        W[:, 4:] += np.random.randn(6, 6) * 0.3
        inits.append((U, V, W))
        
        # Strategy 3: Strassen-inspired for 2x2x2 + extra
        U = np.zeros((4, 10))
        V = np.zeros((6, 10))
        W = np.zeros((6, 10))
        # Use Strassen for first 7 terms (columns 0,1)
        for r in range(7):
            U[:, r] = U_strassen[:, r]
            V[0:2, r] = V_strassen[0:2, r]  # k=0
            V[2:4, r] = V_strassen[2:4, r]  # k=1
            W[0:2, r] = W_strassen[0:2, r]  # k=0
            W[2:4, r] = W_strassen[2:4, r]  # k=1
            W[4:6, r] = 0  # k=2 not handled by Strassen
        # Add random for last 3 terms
        U[:, 7:] += np.random.randn(4, 3) * 0.5
        V[:, 7:] += np.random.randn(6, 3) * 0.5
        W[:, 7:] += np.random.randn(6, 3) * 0.5
        inits.append((U, V, W))
        
        # Strategy 4: Another structured init
        U = np.zeros((4, 10))
        V = np.zeros((6, 10))
        W = np.zeros((6, 10))
        # Diagonal-like terms
        for r in range(4):
            U[r, r] = 1
            V[r, r] = 1
            W[r, r] = 1
        U[:, 4:] += np.random.randn(4, 6) * 0.5
        V[:, 4:] += np.random.randn(6, 6) * 0.5
        W[:, 4:] += np.random.randn(6, 6) * 0.5
        inits.append((U, V, W))
        
        for U0, V0, W0 in inits:
            if time.time() > deadline - 5:
                break
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = least_squares(
                    objective, x0, method='lm', max_nfev=3000,
                    ftol=1e-14, xtol=1e-14, gtol=1e-14
                )
                
                U_opt, V_opt, W_opt = flatten(result.x)
                res = residuals(U_opt, V_opt, W_opt)
                max_res = np.max(np.abs(res))
                
                if max_res < best_res:
                    best_res = max_res
                    best_sol = (U_opt.copy(), V_opt.copy(), W_opt.copy())
                
                if max_res < 1e-8:
                    # Try rounding
                    for scale in [1, 2]:
                        U_r = np.round(U_opt * scale) / scale
                        V_r = np.round(V_opt * scale) / scale
                        W_r = np.round(W_opt * scale) / scale
                        res_r = residuals(U_r, V_r, W_r)
                        if np.max(np.abs(res_r)) < 1e-6:
                            return U_r, V_r, W_r
                    
                    if max_res < 1e-10:
                        return U_opt, V_opt, W_opt
                        
            except Exception:
                continue
        
        if best_sol is not None and best_res < 1e-6:
            return best_sol
        
        return None
    
    # Try to find rank-52 solution
    result = try_m223_rank10()
    
    if result is not None:
        U_223, V_223, W_223 = result
        if verify_223(U_223, V_223, W_223):
            return build_rank52(U_223, V_223, W_223)
    
    # Fallback to rank 53
    return build_rank53()
# EVOLVE-BLOCK-END
