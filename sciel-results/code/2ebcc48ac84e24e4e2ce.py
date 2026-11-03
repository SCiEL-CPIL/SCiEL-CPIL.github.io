"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _validate(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k."""
    R = len(U[0]) if U else 0
    if len(V[0]) != R or len(W[0]) != R:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS  # 3, 3, 4

    # Schoolbook fallback: rank 36
    def schoolbook():
        R = n * m * p
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Strassen-based fallback: rank 35
    def strassen_fallback():
        total_rank = 35
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        col = 0
        # Strassen 2x2x2
        strassen_U = [
            [[1,0],[0,1]],
            [[0,0],[1,1]],
            [[1,0],[0,0]],
            [[0,0],[0,1]],
            [[1,1],[0,0]],
            [[-1,0],[1,0]],
            [[0,1],[-1,0]],
        ]
        strassen_V = [
            [[1,0],[0,1]],
            [[1,0],[0,0]],
            [[0,0],[1,-1]],
            [[0,1],[-1,0]],
            [[0,0],[0,1]],
            [[1,0],[1,0]],
            [[0,1],[0,1]],
        ]
        strassen_W = [
            [[1,1],[1,1]],
            [[0,0],[1,0]],
            [[0,0],[1,0]],
            [[0,1],[0,0]],
            [[0,1],[0,0]],
            [[0,0],[1,0]],
            [[0,1],[0,0]],
        ]
        
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for r in range(7):
                        s += strassen_U[r][i][j] * strassen_V[r][j][k] * strassen_W[r][i][k]
                    if abs(s - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i*m + j][col] = strassen_U[r][i][j]
                    for k in range(2):
                        W[k*n + i][col] = strassen_W[r][i][k]
                for j in range(2):
                    for k in range(2):
                        V[j*p + k][col] = strassen_V[r][j][k]
                col += 1
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if i < 2 and j < 2 and k < 2:
                            continue
                        U[i*m + j][col] = 1
                        V[j*p + k][col] = 1
                        W[k*n + i][col] = 1
                        col += 1
            
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None

    sf = strassen_fallback()
    if sf:
        best = sf
        best_rank = 35

    # Numerical search for lower rank decompositions
    def tensor_residual(params, R, n, m, p):
        """Compute ||T - sum_r u_r ⊗ v_r ⊗ w_r||_F^2"""
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residual += (s - 1.0) ** 2
        return residual

    def tensor_residual_and_grad(params, R, n, m, p):
        """Compute residual and gradient."""
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        residual = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err = s - 1.0
                    residual += err * err
                    for r in range(R):
                        grad_U[i*m+j, r] += 2 * err * V[j*p+k, r] * W[k*n+i, r]
                        grad_V[j*p+k, r] += 2 * err * U[i*m+j, r] * W[k*n+i, r]
                        grad_W[k*n+i, r] += 2 * err * U[i*m+j, r] * V[j*p+k, r]
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return residual, grad

    # Try ranks from 34 down to 20
    for R in range(34, 19, -1):
        if time.time() - t0 > DEADLINE - 20:
            break
        
        # Multiple random restarts
        found = False
        for trial in range(3):
            if time.time() - t0 > DEADLINE - 10:
                break
            
            np.random.seed(42 + trial * 1000 + R)
            # Initialize with small random values
            params = np.random.randn(n*m*R + m*p*R + p*n*R) * 0.1
            
            # Use L-BFGS-B
            result = minimize(
                tensor_residual,
                params,
                args=(R, n, m, p),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < 1e-8:
                # Found a valid decomposition
                U_flat = result.x[:n*m*R].reshape(n*m, R)
                V_flat = result.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_flat = result.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Convert to list of lists with rounding to half-integers
                U_list = [[round(val * 2) / 2 for val in row] for row in U_flat]
                V_list = [[round(val * 2) / 2 for val in row] for row in V_flat]
                W_list = [[round(val * 2) / 2 for val in row] for row in W_flat]
                
                if _validate(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
                else:
                    # Try without rounding
                    U_list = [[float(val) for val in row] for row in U_flat]
                    V_list = [[float(val) for val in row] for row in V_flat]
                    W_list = [[float(val) for val in row] for row in W_flat]
                    if _validate(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try with Strassen-inspired initialization for this rank
        if time.time() - t0 > DEADLINE - 10:
            break
        
        # Initialize using schoolbook pattern with some structure
        np.random.seed(42 + R)
        params = np.zeros(n*m*R + m*p*R + p*n*R)
        # Set up a partial schoolbook pattern
        for r in range(min(R, n*m*p)):
            i = r // (m*p)
            j = (r // p) % m
            k = r % p
            if i < n and j < m and k < p:
                params[i*m*j + r] = 1  # U
                params[n*m*R + j*p*k + r] = 1  # V
                params[n*m*R + m*p*R + k*n*i + r] = 1  # W
        # Add small random perturbation
        params += np.random.randn(len(params)) * 0.01
        
        result = minimize(
            tensor_residual,
            params,
            args=(R, n, m, p),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.fun < 1e-8:
            U_flat = result.x[:n*m*R].reshape(n*m, R)
            V_flat = result.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W_flat = result.x[n*m*R + m*p*R:].reshape(p*n, R)
            
            U_list = [[round(val * 2) / 2 for val in row] for row in U_flat]
            V_list = [[round(val * 2) / 2 for val in row] for row in V_flat]
            W_list = [[round(val * 2) / 2 for val in row] for row in W_flat]
            
            if _validate(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
            else:
                U_list = [[float(val) for val in row] for row in U_flat]
                V_list = [[float(val) for val in row] for row in V_flat]
                W_list = [[float(val) for val in row] for row in W_flat]
                if _validate(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}

    return best

# EVOLVE-BLOCK-END
