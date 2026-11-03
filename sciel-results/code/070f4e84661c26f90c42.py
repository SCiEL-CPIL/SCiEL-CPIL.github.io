import numpy as np
import time
import scipy.optimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Target tensor for 3x3x3
    T_333 = np.ones((3, 3, 3))
    
    best_decomp = None
    best_rank = 27
    
    # Try to find a decomposition for 3x3x3 with rank R
    for R in range(26, 22, -1): # 26, 25, 24, 23
        if time.time() - start_time > 100:
            break
            
        # We will try a few random restarts
        found = False
        for restart in range(50):
            if time.time() - start_time > 100:
                break
                
            # Random initialization
            U = np.random.randn(3, 3, R)
            V = np.random.randn(3, 3, R)
            W = np.random.randn(3, 3, R)
            
            # Scale
            U = U * 0.5
            V = V * 0.5
            W = W * 0.5
            
            x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_r = x[:27*R].reshape(3, 3, R)
                V_r = x[27*R:54*R].reshape(3, 3, R)
                W_r = x[54*R:].reshape(3, 3, R)
                prod = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                error = prod - T_333
                return np.sum(error**2)
            
            def jac(x):
                # Numerical gradient
                eps = 1e-6
                grad = np.zeros_like(x)
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    grad[i] = (objective(x_plus) - objective(x)) / eps
                return grad
                
            # Use L-BFGS-B
            try:
                res = scipy.optimize.minimize(objective, x, jac=jac, method='L-BFGS-B', options={'maxiter': 200})
                if res.fun < 1e-6:
                    U_r = res.x[:27*R].reshape(3, 3, R)
                    V_r = res.x[27*R:54*R].reshape(3, 3, R)
                    W_r = res.x[54*R:].reshape(3, 3, R)
                    
                    # Round to nearest 0.5
                    U_int = np.round(U_r * 2) / 2
                    V_int = np.round(V_r * 2) / 2
                    W_int = np.round(W_r * 2) / 2
                    
                    # Verify
                    prod = np.einsum('ijr,jkr,kir->ijk', U_int, V_int, W_int)
                    if np.allclose(prod, T_333, atol=1e-9):
                        best_decomp = (U_int, V_int, W_int)
                        best_rank = R
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
            
    # If not found, use schoolbook
    if best_decomp is None:
        R_333 = 27
        U_333 = np.zeros((3, 3, R_333))
        V_333 = np.zeros((3, 3, R_333))
        W_333 = np.zeros((3, 3, R_333))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_333[i, j, col] = 1
                    V_333[j, k, col] = 1
                    W_333[k, i, col] = 1
                    col += 1
        best_decomp = (U_333, V_333, W_333)
        best_rank = 27

    # Construct full 3x3x4
    R_333 = best_rank
    U_333, V_333, W_333 = best_decomp
    
    R_331 = 9
    U_331 = np.zeros((3, 3, R_331))
    V_331 = np.zeros((3, 1, R_331))
    W_331 = np.zeros((1, 3, R_331))
    col = 0
    for i in range(3):
        for j in range(3):
            U_331[i, j, col] = 1
            V_331[j, 0, col] = 1
            W_331[0, i, col] = 1
            col += 1
            
    R_total = R_333 + R_331
    
    U_full = np.zeros((9, R_total))
    V_full = np.zeros((12, R_total))
    W_full = np.zeros((12, R_total))
    
    for i in range(3):
        for j in range(3):
            for r in range(R_333):
                U_full[i*3+j, r] = U_333[i, j, r]
                
    for j in range(3):
        for k in range(3):
            for r in range(R_333):
                V_full[j*4+k, r] = V_333[j, k, r]
                
    for k in range(3):
        for i in range(3):
            for r in range(R_333):
                W_full[k*3+i, r] = W_333[k, i, r]
                
    offset = R_333
    for i in range(3):
        for j in range(3):
            for r in range(R_331):
                U_full[i*3+j, offset+r] = U_331[i, j, r]
                
    for j in range(3):
        for r in range(R_331):
            V_full[j*4+3, offset+r] = V_331[j, 0, r]
            
    for i in range(3):
        for r in range(R_331):
            W_full[3*3+i, offset+r] = W_331[0, i, r]
            
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
