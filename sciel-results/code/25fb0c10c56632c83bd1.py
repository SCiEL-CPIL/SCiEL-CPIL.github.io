import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T_target of shape (nm, mp, pn)
    # T_target[i*m+j, j*p+k, k*n+i] = 1
    T_target = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T_target[a, b, d] = 1.0

    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Try ranks from 24 to 31
    for r in range(24, 32):
        if time.time() > deadline - 20.0:
            break
            
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
                
            # Initialize
            U = np.random.uniform(-1, 1, (nm, r))
            V = np.random.uniform(-1, 1, (mp, r))
            W = np.random.uniform(-1, 1, (pn, r))
            
            # Objective function
            def objective(x):
                U_ = x[:nm*r].reshape(nm, r)
                V_ = x[nm*r:nm*r+mp*r].reshape(mp, r)
                W_ = x[nm*r+mp*r:].reshape(pn, r)
                
                T_calc = np.einsum('ac,bc,dc->abd', U_, V_, W_)
                residual = T_calc - T_target
                return residual.ravel()
            
            # Gradient function
            def jac(x):
                U_ = x[:nm*r].reshape(nm, r)
                V_ = x[nm*r:nm*r+mp*r].reshape(mp, r)
                W_ = x[nm*r+mp*r:].reshape(pn, r)
                
                T_calc = np.einsum('ac,bc,dc->abd', U_, V_, W_)
                diff = T_calc - T_target
                
                # dU = 2 * diff * V * W
                # diff shape (nm, mp, pn), V shape (mp, r), W shape (pn, r)
                # Result shape (nm, r)
                dU = 2.0 * np.einsum('abd,bc,dc->ac', diff, V_, W_)
                dV = 2.0 * np.einsum('abd,ac,dc->bc', diff, U_, W_)
                dW = 2.0 * np.einsum('abd,ac,bc->dc', diff, U_, V_)
                
                return np.concatenate([dU.ravel(), dV.ravel(), dW.ravel()])
            
            x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            
            try:
                # Use Levenberg-Marquardt
                res = opt.least_squares(objective, x0, jac=jac, method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if res.success or res.cost < 1e-6:
                # Check for integer/half-integer solution
                U_sol = res.x[:nm*r].reshape(nm, r)
                V_sol = res.x[nm*r:nm*r+mp*r].reshape(mp, r)
                W_sol = res.x[nm*r+mp*r:].reshape(pn, r)
                
                # Round to nearest 0.5
                U_int = np.round(U_sol * 2) / 2
                V_int = np.round(V_sol * 2) / 2
                W_int = np.round(W_sol * 2) / 2
                
                # Verify
                T_check = np.einsum('ac,bc,dc->abd', U_int, V_int, W_int)
                if np.allclose(T_check, T_target, atol=1e-9):
                    return {"U": U_int, "V": V_int, "W": W_int}
            
    return schoolbook()
