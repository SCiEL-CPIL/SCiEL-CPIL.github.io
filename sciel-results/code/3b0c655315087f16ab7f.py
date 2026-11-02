import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Leave 10s margin
    
    # Target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0

    def check(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        R = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(R, T, atol=1e-9)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    
    # Try to find rank 22
    # We will use a random restart strategy with gradient descent.
    # Variables: flatten U, V, W.
    # Objective: || T - sum(U_c V_c W_c) ||^2
    
    def objective(params, R):
        if time.time() > deadline:
            return float('inf')
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R : n*m*R + m*p*R + p*n*R].reshape(p*n, R)
        R_tensor = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.sum((R_tensor - T)**2)
        return err

    found = False
    for rank in [22, 23]:
        if time.time() > deadline:
            break
        for restart in range(50):
            if time.time() > deadline:
                break
            np.random.seed(42 + restart)
            # Initialize with small random values
            U0 = np.random.randn(n*m, rank) * 0.1
            V0 = np.random.randn(m*p, rank) * 0.1
            W0 = np.random.randn(p*n, rank) * 0.1
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Use L-BFGS-B
            try:
                res = minimize(objective, params0, args=(rank,), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-8})
                if res.fun < 1e-8:
                    U = res.x[:n*m*rank].reshape(n*m, rank)
                    V = res.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                    W = res.x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank].reshape(p*n, rank)
                    if check(U, V, W):
                        best = {"U": U, "V": V, "W": W}
                        found = True
                        break
            except Exception:
                pass
        if found:
            break

    return best
