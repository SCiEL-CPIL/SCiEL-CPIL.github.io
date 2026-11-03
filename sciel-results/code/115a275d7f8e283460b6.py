import numpy as np
from scipy.optimize import minimize
import time

def find_rank5_223(timeout=60):
    R = 5
    n, m, p = 2, 2, 3
    start = time.time()
    
    def loss(params):
        U = params[0:4*R].reshape(4, R)
        V = params[4*R:4*R+6*R].reshape(6, R)
        W = params[4*R+6*R:4*R+6*R+6*R].reshape(6, R)
        
        # Use a fixed set of test matrices
        A0 = np.array([[1, 2], [3, 4]], dtype=float)
        B0 = np.array([[5, 6, 7], [8, 9, 10]], dtype=float)
        C0 = A0 @ B0
        A1 = np.array([[1, -1], [0, 1]], dtype=float)
        B1 = np.array([[1, 0, 1], [-1, 1, 0]], dtype=float)
        C1 = A1 @ B1
        A2 = np.array([[0.5, 1.5], [2.5, -0.5]], dtype=float)
        B2 = np.array([[1, 2, -1], [3, -2, 4]], dtype=float)
        C2 = A2 @ B2
        
        total = 0
        for A, B, C in [(A0, B0, C0), (A1, B1, C1), (A2, B2, C2)]:
            C_recon = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        for r in range(R):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    C_recon[i, k] = s
            total += np.sum((C_recon - C)**2)
        return total
    
    best_loss = float('inf')
    best_params = None
    trial = 0
    while time.time() - start < timeout - 5:
        np.random.seed(100000 + trial)
        U_init = np.random.randn(4, R)
        V_init = np.random.randn(6, R)
        W_init = np.random.randn(6, R)
        for r in range(R):
            nU = np.linalg.norm(U_init[:, r])
            nV = np.linalg.norm(V_init[:, r])
            if nU > 1e-10 and nV > 1e-10:
                U_init[:, r] /= nU
                V_init[:, r] *= nU
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        res = minimize(loss, params_init, method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-16})
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
            if best_loss < 1e-12:
                break
        trial += 1
    
    U = best_params[0:4*R].reshape(4, R)
    V = best_params[4*R:4*R+6*R].reshape(6, R)
    W = best_params[4*R+6*R:4*R+6*R+6*R].reshape(6, R)
    
    # Verify with random matrices
    np.random.seed(42)
    A = np.random.randn(2, 2)
    B = np.random.randn(2, 3)
    C = A @ B
    C_recon = np.zeros((2, 3))
    for i in range(2):
        for k in range(3):
            s = 0
            for j in range(2):
                for r in range(R):
                    s += U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
            C_recon[i, k] = s
    err = np.max(np.abs(C_recon - C))
    return U, V, W, err, best_loss
