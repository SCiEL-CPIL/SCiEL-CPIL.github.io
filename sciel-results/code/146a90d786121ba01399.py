import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 3, 3, 6
    
    # Target: Rank 46 for 3x3x6
    # Strategy: Find Rank 23 for 3x3x3, then duplicate.
    
    # Verification for 3x3x3
    def verify_333(U, V, W):
        # U, V, W are 9xR matrices
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = np.sum(U[i*3+j] * V[j*3+k] * W[k*3+i])
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Try to find rank 23 decomposition for 3x3x3
    rank_333 = 23
    dim = 9
    total_dim = 3 * dim * rank_333
    
    def objective(x):
        U = x[:dim*rank_333].reshape(dim, rank_333)
        V = x[dim*rank_333:2*dim*rank_333].reshape(dim, rank_333)
        W = x[2*dim*rank_333:].reshape(dim, rank_333)
        
        error = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = np.sum(U[i*3+j] * V[j*3+k] * W[k*3+i])
                    error += (s - 1.0)**2
        return error

    def gradient(x):
        U = x[:dim*rank_333].reshape(dim, rank_333)
        V = x[dim*rank_333:2*dim*rank_333].reshape(dim, rank_333)
        W = x[2*dim*rank_333:].reshape(dim, rank_333)
        
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = np.sum(U[i*3+j] * V[j*3+k] * W[k*3+i])
                    err = s - 1.0
                    gU[i*3+j] += err * V[j*3+k] * W[k*3+i]
                    gV[j*3+k] += err * U[i*3+j] * W[k*3+i]
                    gW[k*3+i] += err * U[i*3+j] * V[j*3+k]
        
        return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])

    best_333 = None
    start_time = time.time()
    
    # Try a few random seeds
    for seed in [42, 123, 999, 1234]:
        if time.time() - start_time > 100: break
        np.random.seed(seed)
        x0 = np.random.randn(total_dim) * 0.1
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < 1e-8:
                U = res.x[:dim*rank_333].reshape(dim, rank_333)
                V = res.x[dim*rank_333:2*dim*rank_333].reshape(dim, rank_333)
                W = res.x[2*dim*rank_333:].reshape(dim, rank_333)
                if verify_333(U, V, W):
                    best_333 = (U, V, W)
                    break
        except:
            pass

    if best_333 is not None:
        U3, V3, W3 = best_333
        # Construct 3x3x6
        # U is 9x46
        # V is 18x46
        # W is 18x46
        
        U6 = np.zeros((9, 46))
        V6 = np.zeros((18, 46))
        W6 = np.zeros((18, 46))
        
        # First 3 columns of B (k=0,1,2)
        # U is same
        U6[:, 0:23] = U3
        # V: j*6+k. For k=0,1,2, index is j*6+k.
        # V3 indices are j*3+k.
        # We need to map V3[j*3+k] to V6[j*6+k].
        for j in range(3):
            for k in range(3):
                V6[j*6+k, 0:23] = V3[j*3+k]
        # W: k*3+i. For k=0,1,2, index is k*3+i.
        # W6 indices are k*3+i (since n=3, p=6, W is 18x46? No, W is p*n = 18x46).
        # W index is k*n + i = k*3 + i.
        # This matches W3 index k*3+i.
        W6[0:9, 0:23] = W3
        
        # Last 3 columns of B (k=3,4,5)
        # U is same
        U6[:, 23:46] = U3
        # V: j*6+k. For k=3,4,5, index is j*6+k.
        # Map V3[j*3+(k-3)] to V6[j*6+k].
        for j in range(3):
            for k in range(3):
                V6[j*6 + (k+3), 23:46] = V3[j*3+k]
        # W: k*3+i. For k=3,4,5, index is k*3+i.
        # W3 index is (k-3)*3+i.
        W6[9:18, 23:46] = W3
        
        # Convert to lists
        return {
            "U": U6.tolist(),
            "V": V6.tolist(),
            "W": W6.tolist()
        }
    else:
        # Fallback to schoolbook
        rank = 54
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(18)]
        W = [[0]*rank for _ in range(18)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    U[i*3+j][col] = 1
                    V[j*6+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
