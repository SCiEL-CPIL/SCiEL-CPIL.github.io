import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 6, 12, 8
    
    # Build the target tensor T[a,b,c]
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # T[a,b,c] = 1 if i=i', j=j', k=k', else 0
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    def residual(U, V, W):
        # Compute sum_r U[a,r] V[b,r] W[c,r] for all a,b,c
        # and return the difference from T
        # U: (nm, R), V: (mp, R), W: (pn, R)
        R = U.shape[1]
        # Compute the tensor
        # T_comp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # = (U * V^T * W) in some sense
        # Let's compute it as: for each r, outer(U[:,r], outer(V[:,r], W[:,r]))
        T_comp = np.zeros((nm, mp, pn))
        for r in range(R):
            T_comp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return T_comp - T
    
    def check_valid(U, V, W, tol=1e-6):
        res = residual(U, V, W)
        return np.max(np.abs(res)) < tol
    
    # Try ranks from 22 to 24
    for R in [22, 23, 24]:
        # Initialize with schoolbook and truncate
        U0 = np.zeros((nm, R))
        V0 = np.zeros((mp, R))
        W0 = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < R:
                        U0[i*m+j, col] = 1
                        V0[j*p+k, col] = 1
                        W0[k*n+i, col] = 1
                        col += 1
        # Add some random perturbation for the remaining columns
        for r in range(col, R):
            U0[:, r] = np.random.randn(nm) * 0.1
            V0[:, r] = np.random.randn(mp) * 0.1
            W0[:, r] = np.random.randn(pn) * 0.1
        
        # Use ALS
        U, V, W = U0.copy(), V0.copy(), W0.copy()
        for iteration in range(1000):
            # Fix V, W, solve for U
            # For each a: A * U[a,:]^T = t_a
            # A[(b,c), r] = V[b,r] * W[c,r]
            A = np.zeros((mp * pn, R))
            idx = 0
            for b in range(mp):
                for c in range(pn):
                    A[idx, :] = V[b, :] * W[c, :]
                    idx += 1
            # t_a is T[a, :, :] reshaped to (mp*pn,)
            for a in range(nm):
                t_a = T[a, :, :].reshape(-1)
                # Solve A * x = t_a by least squares
                U[a, :], residuals, rank, sv = np.linalg.lstsq(A, t_a, rcond=None)
            
            # Fix U, W, solve for V
            # For each b: A * V[b,:]^T = t_b
            # A[(a,c), r] = U[a,r] * W[c,r]
            A = np.zeros((nm * pn, R))
            idx = 0
            for a in range(nm):
                for c in range(pn):
                    A[idx, :] = U[a, :] * W[c, :]
                    idx += 1
            for b in range(mp):
                t_b = T[:, b, :].reshape(-1)
                V[b, :], residuals, rank, sv = np.linalg.lstsq(A, t_b, rcond=None)
            
            # Fix U, V, solve for W
            # For each c: A * W[c,:]^T = t_c
            # A[(a,b), r] = U[a,r] * V[b,r]
            A = np.zeros((nm * mp, R))
            idx = 0
            for a in range(nm):
                for b in range(mp):
                    A[idx, :] = U[a, :] * V[b, :]
                    idx += 1
            for c in range(pn):
                t_c = T[:, :, c].reshape(-1)
                W[c, :], residuals, rank, sv = np.linalg.lstsq(A, t_c, rcond=None)
            
            # Check convergence
            if iteration % 10 == 0:
                res = residual(U, V, W)
                max_res = np.max(np.abs(res))
                if max_res < 1e-6:
                    break
        
        if check_valid(U, V, W):
            # Convert to lists
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    R = n * m * p
    U = [[0] * R for _ in range(nm)]
    V = [[0] * R for _ in range(mp)]
    W = [[0] * R for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
