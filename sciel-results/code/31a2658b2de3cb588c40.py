import numpy as np
from itertools import product
import time
import sys

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill
    
    # Target Tensor T
    # T[u, v, w] = 1 if u=3i+j, v=6j+k, w=3k+i
    # u in [0, 9), v in [0, 18), w in [0, 18)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def verify(U, V, W):
        # Check if U, V, W are half-integers
        U_int = np.round(U * 2)
        V_int = np.round(V * 2)
        W_int = np.round(W * 2)
        if not np.allclose(U * 2, U_int) or not np.allclose(V * 2, V_int) or not np.allclose(W * 2, W_int):
            return False
        U_half = U_int / 2.0
        V_half = V_int / 2.0
        W_half = W_int / 2.0
        
        # Check tensor equality
        # Reconstruct tensor
        # T_recon[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # Use einsum
        T_recon = np.einsum('ur,vr,wr->uvw', U_half, V_half, W_half)
        if np.allclose(T_recon, T, atol=1e-9):
            return True
        return False

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def try_rank(R):
        # Optimization approach
        # Flatten U, V, W into a single vector x
        # x = [U.flatten(), V.flatten(), W.flatten()]
        
        def objective(x):
            U_flat = x[:n*m*R]
            V_flat = x[n*m*R : n*m*R + m*p*R]
            W_flat = x[n*m*R + m*p*R:]
            
            U = U_flat.reshape(n*m, R)
            V = V_flat.reshape(m*p, R)
            W = W_flat.reshape(p*n, R)
            
            T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            diff = T_recon - T
            return np.sum(diff**2)

        # Initial guess: Random
        np.random.seed(42) # Fixed seed for reproducibility in search, or random?
        # Let's try a few random restarts
        for attempt in range(5):
            if time.time() > deadline:
                return None
            
            # Initialize with small random values
            x0 = np.random.uniform(-1, 1, size=(n*m + m*p + p*n) * R)
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
                if res.success or res.fun < 1e-6:
                    x_opt = res.x
                    U_flat = x_opt[:n*m*R]
                    V_flat = x_opt[n*m*R : n*m*R + m*p*R]
                    W_flat = x_opt[n*m*R + m*p*R:]
                    
                    U = U_flat.reshape(n*m, R)
                    V = V_flat.reshape(m*p, R)
                    W = W_flat.reshape(p*n, R)
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U * 2) / 2.0
                    V_rounded = np.round(V * 2) / 2.0
                    W_rounded = np.round(W * 2) / 2.0
                    
                    if verify(U_rounded, V_rounded, W_rounded):
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            except Exception:
                pass
            
            # If L-BFGS-B fails, try Powell? Or just move to next attempt
            # Powell is slower but derivative-free
            if attempt == 2: # Try Powell in the middle
                 try:
                    from scipy.optimize import minimize
                    res = minimize(objective, x0, method='Powell', options={'maxiter': 100, 'ftol': 1e-8})
                    if res.fun < 1e-6:
                        x_opt = res.x
                        U_flat = x_opt[:n*m*R]
                        V_flat = x_opt[n*m*R : n*m*R + m*p*R]
                        W_flat = x_opt[n*m*R + m*p*R:]
                        U = U_flat.reshape(n*m, R)
                        V = V_flat.reshape(m*p, R)
                        W = W_flat.reshape(p*n, R)
                        U_rounded = np.round(U * 2) / 2.0
                        V_rounded = np.round(V * 2) / 2.0
                        W_rounded = np.round(W * 2) / 2.0
                        if verify(U_rounded, V_rounded, W_rounded):
                            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                 except Exception:
                     pass
        return None

    # Try ranks from 53 down to 40
    # 54 is the schoolbook
    for R in range(53, 39, -1):
        if time.time() > deadline:
            break
        result = try_rank(R)
        if result is not None:
            return result
    
    # Fallback
    return schoolbook()
