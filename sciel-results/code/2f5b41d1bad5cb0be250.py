import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def verify_decomposition(U, V, W):
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.sum((T_comp - T)**2)
        return error < 1e-9

    def try_rank_gd(R, timeout=5.0):
        """
        Attempts to find a decomposition of rank R using Gradient Descent.
        Returns (U, V, W) as numpy arrays if successful, else None.
        """
        start_time = time.time()
        
        # Try a few random initializations
        for attempt in range(20):
            if time.time() - start_time > timeout:
                break
            
            np.random.seed(1000 + attempt)
            
            # Initialize with small random values
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            for step in range(500):
                if time.time() - start_time > timeout:
                    break
                
                # Compute current tensor
                T_comp = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_comp - T
                error = np.sum(diff**2)
                
                if error < 1e-8:
                    # Snap to nearest half-integer
                    U_snap = np.round(U * 2) / 2.0
                    V_snap = np.round(V * 2) / 2.0
                    W_snap = np.round(W * 2) / 2.0
                    if verify_decomposition(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap
                    break
                
                # Compute gradients
                # grad_U[a, r] = -2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
                grad_U = -2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
                grad_V = -2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
                grad_W = -2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
                
                # Update momentum
                vU = momentum * vU - lr * grad_U
                vV = momentum * vV - lr * grad_V
                vW = momentum * vW - lr * grad_W
                
                U += vU
                V += vV
                W += vW
                
                # Adaptive learning rate
                if error > 1.0:
                    lr *= 0.5
            
        return None

    # Deadline guard
    deadline = time.time() + 140  # 140s safety margin
    
    # Try to find rank 104
    # If found, return it.
    result = try_rank_gd(rank_school - 1, timeout=10.0)
    if result is not None:
        U, V, W = result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Fallback to schoolbook
    rank = rank_school
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
