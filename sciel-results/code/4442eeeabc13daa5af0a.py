import numpy as np
import time
import sys
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    def make_schoolbook():
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
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Check if U, V, W form a valid rank decomposition."""
        R = U.shape[1]
        # Check entries are integer or half-integer
        for mat in [U, V, W]:
            doubled = mat * 2
            if np.any(np.abs(doubled - np.round(doubled)) > 1e-6):
                return False
        
        # Check tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    s = np.sum(u_row * v_row * w_row)
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def try_rank(R, max_iters=5000, tolerance=1e-10):
        """
        Try to find a rank-R decomposition using alternating least squares
        with random restarts.
        """
        for restart in range(20):
            if time.time() > deadline - 5:
                return None
            
            # Initialize with small random values
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            best_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Compute current error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        u_row = U[i * m + j]
                        for k in range(p):
                            v_row = V[j * p + k]
                            w_row = W[k * n + i]
                            s = np.sum(u_row * v_row * w_row)
                            err += (s - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                
                if err < tolerance:
                    break
                
                # Update U: for each (i,j), solve linear system
                # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
                # This is a system: A_u * u = b where A_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
                for i in range(n):
                    for j in range(m):
                        A_u = np.zeros((p, R))
                        b_u = np.ones(p)
                        for k in range(p):
                            A_u[k] = V[j * p + k] * W[k * n + i]
                        # Solve least squares
                        try:
                            U[i * m + j] = np.linalg.lstsq(A_u, b_u, rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), solve
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                for j in range(m):
                    for k in range(p):
                        A_v = np.zeros((n, R))
                        b_v = np.ones(n)
                        for i in range(n):
                            A_v[i] = U[i * m + j] * W[k * n + i]
                        try:
                            V[j * p + k] = np.linalg.lstsq(A_v, b_v, rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), solve
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                for k in range(p):
                    for i in range(n):
                        A_w = np.zeros((m, R))
                        b_w = np.ones(m)
                        for j in range(m):
                            A_w[j] = U[i * m + j] * V[j * p + k]
                        try:
                            W[k * n + i] = np.linalg.lstsq(A_w, b_w, rcond=None)[0]
                        except:
                            pass
            
            # Check if we converged
            if best_err < tolerance:
                # Round to half-integers
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
        
        return None

    # Try ranks from 40 to 59
    for R in range(40, 60):
        if time.time() > deadline - 10:
            break
        result = try_rank(R)
        if result is not None:
            return result

    # Fallback to schoolbook
    return make_schoolbook()
