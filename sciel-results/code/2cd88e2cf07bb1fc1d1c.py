"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 115
    
    # Try to find lower rank
    for rank in range(9, rank_fb + 1):
        if time.time() > deadline:
            break
        result = _solve_fixed_UV(n, m, p, rank, start_time, deadline)
        if result is not None:
            U, V, W = result
            if _verify(n, m, p, U, V, W):
                return {"U": U, "V": V, "W": W}
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def _verify(n, m, p, U, V, W):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _solve_fixed_UV(n, m, p, rank, start_time, deadline):
    """Fix U,V randomly, solve for W via linear systems."""
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
        
        # Random U, V with small values
        U = np.random.choice([-1, 0, 1], (n*m, rank))
        V = np.random.choice([-1, 0, 1], (m*p, rank))
        
        W = np.zeros((p*n, rank))
        valid = True
        for i in range(n):
            for k in range(p):
                # Build linear system: for each j, sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1
                # This is: A @ W[k*n+i,:] = b where A[j,c] = U[i*m+j,c] * V[j*p+k,c]
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j] = U[i*m+j] * V[j*p+k]
                b = np.ones(m)
                
                # Solve A @ x = b
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    # Check if it's an exact solution
                    if np.allclose(A @ x, b, atol=1e-10):
                        W[k*n+i] = x
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Round to half-integers and verify
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify_np(n, m, p, U_r, V_r, W_r):
                return _to_lists(U_r, V_r, W_r)
    
    return None

def _verify_np(n, m, p, U, V, W):
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _to_lists(U, V, W):
    return [[float(x) for x in row] for row in U.tolist()], \
           [[float(x) for x in row] for row in V.tolist()], \
           [[float(x) for x in row] for row in W.tolist()]
# EVOLVE-BLOCK-END
