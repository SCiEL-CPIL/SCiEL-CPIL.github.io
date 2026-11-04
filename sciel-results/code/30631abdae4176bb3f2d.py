import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Try ranks from low to high
    # Based on the hint, rank 24 might be possible.
    # Schoolbook is 80.
    # Let's try to find a solution below 80.
    
    # First, try a structured construction if possible, otherwise search.
    # Since constructing Winograd for 4x4x5 is complex, we rely on search
    # but with a better fallback.
    
    # If we can't find a low rank quickly, return schoolbook.
    # The current program already does this.
    # To improve, we can try more ranks or better initialization.
    
    # However, the directive says "Keep this design and investigate improvements within its family".
    # So we stick to the search but maybe refine the "exact recovery" step.
    
    # Let's try ranks 20 to 40.
    for rank in [20, 22, 24, 26, 28, 30, 32, 35, 40, 50, 60, 80]:
        if time.time() > deadline - 20:
            break
        
        result = _find_decomposition(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
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

def _find_decomposition(n, m, p, rank, deadline):
    """Find a rank decomposition using alternating minimization with exact recovery."""
    nm = n * m  # 16
    mp = m * p  # 20
    pn = p * n  # 20
    
    np.random.seed(42)
    random.seed(42)
    
    max_restarts = 50 # Increased restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            return None
        
        # Initialization
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Alternating minimization
        for iteration in range(500):
            if time.time() > deadline - 15:
                break
            
            U_new = _solve_factor_U(n, m, p, rank, V, W)
            V_new = _solve_factor_V(n, m, p, rank, U_new, W)
            W_new = _solve_factor_W(n, m, p, rank, U_new, V_new)
            
            U, V, W = U_new, V_new, W_new
        
        # Try to recover exact rational solution
        result = _recover_exact(n, m, p, U, V, W, deadline)
        if result is not None:
            return result
        
        # Verify with tolerance
        if _verify_numpy(n, m, p, U, V, W, tol=1e-8):
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify_numpy(n, m, p, U_r, V_r, W_r, tol=1e-6):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
    
    return None

def _solve_factor_U(n, m, p, rank, V, W):
    nm = n * m
    U = np.zeros((nm, rank))
    for i in range(n):
        for j in range(m):
            A = np.zeros(rank)
            for k in range(p):
                A += V[j * p + k, :] * W[k * n + i, :]
            norm_A = np.linalg.norm(A)
            if norm_A < 1e-12:
                U[i * m + j, :] = 0
            else:
                U[i * m + j, :] = A / (norm_A ** 2)
    return U

def _solve_factor_V(n, m, p, rank, U, W):
    mp = m * p
    V = np.zeros((mp, rank))
    for j in range(m):
        for k in range(p):
            B = np.zeros(rank)
            for i in range(n):
                B += U[i * m + j, :] * W[k * n + i, :]
            norm_B = np.linalg.norm(B)
            if norm_B < 1e-12:
                V[j * p + k, :] = 0
            else:
                V[j * p + k, :] = B / (norm_B ** 2)
    return V

def _solve_factor_W(n, m, p, rank, U, V):
    pn = p * n
    W = np.zeros((pn, rank))
    for k in range(p):
        for i in range(n):
            C = np.zeros(rank)
            for j in range(m):
                C += U[i * m + j, :] * V[j * p + k, :]
            norm_C = np.linalg.norm(C)
            if norm_C < 1e-12:
                W[k * n + i, :] = 0
            else:
                W[k * n + i, :] = C / (norm_C ** 2)
    return W

def _recover_exact(n, m, p, U, V, W, deadline):
    if _verify_numpy(n, m, p, U, V, W, tol=1e-6):
        for scale in [1, 2, 4, 10]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if _verify_numpy(n, m, p, U_r, V_r, W_r, tol=1e-6):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
    return None

def _verify_numpy(n, m, p, U, V, W, tol=1e-6):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
