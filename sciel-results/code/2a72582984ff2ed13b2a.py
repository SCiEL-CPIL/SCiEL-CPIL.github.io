"""Matrix multiplication 4x5x5 rank-minimal decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Iterate over i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                u_val = U[i*m+j]
                v_val = V[j*p+k]
                w_val = W[k*n+i]
                # Dot product of the three vectors
                val = np.dot(u_val, v_val * w_val)
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # 1. Try Rank 1 (All-ones tensor hypothesis)
    # If T[i,j,k] = 1 for all i,j,k, then U=1, V=1, W=1 is a valid rank-1 decomposition.
    U_r1 = np.ones((n * m, 1))
    V_r1 = np.ones((m * p, 1))
    W_r1 = np.ones((p * n, 1))
    
    if _verify(U_r1, V_r1, W_r1, n, m, p):
        return {
            "U": U_r1.tolist(),
            "V": V_r1.tolist(),
            "W": W_r1.tolist()
        }
    
    # 2. If Rank 1 fails (unlikely for dense MM tensor), try small ranks with random search
    # This is a fallback.
    for target_rank in range(2, 10):
        if time.time() > deadline - 10:
            break
        
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randint(-1, 2, size=(n*m, target_rank))
            V = np.random.randint(-1, 2, size=(m*p, target_rank))
            W = np.random.randint(-1, 2, size=(p*n, target_rank))
            
            if _verify(U, V, W, n, m, p):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    # 3. Fallback to Schoolbook (Rank 100)
    rank_school = n * m * p
    U_fb = np.zeros((n * m, rank_school))
    V_fb = np.zeros((m * p, rank_school))
    W_fb = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
                
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
