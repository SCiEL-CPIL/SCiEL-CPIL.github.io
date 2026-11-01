import numpy as np
import time
import random

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Schoolbook decomposition (rank 12)
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_school
    
    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T_target = get_target_tensor()
    
    def compute_tensor(U, V, W):
        """Compute tensor from factor matrices using einsum for speed"""
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def verify(U, V, W):
        """Check if decomposition is exact"""
        T = compute_tensor(U, V, W)
        return np.allclose(T, T_target, atol=1e-9)
    
    # Try to find lower rank solutions using structured search
    # Allowed values: integers and half-integers in range [-3, 3]
    allowed_vals = [float(x) for x in range(-6, 7)]  # -3, -2.5, -2, ..., 2.5, 3
    
    def try_rank(target_rank, max_attempts=50):
        nonlocal best_U, best_V, best_W, best_rank
        if target_rank >= best_rank:
            return False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                return False
            
            # Random initialization with allowed values
            U = np.random.choice(allowed_vals, size=(n*m, target_rank))
            V = np.random.choice(allowed_vals, size=(m*p, target_rank))
            W = np.random.choice(allowed_vals, size=(p*n, target_rank))
            
            if verify(U, V, W):
                best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                best_rank = target_rank
                return True
        
        return False
    
    # Try ranks from 10 up to 11 (12 is schoolbook)
    for target_rank in [10, 11]:
        if try_rank(target_rank, max_attempts=100):
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
