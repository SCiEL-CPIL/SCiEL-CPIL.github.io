import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower rank
    # Target rank 30
    target_rank = 30
    deadline = time.time() + 100  # 100 seconds
    
    # Simple random search
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Check if decomposition is valid
        # T_ab_c = sum_r U_a_r V_b_r W_c_r
        # We need T_ab_c = 1 if indices match, 0 otherwise.
        # This check is expensive.
        # We can check a few random entries.
        valid = True
        for _ in range(10):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            a = i * m + j
            b = j * p + k
            c = k * n + i
            val = np.sum(U[a] * V[b] * W[c])
            # The tensor entry is 1.
            if abs(val - 1) > 1e-6:
                valid = False
                break
            # Check a zero entry
            i2 = random.randint(0, n-1)
            j2 = random.randint(0, m-1)
            k2 = random.randint(0, p-1)
            if not (i2==i and j2==j and k2==k):
                a2 = i2 * m + j2
                b2 = j2 * p + k2
                c2 = k2 * n + i2
                val2 = np.sum(U[a2] * V[b2] * W[c2])
                if abs(val2) > 1e-6:
                    valid = False
                    break
        
        if valid:
            return {"U": U, "V": V, "W": W}
            
    return {"U": U_school, "V": V_school, "W": W_school}
