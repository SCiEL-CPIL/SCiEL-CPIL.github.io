import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Build target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
                valid_indices.append((a, b, c))
                
    # Schoolbook
    def get_schoolbook():
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
        
    # Try to find a lower rank
    # We will try ranks 24, 25, 26, 27, 28, 29
    for R in [24, 25, 26, 27, 28, 29]:
        if time.time() > deadline:
            break
        # Try a few random initializations
        for trial in range(100):
            if time.time() > deadline:
                break
            # Random init with integers -1, 0, 1
            U = np.random.randint(-1, 2, (n * m, R))
            V = np.random.randint(-1, 2, (m * p, R))
            W = np.random.randint(-1, 2, (p * n, R))
            
            # Check
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            if np.array_equal(T_calc, T_target):
                return {"U": U, "V": V, "W": W}
                
    return get_schoolbook()
