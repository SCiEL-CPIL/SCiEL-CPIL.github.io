import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_MARGIN = 10.0  # seconds before the 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    def time_left():
        return 150.0 - DEADLINE_MARGIN - (time.time() - start_time)
    
    # Construct schoolbook decomposition
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
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
