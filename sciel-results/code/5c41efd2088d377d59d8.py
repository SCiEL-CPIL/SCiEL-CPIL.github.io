import time
import numpy as np

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS

def construct():
    # Trivial decomposition: rank = n * m * p
    # For each (i, j, k), we have a rank-1 term
    rank_total = n * m * p
    
    U = [[0.0] * rank_total for _ in range(n * m)]
    V = [[0.0] * rank_total for _ in range(m * p)]
    W = [[0.0] * rank_total for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set up rank-1 term for this (i,j,k)
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time taken: {end - start}")
    # Basic verification
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    errors += 1
    if errors == 0:
        print("Verification PASSED")
    else:
        print(f"Verification FAILED with {errors} errors")
