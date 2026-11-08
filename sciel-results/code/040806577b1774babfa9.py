import numpy as np
import time

def construct():
    """
    Constructs a rank-75 decomposition for 3x5 * 5x5 matrix multiplication.
    Uses standard matrix multiplication tensor decomposition.
    """
    n, m, p = 3, 5, 5
    rank = n * m * p # 75
    
    # Initialize U, V, W as lists of lists to match the interface expectation if needed,
    # but the evaluator likely accepts numpy arrays or lists. The seed used lists.
    # Let's stick to lists to be safe, or convert at the end.
    # The seed returned lists.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i*m + j
                # B index: j*p + k
                # C index: k*n + i (Trace convention: W[k*n+i, r])
                U[i*m + j][r] = 1.0
                V[j*p + k][r] = 1.0
                W[k*n + i][r] = 1.0
                r += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
