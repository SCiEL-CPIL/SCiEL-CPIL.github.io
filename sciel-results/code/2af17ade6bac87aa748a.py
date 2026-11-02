import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Returns a rank-minimal (or best found) exact tensor decomposition 
    for multiplying a 3x5 matrix by a 5x7 matrix.
    
    The output format is a dictionary with keys 'U', 'V', 'W'.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    The decomposition satisfies:
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    where T is the matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    
    # Trivial (schoolbook) decomposition
    # Rank = n * m * p
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }

if __name__ == '__main__':
    res = construct()
    print(f"Rank: {len(res['U'][0]) if res['U'] else 0}")
