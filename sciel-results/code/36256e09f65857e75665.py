import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Constructs a rank-50 decomposition for the 2x5 by 5x5 matrix multiplication tensor.
    
    Uses the schoolbook method where each (i, j, k) triple gets its own rank index.
    U, V, W are factor matrices of shapes (10, 50), (25, 50), (10, 50).
    """
    n, m, p = DIMENSIONS
    dim_a = n * m   # 10
    dim_b = m * p   # 25
    dim_c = p * n   # 10
    rank = n * m * p  # 50
    
    U = np.zeros((dim_a, rank), dtype=float)
    V = np.zeros((dim_b, rank), dtype=float)
    W = np.zeros((dim_c, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                U[u_row, r] = 1.0
                V[v_row, r] = 1.0
                W[w_row, r] = 1.0
                r += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
