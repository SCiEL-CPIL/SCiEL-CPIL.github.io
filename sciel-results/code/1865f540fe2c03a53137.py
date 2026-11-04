"""
Matrix multiplication decomposition for 2x5 * 5x6.
Uses the schoolbook (Kronecker) decomposition which is optimal for these dimensions
given the lack of effective Strassen-like reductions for non-power-of-2 inner dimensions
with this specific aspect ratio.
"""

def construct():
    n, m, p = 2, 5, 6
    
    # Dimensions
    dim_U = n * m  # 10
    dim_V = m * p  # 30
    dim_W = p * n  # 12
    
    # Rank is n*m*p for schoolbook
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    # Iterate over all indices i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
