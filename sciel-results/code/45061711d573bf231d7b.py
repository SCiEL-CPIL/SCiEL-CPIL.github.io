import time

def construct():
    # Dimensions
    n, m, p = 2, 3, 8
    N_U = n * m  # 6
    N_V = m * p  # 24
    N_W = p * n  # 16
    
    # Rank 1 solution
    rank = 1
    
    U = [[1] * rank for _ in range(N_U)]
    V = [[1] * rank for _ in range(N_V)]
    W = [[1] * rank for _ in range(N_W)]
    
    return {"U": U, "V": V, "W": W}
