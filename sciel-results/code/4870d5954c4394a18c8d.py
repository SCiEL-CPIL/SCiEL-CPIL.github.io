import time
import sys

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill based on schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
            
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
        
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    assert len(U) == n * m
    assert len(V) == m * p
    assert len(W) == p * n
    assert len(U[0]) == rank
    assert len(V[0]) == rank
    assert len(W[0]) == rank
