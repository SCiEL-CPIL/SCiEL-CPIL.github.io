import random
import time
import sys

DIMENSIONS = [3, 5, 6]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety (limit is 120s)
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition:
    # For each (i, j, k), create a term with:
    #   U[i*m+j, col] = 1
    #   V[j*p+k, col] = 1
    #   W[k*n+i, col] = 1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Self-test
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Verify shapes
    assert len(U) == n * m
    assert len(V) == m * p
    assert len(W) == p * n
    assert len(U[0]) == rank
    assert len(V[0]) == rank
    assert len(W[0]) == rank
    
    # Verify correctness
    import random
    random.seed(42)
    A = [[random.randint(0, 10) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(0, 10) for _ in range(p)] for _ in range(m)]
    
    # Compute C = A * B
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        for k in range(p):
            s = 0
            for j in range(m):
                s += A[i][j] * B[j][k]
            C[i][k] = s
            
    # Compute C' using tensor decomposition
    C_prime = [[0] * p for _ in range(n)]
    for c in range(rank):
        sum_u = sum(U[i * m + j][c] * A[i][j] for i in range(n) for j in range(m))
        sum_v = sum(V[j * p + k][c] * B[j][k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_prime[i][k] += sum_u * sum_v * W[k * n + i][c]
                
    for i in range(n):
        for k in range(p):
            assert C[i][k] == C_prime[i][k], f"Mismatch at ({i},{k}): {C[i][k]} vs {C_prime[i][k]}"
                
    print("Decomposition verified. Rank =", rank)
