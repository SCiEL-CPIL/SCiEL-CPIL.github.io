"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import time
_start_time = time.time()
TIME_LIMIT = 140.0  # 150s limit, 10s safety margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 6
    
    def check_time():
        return (time.time() - _start_time) > TIME_LIMIT
    
    # Construction 1: Rank-18 construction
    # For each (j,k) pair, use one rank-1 term:
    # U[i*m+j, c] = 1 for all i, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for all i
    # This gives sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k*n+i,c] = delta_{j,j'}
    def rank18_decomposition():
        rank = m * p  # 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                for i in range(n):
                    W[k * n + i][col] = 1
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition
    def verify_decomposition(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j][c] * V[jp*p+k][c] * W[k*n+i][c]
                        expected = 1.0 if j == jp else 0.0
                        if abs(s - expected) > 1e-9:
                            return False
        return True
    
    # Try rank-18 first
    result = rank18_decomposition()
    if verify_decomposition(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: trivial rank-36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
