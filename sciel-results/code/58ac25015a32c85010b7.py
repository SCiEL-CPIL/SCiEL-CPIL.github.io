import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # seconds, safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Matrix multiplication (n x m) * (m x p) -> (n x p)
    # A is 2x2, B is 2x13, C is 2x13
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Indices:
    # A: i*m + j = 2*i + j (i in 0..1, j in 0..1) -> 4 entries
    # B: j*p + k = 13*j + k (j in 0..1, k in 0..12) -> 26 entries
    # C: k*n + i = 2*k + i (k in 0..12, i in 0..1) -> 26 entries (official trace convention W[k*n+i, r])
    
    # Strategy: Use rank 4 per column for the direct decomposition.
    # Total rank = 4 * 13 = 52.
    # This is guaranteed to be correct.
    
    rank = 4 * p
    U = [[0] * rank for _ in range(n * m)]  # 4 x 52
    V = [[0] * rank for _ in range(m * p)]  # 26 x 52
    W = [[0] * rank for _ in range(p * n)]  # 26 x 52
    
    r = 0
    for k in range(p):
        # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k]
        # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k]
        
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        U[0][r] = 1       # A[0,0]
        V[k][r] = 1       # B[0,k] (j=0)
        W[2*k][r] = 1     # C[0,k] (i=0)
        r += 1
        
        # Term 2: A[0,1] * B[1,k] -> C[0,k]
        U[1][r] = 1       # A[0,1]
        V[13+k][r] = 1    # B[1,k] (j=1)
        W[2*k][r] = 1     # C[0,k] (i=0)
        r += 1
        
        # Term 3: A[1,0] * B[0,k] -> C[1,k]
        U[2][r] = 1       # A[1,0]
        V[k][r] = 1       # B[0,k] (j=0)
        W[2*k+1][r] = 1   # C[1,k] (i=1)
        r += 1
        
        # Term 4: A[1,1] * B[1,k] -> C[1,k]
        U[3][r] = 1       # A[1,1]
        V[13+k][r] = 1    # B[1,k] (j=1)
        W[2*k+1][r] = 1   # C[1,k] (i=1)
        r += 1
        
        if time.time() - start_time > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
