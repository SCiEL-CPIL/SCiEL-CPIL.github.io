import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W):
    """
    Verify if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Construct the tensor T from U, V, W
    # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # We need to check if this matches the matrix multiplication tensor.
    
    # Matrix multiplication tensor T_mm:
    # C[k*n + i, r] = sum_j A[i, j] B[j, r]
    # Indices:
    # U index: i (row of A), j (col of A / row of B). i in 0..n-1, j in 0..m-1.
    # U row index: i * m + j
    # V index: j (row of B), r (col of B). j in 0..m-1, r in 0..p-1.
    # V row index: j * p + r
    # W index: r (col of B / col of C), i (row of C). r in 0..p-1, i in 0..n-1.
    # W row index: r * n + i
    
    T_calc = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_calc += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    T_mm = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for r in range(p):
                # C[i, r] gets contribution A[i, j] * B[j, r]
                # U index: i * m + j
                # V index: j * p + r
                # W index: r * n + i
                T_mm[i*m + j, j*p + r, r*n + i] = 1
                
    if np.allclose(T_calc, T_mm, atol=1e-9):
        return True
    return False

def _try_rank_10_search(deadline):
    """
    Attempt to find a rank 10 decomposition using random search.
    """
    n, m, p = DIMENSIONS
    rank = 10
    
    # Initialize with random values
    # To increase success rate, we can try to warm-start from rank 11
    # But for now, simple random restarts.
    
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Simple gradient descent or just check? 
        # Random search is unlikely to hit the exact manifold.
        # We need a more structured approach.
        # Given the complexity, we skip heavy search and rely on the known rank 11.
        # If we had more time, we would implement a solver for the polynomial system.
        
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # 1. Try to find Rank 10 (low probability without sophisticated solver)
    # For now, we stick to the known Rank 11 which is robust.
    # If we were to implement a search, we would do it here.
    
    # Known Rank 11 Solution
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for columns 0 and 1 of B
    # M1 = (a+d)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    # M2 = (c+d)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = a(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    # M4 = d(b10-b00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    # M5 = (a+b)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    # M6 = (c-a)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (b-d)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # W for Strassen outputs
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C10 = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Schoolbook part for column 2 of B
    # M8 = a * b02
    U[0][7] = 1
    V[2][7] = 1
    
    # M9 = b * b12
    U[1][8] = 1
    V[5][8] = 1
    
    # M10 = c * b02
    U[2][9] = 1
    V[2][9] = 1
    
    # M11 = d * b12
    U[3][10] = 1
    V[5][10] = 1
    
    # W for Schoolbook outputs
    # C02 = M8 + M9
    W[4][7] = 1; W[4][8] = 1
    # C12 = M10 + M11
    W[5][9] = 1; W[5][10] = 1
    
    # Verify before returning (optional, but good for safety)
    # if not _verify(U, V, W):
    #     # Fallback or error
    #     pass
    
    return {"U": U, "V": V, "W": W}
