import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build list of (a, b, c) indices where tensor is 1
    # a = i*m+j, b = j*p+k, c = k*n+i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    start_time = time.time()
    deadline = start_time + 100  # 50s margin
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a candidate decomposition
    def verify(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for (a, b, c) in triples:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if val != 1:
                return False
        return True
    
    # Try to find low-rank solutions using structured random search
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from 10 to 44
    for R in range(10, 45):
        if time.time() > deadline:
            break
        
        # Multiple random attempts per rank
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U = np.random.randint(-2, 3, (n*m, R))
            V = np.random.randint(-2, 3, (m*p, R))
            W = np.random.randint(-2, 3, (p*n, R))
            
            if verify(U, V, W):
                best_rank = R
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    return schoolbook()
# EVOLVE-BLOCK-END
