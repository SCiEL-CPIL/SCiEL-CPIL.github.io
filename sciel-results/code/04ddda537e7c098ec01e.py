"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    rank_school = n * m * p
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank_school
    
    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > 1e-7:
                        return False
        return True
    
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    np.random.seed(42)
    
    # Try ranks from 55 down to 30
    for target_rank in range(55, 29, -1):
        if time.time() > deadline - 10:
            break
        
        for trial in range(100):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Scale
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)
            
            if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                best_U, best_V, best_W, best_rank = U.tolist(), V.tolist(), W.tolist(), target_rank
                break
        
        if best_rank < target_rank:
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
