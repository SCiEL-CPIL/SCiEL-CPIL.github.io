import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 96 schoolbook
    
    # Schoolbook decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try random search for lower rank
    # This is unlikely to succeed but worth trying
    np.random.seed(42)
    
    for try_rank in range(rank - 1, max(rank - 20, 80), -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            U_rand = np.random.randint(-1, 2, (n * m, try_rank))
            V_rand = np.random.randint(-1, 2, (m * p, try_rank))
            W_rand = np.random.randint(-1, 2, (p * n, try_rank))
            
            # Check if this is a valid decomposition
            # Condition: for all i, k, j, j': sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] == delta_{j,j'}
            valid = True
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            s = 0
                            for r in range(try_rank):
                                s += W_rand[k * n + i, r] * U_rand[i * m + j, r] * V_rand[jp * p + k, r]
                            expected = 1 if j == jp else 0
                            if s != expected:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = try_rank
                best_U = U_rand.tolist()
                best_V = V_rand.tolist()
                best_W = W_rand.tolist()
                break
        
        if best_rank < rank:
            break
    
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
    # EVOLVE-BLOCK-END
