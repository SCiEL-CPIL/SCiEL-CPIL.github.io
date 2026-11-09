import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    start_time = time.time()
    deadline = start_time + 140

    # Search for lower rank
    # We try ranks from 32 to 55
    for R in range(32, rank_sb):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(10):
            if time.time() > deadline:
                break
            
            U = np.random.randint(-1, 2, (n * m, R))
            V = np.random.randint(-1, 2, (m * p, R))
            W = np.random.randint(-1, 2, (p * n, R))
            
            # Local search
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Check if exact
                # Compute the tensor product sum
                # T_ij,k = sum_c U_ic V_ck W_ki
                # We need T_ij,k = delta_jk (for the inner product)
                # Actually, the condition is:
                # sum_c W_{k,i,c} U_{i,j,c} V_{j,k,c} = delta_{j,j'}
                
                # Let's compute the error
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for j2 in range(m):
                                val = 0
                                for c in range(R):
                                    val += W[k*n + i, c] * U[i*m + j, c] * V[j2*p + k, c]
                                target = 1 if j == j2 else 0
                                error += (val - target) ** 2
                
                if error == 0:
                    best_rank = R
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    return {"U": best_U, "V": best_V, "W": best_W}
                
                # Perturb
                # Pick a random element and flip it
                idx = np.random.randint(0, R)
                if np.random.rand() < 0.33:
                    r = np.random.randint(0, n*m)
                    U[r, idx] *= -1
                elif np.random.rand() < 0.66:
                    r = np.random.randint(0, m*p)
                    V[r, idx] *= -1
                else:
                    r = np.random.randint(0, p*n)
                    W[r, idx] *= -1

    return {"U": best_U, "V": best_V, "W": best_W}
