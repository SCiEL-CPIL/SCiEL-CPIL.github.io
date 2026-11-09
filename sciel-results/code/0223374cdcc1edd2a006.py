"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    # Schoolbook solution (always valid)
    def make_schoolbook():
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

    best_result = make_schoolbook()
    best_rank = n * m * p

    # Fast verification using numpy
    def verify_np(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U_np[i * m + j]
                    vj = V_np[j * p + k]
                    wk = W_np[k * n + i]
                    s = np.dot(ui * vj, wk)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try to find a lower rank decomposition using random search
    # Focus on ranks slightly below 72
    search_ranks = [64, 60, 54, 48, 36]
    
    # Define the set of values to use in random search
    # Integer and half-integer values
    values = [0, 0.5, -0.5, 1, -1, 2, -2]

    for target_rank in search_ranks:
        if time_left() < 10:
            break
        if target_rank >= best_rank:
            continue
            
        # Try a few random initializations
        num_trials = 100
        for trial in range(num_trials):
            if time_left() < 5:
                break
                
            U = [[random.choice(values) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(values) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(values) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify_np(U, V, W):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break
        
        if best_rank < n * m * p:
            break

    # Final verification
    if not verify_np(best_result["U"], best_result["V"], best_result["W"]):
        best_result = make_schoolbook()

    return best_result
# EVOLVE-BLOCK-END
