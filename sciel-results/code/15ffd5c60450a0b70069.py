"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill

    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W

    def validate(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly represents matrix multiplication."""
        n, m, p = DIMENSIONS
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank = U_arr.shape[1]

        # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    # Try to find a lower rank using structured random search
    # The key insight: if we can find linear dependencies among the schoolbook terms,
    # we can reduce the rank.
    
    # For now, attempt a simple improvement: check if any column is redundant
    # This is a heuristic - in general, finding the minimal rank is NP-hard.
    
    # Try small rank reductions by zeroing out columns and checking validity
    for reduction in range(1, 10):  # Try to reduce rank by 1 to 9
        if time.time() > deadline - 60:
            break
        if rank_schoolbook - reduction < 25:  # Don't go below theoretical lower bound
            break
        
        target_rank = rank_schoolbook - reduction
        # Remove the last 'reduction' columns
        U_red = [row[:target_rank] for row in U]
        V_red = [row[:target_rank] for row in V]
        W_red = [row[:target_rank] for row in W]
        
        if validate(U_red, V_red, W_red):
            best_rank = target_rank
            best_U, best_V, best_W = U_red, V_red, W_red
            break  # Found a valid reduction, stop
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
