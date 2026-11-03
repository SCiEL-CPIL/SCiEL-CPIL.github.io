import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    time_limit = 115  # seconds, safety margin for 120s limit

    def is_valid(U, V, W):
        """
        Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        """
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(R):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if val != 1:
                        return False
        return True

    # Fallback to Schoolbook
    def get_schoolbook():
        rank = rank_schoolbook
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
        return {"U": U, "V": V, "W": W}

    # Strategy: Search for integer decompositions with rank < rank_schoolbook
    # We will try to find a solution where U, V, W have entries in {-1, 0, 1}
    # or small integers.
    
    # We start from rank_schoolbook - 1 down to a lower bound.
    # The theoretical lower bound is roughly max(n*m, m*p, p*n) = 12.
    # Known best for 3x3x4 is likely around 20-25?
    # We will try a few specific ranks.
    
    ranks_to_try = [rank_schoolbook - 1, rank_schoolbook - 2, rank_schoolbook - 3, 25, 24, 23, 22, 21, 20]
    # Ensure we don't try ranks too low or duplicates
    seen_ranks = set()
    valid_ranks = []
    for r in ranks_to_try:
        if r > 12 and r < rank_schoolbook and r not in seen_ranks:
            valid_ranks.append(r)
            seen_ranks.add(r)

    for R in valid_ranks:
        if time.time() - start_time > time_limit:
            break

        # For each rank, we perform a random search.
        # We generate random sparse U and V, then solve for W.
        # If W is integer and exact, we have a solution.
        
        # To increase the chance of finding an integer solution, we can:
        # 1. Use very sparse U and V (mostly zeros, some 1s and -1s).
        # 2. Try to make the system A_mat @ W_flat = b_vec have a unique integer solution.
        
        attempts = 0
        max_attempts = 200  # Adjust based on time
        
        while attempts < max_attempts and (time.time() - start_time) < time_limit:
            attempts += 1
            
            # Generate random sparse U and V
            # Density of non-zeros: ~0.1 to 0.3
            density = 0.2
            U = np.zeros((n * m, R), dtype=int)
            V = np.zeros((m * p, R), dtype=int)
            
            # Randomly place 1s and -1s
            for i in range(n * m):
                for r in range(R):
                    if np.random.rand() < density:
                        U[i, r] = np.random.choice([-1, 1])
            
            for i in range(m * p):
                for r in range(R):
                    if np.random.rand() < density:
                        V[i, r] = np.random.choice([-1, 1])
            
            # Construct linear system for W
            num_vars = p * n * R
            A_mat = np.zeros((n * m * p, num_vars))
            b_vec = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_base = k * n + i
                        for r in range(R):
                            var_idx = w_base * R + r
                            A_mat[eq_idx, var_idx] = U[u_idx, r] * V[v_idx, r]
                        eq_idx += 1
            
            # Solve A_mat @ W_flat = b_vec
            try:
                # Use lstsq to find the least squares solution
                W_flat, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check if the solution is close to integers
                # If the residual is small and the values are close to integers, it's a candidate
                if residuals[0] < 1e-9:
                    W_int = np.round(W_flat)
                    if np.allclose(A_mat @ W_int, b_vec, atol=1e-9):
                        W_sol = W_int.reshape((p * n, R))
                        # Final validation
                        if is_valid(U, V, W_sol):
                            return {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W_sol.tolist()
                            }
            except:
                pass

    # Return schoolbook
    return get_schoolbook()

# EVOLVE-BLOCK-START
# (No changes to interface)
# EVOLVE-BLOCK-END
