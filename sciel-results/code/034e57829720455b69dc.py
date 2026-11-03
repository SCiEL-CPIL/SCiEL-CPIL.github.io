import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    time_limit = 115 # seconds, safety margin for 120s limit

    def is_valid(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
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

    # Try to find a decomposition with rank < rank_schoolbook
    # We will search for integer solutions.
    
    for R in range(rank_schoolbook - 1, 20, -1):
        if time.time() - start_time > time_limit:
            break
            
        attempts = 0
        # Increased attempts to allow more search within time limit
        max_attempts = 500
        
        while attempts < max_attempts and (time.time() - start_time) < time_limit:
            attempts += 1
            
            # Generate random U, V with entries in {-1, 0, 1}
            # Sparse matrices are more likely to have integer solutions
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
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
