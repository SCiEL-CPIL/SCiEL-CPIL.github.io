"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120s to be safe within 150s
    
    # Index mappings
    # U index: i*m + j, where i in [0,n), j in [0,m)
    # V index: j*p + k, where j in [0,m), k in [0,p)
    # W index: k*n + i, where k in [0,p), i in [0,n)
    
    # The tensor equation: for all (i,j,k) in [0,n)x[0,m)x[0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    best_rank = n * m * p  # schoolbook
    best_solution = None
    
    # Generate schoolbook solution as baseline (Integers are valid half-integers)
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    for col in range(best_rank):
        i = col // (m*p)
        j = (col // p) % m
        k = col % p
        U_sb[i*m+j, col] = 1.0
        V_sb[j*p+k, col] = 1.0
        W_sb[k*n+i, col] = 1.0
    best_solution = (U_sb, V_sb, W_sb)
    
    def check_solution(U, V, W, tol=1e-8):
        """Check if the solution is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def is_valid_half_integer(arr, tol=1e-9):
        """Check if all entries are integers or half-integers."""
        for val in arr.flatten():
            if abs(val * 2 - round(val * 2)) > tol:
                return False
        return True

    # Random search for lower rank using only half-integer values
    # This is a heuristic search within the family of exact decompositions
    search_time_limit = 10.0 # seconds
    start_time = time.time()
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound is roughly n*m*p / max(n,m,p) but let's start from 30
    min_rank_to_try = 30
    
    for rank in range(min_rank_to_try, best_rank):
        if time.time() - start_time > search_time_limit:
            break
            
        # Limit number of restarts per rank to avoid hanging
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - start_time > search_time_limit:
                break
            
            np.random.seed(rank * 1000 + restart)
            
            # Sample from {-1, -0.5, 0, 0.5, 1}
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            U = np.random.choice(vals, size=(n*m, rank))
            V = np.random.choice(vals, size=(m*p, rank))
            W = np.random.choice(vals, size=(p*n, rank))
            
            if check_solution(U, V, W):
                best_rank = rank
                best_solution = (U, V, W)
                break
        if best_rank == rank:
            break
            
    # If no better solution found, use schoolbook
    U, V, W = best_solution
    
    # Final safety check: ensure returned values are exactly representable
    # and valid. The schoolbook solution is always valid.
    # If random search found one, it was validated.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
