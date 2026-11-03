"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for matrix multiplication
    of dimensions n x m and m x p.
    Uses the schoolbook algorithm which has rank n*m*p.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin of 5 seconds before 150s limit

    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i, j, c] = 1
                # V[j, k, c] = 1
                # W[k, i, c] = 1
                # Indices in flattened arrays:
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
                # Check time
                if time.time() > deadline:
                    # Return current partial result if it happens to be complete, 
                    # otherwise this shouldn't happen for such small n,m,p
                    # But to be safe, we break and return what we have.
                    # Since rank is small, this loop finishes in microseconds.
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
