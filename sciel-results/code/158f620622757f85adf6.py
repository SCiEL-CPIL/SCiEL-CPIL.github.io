import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the matrix multiplication tensor for dimensions n x m times m x p.
    The rank is n*m*p (schoolbook).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin of 10s

    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
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
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (per trace convention W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
                
    if time.time() > deadline:
        # If for some reason this took too long (it shouldn't), return what we have
        # But for 42 columns, it's instantaneous.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Optional: simple validation print for local testing
    res = construct()
    # print("Constructed rank:", len(res["U"][0]))
