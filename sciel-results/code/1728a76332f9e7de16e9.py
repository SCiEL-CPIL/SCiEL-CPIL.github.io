"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition (proven optimal rank for 2x2).
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 2
    
    # Strassen's 7 products coefficients
    # U[i*m+j, c]: coefficient of A[i,j] in product c
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1]
    ]

    # V[j*p+k, c]: coefficient of B[j,k] in product c
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],   # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]   # B[1,1]
    ]

    # W[k*n+i, c]: coefficient for C[i,k] from product c
    # Row index = k*n + i
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0]: M1+M4-M5+M7
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]: M2+M4
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]: M3+M5
        [1, -1, 1, 0, 0, 1, 0]   # C[1,1]: M1-M2+M3+M6
    ]
    
    # Check deadline (trivially satisfied, but required by directive)
    if time.time() > deadline:
        # Fallback to schoolbook rank 8 if we are extremely late (should never happen)
        rank = 8
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U_data, "V": V_data, "W": W_data}
# EVOLVE-BLOCK-END
