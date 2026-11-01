import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the optimal rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The rank of 2x2 matrix multiplication is known to be 7.
    This function returns the factor matrices U, V, W for the Strassen decomposition.
    
    Interface:
    U: (n*m, rank) = (4, 7)
    V: (m*p, rank) = (4, 7)
    W: (p*n, rank) = (4, 7)
    
    Indexing:
    U[i*m + j, r]
    V[j*p + k, r]
    W[k*n + i, r]
    
    n=m=p=2
    """
    start_time = time.time()
    deadline = 110.0 # Safety margin for 120s limit
    
    if time.time() > deadline:
        # This should never happen as the construction is O(1)
        raise TimeoutError("Time limit exceeded during construction")

    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen Products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Output Reconstruction:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[1,0] = M2 + M4
    # C[0,1] = M3 + M5
    # C[1,1] = M1 - M2 + M3 + M6
    
    # U factors (row = i*m + j)
    # i=0,j=0 (a): U[0]
    # i=0,j=1 (b): U[1]
    # i=1,j=0 (c): U[2]
    # i=1,j=1 (d): U[3]
    
    # M1 (r=0): a, d
    U[0][0] = 1
    U[3][0] = 1
    
    # M2 (r=1): c, d
    U[2][1] = 1
    U[3][1] = 1
    
    # M3 (r=2): a
    U[0][2] = 1
    
    # M4 (r=3): d
    U[3][3] = 1
    
    # M5 (r=4): a, b
    U[0][4] = 1
    U[1][4] = 1
    
    # M6 (r=5): c, -a
    U[2][5] = 1
    U[0][5] = -1
    
    # M7 (r=6): b, -d
    U[1][6] = 1
    U[3][6] = -1
    
    # V factors (row = j*p + k)
    # j=0,k=0 (e): V[0]
    # j=0,k=1 (f): V[1]
    # j=1,k=0 (g): V[2]
    # j=1,k=1 (h): V[3]
    
    # M1 (r=0): e, h
    V[0][0] = 1
    V[3][0] = 1
    
    # M2 (r=1): e
    V[0][1] = 1
    
    # M3 (r=2): f, -h
    V[1][2] = 1
    V[3][2] = -1
    
    # M4 (r=3): g, -e
    V[2][3] = 1
    V[0][3] = -1
    
    # M5 (r=4): h
    V[3][4] = 1
    
    # M6 (r=5): e, f
    V[0][5] = 1
    V[1][5] = 1
    
    # M7 (r=6): g, h
    V[2][6] = 1
    V[3][6] = 1
    
    # W factors (row = k*n + i)
    # k=0,i=0 (C[0,0]): W[0]
    # k=0,i=1 (C[1,0]): W[1]
    # k=1,i=0 (C[0,1]): W[2]
    # k=1,i=1 (C[1,1]): W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0][0] = 1  # M1
    W[0][3] = 1  # M4
    W[0][4] = -1 # M5
    W[0][6] = 1  # M7
    
    # C[1,0] = M2 + M4
    W[1][1] = 1  # M2
    W[1][3] = 1  # M4
    
    # C[0,1] = M3 + M5
    W[2][2] = 1  # M3
    W[2][4] = 1  # M5
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3][0] = 1  # M1
    W[3][1] = -1 # M2
    W[3][2] = 1  # M3
    W[3][5] = 1  # M6
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
