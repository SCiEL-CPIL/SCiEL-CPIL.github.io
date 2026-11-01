"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x4():
    """
    Returns U, V, W for a rank-10 decomposition of 2x2 * 2x4.
    Shapes: U (4, 10), V (8, 10), W (8, 10).
    Indices:
    U: row i*m+j (0..3)
    V: row j*p+k (0..7)
    W: row k*n+i (0..7)
    """
    n, m, p = 2, 2, 4
    rank = 10
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices mapping:
    # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V rows: 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(1,0), 5:(1,1), 6:(1,2), 7:(1,3)
    # W rows: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1), 4:(0,2), 5:(1,2), 6:(0,3), 7:(1,3)
    # Note: W index is k*n+i. 
    # k=0, i=0 -> 0. k=0, i=1 -> 1.
    # k=1, i=0 -> 2. k=1, i=1 -> 3.
    # k=2, i=0 -> 4. k=2, i=1 -> 5.
    # k=3, i=0 -> 6. k=3, i=1 -> 7.

    # Product 1: (a+d)(e+j) -> A(0,0)+A(1,1) * B(0,0)+B(1,1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[4, 0] = 1; W[7, 0] = 1 # C00, C11, C02, C13

    # Product 2: (c+d)e -> A(1,0)+A(1,1) * B(0,0)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1 # C10, C11
    W[5, 1] = 1; W[7, 1] = -1 # C12, C13

    # Product 3: a(i-e) -> A(0,0) * B(1,0)-B(0,0)
    U[0, 2] = 1
    V[4, 2] = 1; V[0, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1 # C01, C11
    W[6, 2] = 1; W[7, 2] = 1 # C03, C13

    # Product 4: d(j-f) -> A(1,1) * B(1,1)-B(0,1)
    U[3, 3] = 1
    V[5, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C00, C01
    W[4, 3] = 1; W[6, 3] = 1 # C02, C03

    # Product 5: (a+b)j -> A(0,0)+A(0,1) * B(1,1)
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1 # C00, C01
    W[4, 4] = -1; W[6, 4] = 1 # C02, C03

    # Product 6: (c-a)(e+f) -> A(1,0)-A(0,0) * B(0,0)+B(0,1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1; W[7, 5] = 1 # C11, C13

    # Product 7: (b-d)(i+j) -> A(0,1)-A(1,1) * B(1,0)+B(1,1)
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1; W[4, 6] = 1 # C00, C02

    # Product 8: (a+d)(g+l) -> A(0,0)+A(1,1) * B(0,2)+B(1,3)
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[7, 7] = 1
    W[4, 7] = 1; W[7, 7] = 1 # C02, C13

    # Product 9: (c+d)g -> A(1,0)+A(1,1) * B(0,2)
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1
    W[5, 8] = 1; W[7, 8] = -1 # C12, C13

    # Product 10: a(k-g) -> A(0,0) * B(1,2)-B(0,2)
    U[0, 9] = 1
    V[6, 9] = 1; V[2, 9] = -1
    W[6, 9] = 1; W[7, 9] = 1 # C03, C13

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a, b, c = i*m+j, j*p+k, k*n+i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check zeros for non-matching indices
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i, j = a//m, a%m
                jj, k = b//p, b%p
                kk, ii = c//n, c%n
                if j != jj or k != kk or i != ii:
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > 1e-9:
                        return False
    return True

def construct():
    """
    Construct rank-30 decomposition for 2x2 * 2x12.
    Tiling the rank-10 decomposition of 2x2 * 2x4 three times.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Get the base decomposition for 2x2x4
    U4, V4, W4 = _strassen_2x2x4()
    
    # Verify base decomposition
    if not _verify_decomposition(U4, V4, W4, 2, 2, 4):
        # Fallback to a simpler but higher rank decomposition if base fails
        # Rank 48 schoolbook
        rank = 48
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

    # Tile for p=12
    rank = 30
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for t in range(3):
        base_col = 4*t
        for r in range(10):
            U[:, t*10+r] = U4[:, r]
            for j in range(2):
                for k in range(4):
                    V[j*p + base_col + k, t*10+r] = V4[j*4+k, r]
            for k in range(4):
                for i in range(2):
                    W[(base_col+k)*n+i, t*10+r] = W4[k*2+i, r]
    
    # Final verification
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Should not happen, but fallback
        rank = 48
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

# EVOLVE-BLOCK-END
