DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will use Strassen for 2x2x2 blocks at positions:
    # i in [0,1], j in [0,1] and [2,3], k in [0,1], [2,3], [4,5]
    # Total 6 blocks.
    
    # Rank calculation
    # Schoolbook for all: 90
    # Strassen for 6 blocks: 6 * 7 = 42
    # Schoolbook for remaining: 90 - 6 * 8 = 42
    # Total rank = 84
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen block indices
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # i0, j0, k0 are the starting indices of the 2x2x2 block
        # i0 in 0..1, j0 in 0..3, k0 in 0..4
        
        # Strassen 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to add these to U, V, W.
        # Each multiplication corresponds to a rank-1 tensor.
        # U[:, col] = A_part
        # V[:, col] = B_part
        # W[:, col] = C_part (which is A_part * B_part in the block)
        
        # M1
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # M2
        U[(i0+1)*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + k0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # M3
        U[i0*m + j0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = -1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # M4
        U[(i0+1)*m + (j0+1)][col] = 1
        V[(j0+1)*p + k0][col] = 1
        V[j0*p + k0][col] = -1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # M5
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        V[(j0+1)*p + k0][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # M6
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # M7
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1

    # Add Strassen blocks
    for i0 in range(0, n, 2):
        for j0 in range(0, m, 2):
            for k0 in range(0, p, 2):
                if i0+1 < n and j0+1 < m and k0+1 < p:
                    add_strassen_block(i0, j0, k0)
    
    # Add schoolbook for the rest
    # We need to track which (i,j,k) tuples are covered by Strassen.
    # Strassen covers (i,j,k) where i in [i0, i0+1], j in [j0, j0+1], k in [k0, k0+1].
    # We can use a set to mark covered tuples.
    covered = set()
    for i0 in range(0, n, 2):
        for j0 in range(0, m, 2):
            for k0 in range(0, p, 2):
                if i0+1 < n and j0+1 < m and k0+1 < p:
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                covered.add((i0+di, j0+dj, k0+dk))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    # Trim if necessary (should be exact)
    return {"U": U, "V": V, "W": W}
