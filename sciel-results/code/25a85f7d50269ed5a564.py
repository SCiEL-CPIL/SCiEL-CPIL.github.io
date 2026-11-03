import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We'll build the decomposition incrementally
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    
    # Try to build a rank-50 decomposition using Strassen on 2x2 blocks
    # Split m=4 into two 2s, p=7 into 3*2+1
    
    # We'll use a list of columns, each column is (U_col, V_col, W_col)
    columns = []
    
    def add_strassen_2x2_2x2(i0, j0, k0):
        """Add Strassen decomposition for 2x2 A block at (i0,j0), 2x2 B block at (j0,k0)."""
        # M1 = (a11+a22)(b11+b22)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m+j0] = 1; u[(i0+1)*m+(j0+1)] = 1
        v[j0*p+k0] = 1; v[(j0+1)*p+(k0+1)] = 1
        w[k0*n+i0] = 1; w[(k0+1)*n+(i0+1)] = 1
        columns.append((u, v, w))
        
        # M2 = (a21+a22)b11
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i0+1)*m+j0] = 1; u[(i0+1)*m+(j0+1)] = 1
        v[j0*p+k0] = 1
        w[(k0)*n+(i0+1)] = 1
        columns.append((u, v, w))
        
        # M3 = a11(b12-b22)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m+j0] = 1
        v[j0*p+(k0+1)] = 1; v[(j0+1)*p+(k0+1)] = -1
        w[k0*n+i0] = 1; w[(k0+1)*n+i0] = 1
        columns.append((u, v, w))
        
        # M4 = a22(b21-b11)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i0+1)*m+(j0+1)] = 1
        v[(j0+1)*p+k0] = 1; v[j0*p+k0] = -1
        w[k0*n+i0] = 1; w[(k0+1)*n+(i0+1)] = 1
        columns.append((u, v, w))
        
        # M5 = (a11+a12)b22
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m+j0] = 1; u[i0*m+(j0+1)] = 1
        v[(j0+1)*p+(k0+1)] = 1
        w[k0*n+i0] = -1; w[(k0+1)*n+(i0+1)] = 1
        columns.append((u, v, w))
        
        # M6 = (a21-a11)(b11+b12)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i0+1)*m+j0] = 1; u[i0*m+j0] = -1
        v[j0*p+k0] = 1; v[j0*p+(k0+1)] = 1
        w[(k0+1)*n+(i0+1)] = 1
        columns.append((u, v, w))
        
        # M7 = (a12-a22)(b21+b22)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m+(j0+1)] = 1; u[(i0+1)*m+(j0+1)] = -1
        v[(j0+1)*p+k0] = 1; v[(j0+1)*p+(k0+1)] = 1
        w[k0*n+i0] = 1; w[(k0+1)*n+(i0+1)] = -1
        columns.append((u, v, w))
    
    def add_schoolbook_2x2_2x1(i0, j0, k0):
        """Add schoolbook for 2x2 A block at (i0,j0), 2x1 B block at (j0,k0)."""
        # c11 = a11*b11 + a12*b21
        # c21 = a21*b11 + a22*b21
        for (ai, aj, bi, bj, ci, ck) in [
            (0, 0, 0, 0, 0, 0),  # a11*b11 -> c11
            (0, 1, 1, 0, 0, 0),  # a12*b21 -> c11
            (1, 0, 0, 0, 1, 0),  # a21*b11 -> c21
            (1, 1, 1, 0, 1, 0),  # a22*b21 -> c21
        ]:
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[(i0+ai)*m+(j0+aj)] = 1
            v[(j0+bi)*p+(k0+bj)] = 1
            w[(k0+ck)*n+(i0+ci)] = 1
            columns.append((u, v, w))
    
    # A is 2x4, B is 4x7
    # Split A into A1 (cols 0-1) and A2 (cols 2-3)
    # Split B into B1 (rows 0-1) and B2 (rows 2-3)
    # For each (A_block, B_block) pair, split B_block into 2x2 and 2x1 sub-blocks
    
    # A1*B1: A1 is rows 0-1, cols 0-1 of A; B1 is rows 0-1, cols 0-6 of B
    # B1 sub-blocks: (k=0,1), (k=2,3), (k=4,5), (k=6)
    for k0 in [0, 2, 4]:
        add_strassen_2x2_2x2(0, 0, k0)  # A1 (i=0,1; j=0,1) x B1 (j=0,1; k=k0,k0+1)
    add_schoolbook_2x2_2x1(0, 0, 6)  # A1 x B1 (j=0,1; k=6)
    
    # A1*B2: A1 is rows 0-1, cols 0-1 of A; B2 is rows 2-3, cols 0-6 of B
    for k0 in [0, 2, 4]:
        add_strassen_2x2_2x2(0, 2, k0)  # A1 (i=0,1; j=0,1) x B2 (j=2,3; k=k0,k0+1)
    add_schoolbook_2x2_2x1(0, 2, 6)  # A1 x B2 (j=2,3; k=6)
    
    # A2*B1: A2 is rows 0-1, cols 2-3 of A; B1 is rows 0-1, cols 0-6 of B
    for k0 in [0, 2, 4]:
        add_strassen_2x2_2x2(0, 2, k0)  # Wait, this is wrong!
    
    # I'm confusing myself. Let me redo this.
    
    # A[i,j] for i in {0,1}, j in {0,1,2,3}
    # B[j,k] for j in {0,1,2,3}, k in {0,...,6}
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Split j into {0,1} and {2,3}
    # C[i,k] = sum_{j in {0,1}} A[i,j] B[j,k] + sum_{j in {2,3}} A[i,j] B[j,k]
    
    # For the first sum, A sub-block is A[:, 0:2] (2x2), B sub-block is B[0:2, :] (2x7)
    # For the second sum, A sub-block is A[:, 2:4] (2x2), B sub-block is B[2:4, :] (2x7)
    
    # For each, split the 2x7 B into 2x2 blocks and one 2x1 block
    
    columns = []  # Reset
    
    # First j-block: j in {0,1}
    # A sub-block: rows 0-1, cols 0-1 (i0=0, j0=0)
    # B sub-blocks: rows 0-1, cols 0-1, 2-3, 4-5, 6
    for k0 in [0, 2, 4]:
        add_strassen_2x2_2x2(0, 0, k0)
    add_schoolbook_2x2_2x1(0, 0, 6)
    
    # Second j-block: j in {2,3}
    # A sub-block: rows 0-1, cols 2-3 (i0=0, j0=2)
    # B sub-blocks: rows 2-3, cols 0-1, 2-3, 4-5, 6
    for k0 in [0, 2, 4]:
        add_strassen_2x2_2x2(0, 2, k0)
    add_schoolbook_2x2_2x1(0, 2, 6)
    
    rank = len(columns)
    
    # Build U, V, W
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r, (u, v, w) in enumerate(columns):
        for i in range(n*m):
            if u[i] != 0:
                U[i][r] = u[i]
        for j in range(m*p):
            if v[j] != 0:
                V[j][r] = v[j]
        for k in range(p*n):
            if w[k] != 0:
                W[k][r] = w[k]
    
    return {"U": U, "V": V, "W": W}
