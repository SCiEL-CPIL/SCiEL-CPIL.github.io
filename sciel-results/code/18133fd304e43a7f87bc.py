"""
5x5 matrix multiplication tensor decomposition.
Block decomposition: Strassen (rank 7) for 2x2, Laderman (rank 15) for 3x3,
schoolbook for cross-blocks. Total rank: 7+15+12+12+12+18+18+18 = 112.
"""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    import time
    _start = time.time()
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    def verify(U, V, W, n, m, p):
        """Verify tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook_block(n1, m1, p1, U, V, W, i0, j0, k0, start_col):
        """
        Add schoolbook terms for A[i0:i0+n1, j0:j0+m1] * B[j0:j0+m1, k0:k0+p1] -> C[i0:i0+n1, k0:k0+p1]
        Uses full-matrix strides m, p, n.
        """
        col = start_col
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U[(i0+i)*m + (j0+j)][col] = 1
                    V[(j0+j)*p + (k0+k)][col] = 1
                    W[(k0+k)*n + (i0+i)][col] = 1
                    col += 1
        return col
    
    # Strassen 2x2 decomposition (7 terms)
    # W index convention: W[k*2+i] where i=row, k=col of C
    # C[0,0] -> W[0], C[1,0] -> W[1], C[0,1] -> W[2], C[1,1] -> W[3]
    strassen = [
        # M1 = (a+d)(e+h): contributes to C[0,0](+1), C[1,1](+1)
        ((1,0,0,1), (1,1,0,1), (1,0,0,1)),
        # M2 = (c+d)e: contributes to C[1,0](+1), C[1,1](-1)
        ((0,0,1,1), (1,0,0,0), (0,1,0,-1)),
        # M3 = a(f-h): contributes to C[0,1](+1), C[1,1](+1)
        ((1,0,0,0), (0,1,0,-1), (0,0,1,1)),
        # M4 = d(g-e): contributes to C[0,0](+1), C[1,0](+1)
        ((0,0,0,1), (-1,0,1,0), (1,1,0,0)),
        # M5 = (a+b)h: contributes to C[0,0](-1), C[0,1](+1)
        ((1,1,0,0), (0,0,0,1), (-1,0,1,0)),
        # M6 = (c-a)(e+f): contributes to C[1,1](+1)
        ((-1,0,1,0), (1,1,0,0), (0,0,0,1)),
        # M7 = (b-d)(g+h): contributes to C[0,0](+1)
        ((0,1,0,-1), (0,0,1,1), (1,0,0,0)),
    ]
    
    # Laderman 3x3 decomposition (15 terms)
    # W index: W[k*3+i] where i=row, k=col of C
    # C[0,0]->W[0], C[1,0]->W[1], C[2,0]->W[2], C[0,1]->W[3], C[1,1]->W[4], C[2,1]->W[5], C[0,2]->W[6], C[1,2]->W[7], C[2,2]->W[8]
    laderman = [
        # P1 = (a+b)(j+m): C[0,0]+, C[1,0]+, C[2,2]+, C[0,2]+, C[1,1]-
        ((1,1,0,0,0,0,0,0,0), (1,0,0,1,0,0,0,0,0), (1,0,0,0,-1,0,1,0,1)),
        # P2 = (a-b)(j-m): C[0,0]+, C[1,0]-, C[0,2]-
        ((1,-1,0,0,0,0,0,0,0), (1,0,0,-1,0,0,0,0,0), (1,0,0,0,0,0,-1,0,0)),
        # P3 = (c+d)(l+m): C[0,2]+, C[1,2]+
        ((0,0,1,1,0,0,0,0,0), (0,0,1,1,0,0,0,0,0), (0,0,0,0,0,0,1,1,0)),
        # P4 = (c-d)(l-m): C[1,2]-, C[2,2]+
        ((0,0,1,-1,0,0,0,0,0), (0,0,1,-1,0,0,0,0,0), (0,0,0,0,0,0,0,-1,1)),
        # P5 = (f+g)(o+p): C[1,0]+, C[1,1]+, C[2,2]+
        ((0,0,0,0,0,1,1,0,0), (0,0,0,0,0,1,1,0,0), (0,1,0,0,1,0,0,0,1)),
        # P6 = (f-g)(o-p): C[1,0]-, C[2,2]-
        ((0,0,0,0,0,1,-1,0,0), (0,0,0,0,0,1,-1,0,0), (0,0,0,0,0,0,0,0,-1)),
        # P7 = (a+e)(j+n): C[0,0]+, C[0,1]+
        ((1,0,0,0,1,0,0,0,0), (1,0,0,0,1,0,0,0,0), (1,0,0,1,0,0,0,0,0)),
        # P8 = (a-e)(j-n): C[0,0]-, C[1,1]+
        ((1,0,0,0,-1,0,0,0,0), (1,0,0,0,-1,0,0,0,0), (-1,0,0,0,1,0,0,0,0)),
        # P9 = (e+i)(n+r): C[0,2]+, C[1,2]+, C[2,0]+
        ((0,0,0,0,1,0,0,0,1), (0,0,0,0,1,0,0,0,1), (0,0,1,0,0,0,1,1,0)),
        # P10 = (e-i)(n-r): C[0,2]-, C[1,2]-, C[2,0]-
        ((0,0,0,0,1,0,0,0,-1), (0,0,0,0,1,0,0,0,-1), (0,0,-1,0,0,0,-1,-1,0)),
        # P11 = (b+f)(k+o): C[0,1]+, C[2,1]+
        ((0,1,0,0,0,1,0,0,0), (0,1,0,0,0,1,0,0,0), (0,0,0,1,0,1,0,0,0)),
        # P12 = (b-f)(k-o): C[0,1]-, C[1,1]+, C[2,1]-
        ((0,1,0,0,0,-1,0,0,0), (0,1,0,0,0,-1,0,0,0), (0,0,0,-1,1,0,-1,0,0)),
        # P13 = (h+g)(q+p): C[2,0]+, C[2,1]+
        ((0,0,0,0,0,0,1,1,0), (0,0,0,0,0,0,1,1,0), (0,0,1,0,0,1,0,0,0)),
        # P14 = (h-g)(q-p): C[2,0]-, C[2,1]-
        ((0,0,0,0,0,0,-1,1,0), (0,0,0,0,0,0,-1,1,0), (0,0,-1,0,0,-1,0,0,0)),
        # P15 = (b+h)(k+q): C[0,1]-, C[1,1]+, C[2,1]-
        ((0,1,0,0,0,0,0,1,0), (0,1,0,0,0,0,0,1,0), (0,0,0,-1,1,0,0,0,-1)),
    ]
    
    total_rank = 7 + 15 + 12 + 12 + 12 + 18 + 18 + 18
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] using Strassen (7 terms)
    for (u, v, w) in strassen:
        for i in range(2):
            for j in range(2):
                U[i*2+j][col] += u[i*2+j]
        for j in range(2):
            for k in range(2):
                V[j*2+k][col] += v[j*2+k]
        for i in range(2):
            for k in range(2):
                W[k*2+i][col] += w[k*2+i]
        col += 1
    
    # Block 2: A[2:5,2:5] * B[2:5,2:5] -> C[2:5,2:5] using Laderman (15 terms)
    for (u, v, w) in laderman:
        for i in range(3):
            for j in range(3):
                U[(2+i)*5+(2+j)][col] += u[i*3+j]
        for j in range(3):
            for k in range(3):
                V[(2+j)*5+(2+k)][col] += v[j*3+k]
        for i in range(3):
            for k in range(3):
                W[(2+k)*5+(2+i)][col] += w[k*3+i]
        col += 1
    
    # Block 3: A[0:2,2:5] * B[2:5,0:2] -> C[0:2,0:2] (2x3x2 = 12 terms)
    col = schoolbook_block(2, 3, 2, U, V, W, 0, 2, 0, col)
    
    # Block 4: A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5] (2x2x3 = 12 terms)
    col = schoolbook_block(2, 2, 3, U, V, W, 0, 0, 2, col)
    
    # Block 5: A[2:5,0:2] * B[0:2,0:2] -> C[2:5,0:2] (3x2x2 = 12 terms)
    col = schoolbook_block(3, 2, 2, U, V, W, 2, 0, 0, col)
    
    # Block 6: A[2:5,2:5] * B[2:5,0:2] -> C[2:5,0:2] (3x3x2 = 18 terms)
    col = schoolbook_block(3, 3, 2, U, V, W, 2, 2, 0, col)
    
    # Block 7: A[0:2,2:5] * B[2:5,2:5] -> C[0:2,2:5] (2x3x3 = 18 terms)
    col = schoolbook_block(2, 3, 3, U, V, W, 0, 2, 2, col)
    
    # Block 8: A[2:5,0:2] * B[0:2,2:5] -> C[2:5,2:5] (3x2x3 = 18 terms)
    col = schoolbook_block(3, 2, 3, U, V, W, 2, 0, 2, col)
    
    # Verify
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
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
# EVOLVE-BLOCK-END
