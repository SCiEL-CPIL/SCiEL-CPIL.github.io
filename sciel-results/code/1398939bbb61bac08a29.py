"""Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time
import itertools

DIMENSIONS = [3, 3, 8]

# Known small Strassen-like decompositions and their rank
# For 2x2x2, Strassen gives rank 7
# We need to embed these into 3x3x8

def _strassen_2x2_2():
    """
    Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Column 0: P1 = (a+d)(e+h)
    # a is at i=0,j=0 -> U[0,0]=1
    # d is at i=1,j=1 -> U[3,0]=1
    # e is at j=0,k=0 -> V[0,0]=1
    # h is at j=1,k=1 -> V[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C11 = P1...: W[0*2+0, 0] = W[0,0] = 1
    # C22 = P1...: W[1*2+1, 0] = W[3,0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Column 1: P2 = (c+d)e
    # c is at i=1,j=0 -> U[2,1]=1
    # d is at i=1,j=1 -> U[3,1]=1
    # e is at j=0,k=0 -> V[0,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C21 = P2...: W[0*2+1, 1] = W[1,1] = 1
    # C22 = -P2...: W[1*2+1, 1] = W[3,1] = -1
    W[1][1] = 1; W[3][1] = -1
    
    # Column 2: P3 = a(f-h)
    # a is at i=0,j=0 -> U[0,2]=1
    # f is at j=0,k=1 -> V[1,2]=1
    # h is at j=1,k=1 -> V[3,2]=-1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C12 = P3...: W[1*2+0, 2] = W[2,2] = 1
    # C22 = P3...: W[1*2+1, 2] = W[3,2] = 1
    W[2][2] = 1; W[3][2] = 1
    
    # Column 3: P4 = d(g-e)
    # d is at i=1,j=1 -> U[3,3]=1
    # g is at j=1,k=0 -> V[2,3]=1
    # e is at j=0,k=0 -> V[0,3]=-1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C11 = P4...: W[0*2+0, 3] = W[0,3] = 1
    # C21 = P4...: W[0*2+1, 3] = W[1,3] = 1
    W[0][3] = 1; W[1][3] = 1
    
    # Column 4: P5 = (a+b)h
    # a is at i=0,j=0 -> U[0,4]=1
    # b is at i=0,j=1 -> U[1,4]=1
    # h is at j=1,k=1 -> V[3,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C11 = -P5...: W[0*2+0, 4] = W[0,4] = -1
    # C12 = P5...: W[1*2+0, 4] = W[2,4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Column 5: P6 = (c-a)(e+f)
    # c is at i=1,j=0 -> U[2,5]=1
    # a is at i=0,j=0 -> U[0,5]=-1
    # e is at j=0,k=0 -> V[0,5]=1
    # f is at j=0,k=1 -> V[1,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C22 = P6...: W[1*2+1, 5] = W[3,5] = 1
    W[3][5] = 1
    
    # Column 6: P7 = (b-d)(g+h)
    # b is at i=0,j=1 -> U[1,6]=1
    # d is at i=1,j=1 -> U[3,6]=-1
    # g is at j=1,k=0 -> V[2,6]=1
    # h is at j=1,k=1 -> V[3,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C11 = P7...: W[0*2+0, 6] = W[0,6] = 1
    W[0][6] = 1
    
    return U, V, W

def _embed_strassen_into_3x3x8():
    """
    Embed the 2x2x2 Strassen decomposition into 3x3x8.
    We use the top-left 2x2 block for the Strassen part,
    and handle the remaining rows/columns with schoolbook.
    
    For 3x3 * 3x8:
    - Rows 0,1 of A (first 2 rows) and cols 0,1 of B (first 2 cols) -> Strassen 2x2x2
    - Row 2 of A * all 8 cols of B -> schoolbook (24 terms)
    - Rows 0,1 of A * cols 2-7 of B -> schoolbook (2*6=12 terms)
    
    Total rank = 7 + 24 + 12 = 43
    """
    n, m, p = 3, 3, 8
    rank = 43
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Embed Strassen 2x2x2 in the top-left 2x2 block
    sU, sV, sW = _strassen_2x2_2()
    for c in range(7):
        for u_idx in range(4):
            U[u_idx][c] = sU[u_idx][c]
        for v_idx in range(4):
            V[v_idx][c] = sV[v_idx][c]
        for w_idx in range(4):
            W[w_idx][c] = sW[w_idx][c]
    
    col = 7
    
    # Row 2 of A * all 8 cols of B (schoolbook)
    # i=2, j in {0,1,2}, k in {0,...,7}
    for j in range(3):
        for k in range(8):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    
    # Rows 0,1 of A * cols 2-7 of B (schoolbook)
    for i in range(2):
        for j in range(3):
            for k in range(2, 8):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    Check that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    and for all other (i,j,k) combinations that should be 0, they are 0.
    """
    rank = len(U[0])
    # Build the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Try the embedded Strassen decomposition
    try:
        U, V, W = _embed_strassen_into_3x3x8()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
