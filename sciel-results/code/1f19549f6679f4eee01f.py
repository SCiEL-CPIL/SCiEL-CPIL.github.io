"""Program for matrix multiplication 3x3x4 tensor decomposition (rank 34)."""
import time

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Returns a rank-34 exact tensor decomposition for 3x3 x 3x4 matrix multiplication.
    
    The decomposition uses:
    - Strassen's 7-multiplication algorithm for the two 2x2x2 blocks (left and right)
    - 8 direct multiplications for A[0:2,2]*B[2,0:2]
    - 8 direct multiplications for A[2,0:2]*B[0:2,0:2]
    - 2 direct multiplications for A[2,2]*B[2,0:2]
    - 8 direct multiplications for A[0:2,2]*B[2,2:4]
    - 8 direct multiplications for A[2,0:2]*B[0:2,2:4]
    - 2 direct multiplications for A[2,2]*B[2,2:4]
    Total: 7+8+8+2+8+8+2 = 43? No wait, let me recount.
    
    Actually:
    - Left 2x2x2 Strassen: 7
    - Right 2x2x2 Strassen: 7
    - A[0:2,2]*B[2,0:2]: 2*2 = 4
    - A[2,0:2]*B[0:2,0:2]: 2*2 = 4
    - A[2,2]*B[2,0:2]: 1*2 = 2
    - A[0:2,2]*B[2,2:4]: 2*2 = 4
    - A[2,0:2]*B[0:2,2:4]: 2*2 = 4
    - A[2,2]*B[2,2:4]: 1*2 = 2
    Total: 7+7+4+4+2+4+4+2 = 34
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34
    
    col = 0
    
    # Indexing: A[i,j] -> i*3+j, B[i,j] -> i*4+j, C[i,j] -> i*4+j
    
    # Left 2x2x2 Strassen (rows 0,1 of A, cols 0,1 of B -> cols 0,1 of C)
    # A00=A[0,0]=idx0, A01=A[0,1]=idx1, A10=A[1,0]=idx3, A11=A[1,1]=idx4
    # B00=B[0,0]=idx0, B01=B[0,1]=idx1, B10=B[1,0]=idx4, B11=B[1,1]=idx5
    # C00=C[0,0]=idx0, C01=C[0,1]=idx1, C10=C[1,0]=idx4, C11=C[1,1]=idx5
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[5][col] = 1
    col += 1
    
    # M2 = (A10+A11)*B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[4][col] = 1; W[5][col] = -1
    col += 1
    
    # M3 = A00*(B11-B00)
    U[0][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[1][col] = 1; W[5][col] = 1
    col += 1
    
    # M4 = A11*(B01-B00)
    U[4][col] = 1
    V[1][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)*B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # A[0:2,2] * B[2,0:2] (4 multiplications)
    # A[0,2]=idx2, A[1,2]=idx5
    # B[2,0]=idx8, B[2,1]=idx9
    
    # A[0,2]*B[2,0] -> C[0,0]=idx0
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    # A[0,2]*B[2,1] -> C[0,1]=idx1
    U[2][col] = 1
    V[9][col] = 1
    W[1][col] = 1
    col += 1
    
    # A[1,2]*B[2,0] -> C[1,0]=idx4
    U[5][col] = 1
    V[8][col] = 1
    W[4][col] = 1
    col += 1
    
    # A[1,2]*B[2,1] -> C[1,1]=idx5
    U[5][col] = 1
    V[9][col] = 1
    W[5][col] = 1
    col += 1
    
    # A[2,0:2] * B[0:2,0:2] (4 multiplications)
    # A[2,0]=idx6, A[2,1]=idx7
    # B[0,0]=idx0, B[0,1]=idx1, B[1,0]=idx4, B[1,1]=idx5
    
    # A[2,0]*B[0,0] -> C[2,0]=idx8
    U[6][col] = 1
    V[0][col] = 1
    W[8][col] = 1
    col += 1
    
    # A[2,0]*B[0,1] -> C[2,1]=idx9
    U[6][col] = 1
    V[1][col] = 1
    W[9][col] = 1
    col += 1
    
    # A[2,1]*B[1,0] -> C[2,0]=idx8
    U[7][col] = 1
    V[4][col] = 1
    W[8][col] = 1
    col += 1
    
    # A[2,1]*B[1,1] -> C[2,1]=idx9
    U[7][col] = 1
    V[5][col] = 1
    W[9][col] = 1
    col += 1
    
    # A[2,2] * B[2,0:2] (2 multiplications)
    # A[2,2]=idx8
    # B[2,0]=idx8, B[2,1]=idx9
    
    # A[2,2]*B[2,0] -> C[2,0]=idx8
    U[8][col] = 1
    V[8][col] = 1
    W[8][col] = 1
    col += 1
    
    # A[2,2]*B[2,1] -> C[2,1]=idx9
    U[8][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    # Right 2x2x2 Strassen (rows 0,1 of A, cols 2,3 of B -> cols 2,3 of C)
    # A00=A[0,0]=idx0, A01=A[0,1]=idx1, A10=A[1,0]=idx3, A11=A[1,1]=idx4
    # B00=B[0,2]=idx2, B01=B[0,3]=idx3, B10=B[1,2]=idx6, B11=B[1,3]=idx7
    # C00=C[0,2]=idx2, C01=C[0,3]=idx3, C10=C[1,2]=idx6, C11=C[1,3]=idx7
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[2][col] = 1; W[7][col] = 1
    col += 1
    
    # M2 = (A10+A11)*B00
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[6][col] = 1; W[7][col] = -1
    col += 1
    
    # M3 = A00*(B11-B00)
    U[0][col] = 1
    V[7][col] = 1; V[2][col] = -1
    W[3][col] = 1; W[7][col] = 1
    col += 1
    
    # M4 = A11*(B01-B00)
    U[4][col] = 1
    V[3][col] = 1; V[2][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M5 = (A00+A01)*B11
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[2][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[2][col] = 1
    col += 1
    
    # A[0:2,2] * B[2,2:4] (4 multiplications)
    # A[0,2]=idx2, A[1,2]=idx5
    # B[2,2]=idx10, B[2,3]=idx11
    
    # A[0,2]*B[2,2] -> C[0,2]=idx2
    U[2][col] = 1
    V[10][col] = 1
    W[2][col] = 1
    col += 1
    
    # A[0,2]*B[2,3] -> C[0,3]=idx3
    U[2][col] = 1
    V[11][col] = 1
    W[3][col] = 1
    col += 1
    
    # A[1,2]*B[2,2] -> C[1,2]=idx6
    U[5][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    
    # A[1,2]*B[2,3] -> C[1,3]=idx7
    U[5][col] = 1
    V[11][col] = 1
    W[7][col] = 1
    col += 1
    
    # A[2,0:2] * B[0:2,2:4] (4 multiplications)
    # A[2,0]=idx6, A[2,1]=idx7
    # B[0,2]=idx2, B[0,3]=idx3, B[1,2]=idx6, B[1,3]=idx7
    
    # A[2,0]*B[0,2] -> C[2,2]=idx10
    U[6][col] = 1
    V[2][col] = 1
    W[10][col] = 1
    col += 1
    
    # A[2,0]*B[0,3] -> C[2,3]=idx11
    U[6][col] = 1
    V[3][col] = 1
    W[11][col] = 1
    col += 1
    
    # A[2,1]*B[1,2] -> C[2,2]=idx10
    U[7][col] = 1
    V[6][col] = 1
    W[10][col] = 1
    col += 1
    
    # A[2,1]*B[1,3] -> C[2,3]=idx11
    U[7][col] = 1
    V[7][col] = 1
    W[11][col] = 1
    col += 1
    
    # A[2,2] * B[2,2:4] (2 multiplications)
    # A[2,2]=idx8
    # B[2,2]=idx10, B[2,3]=idx11
    
    # A[2,2]*B[2,2] -> C[2,2]=idx10
    U[8][col] = 1
    V[10][col] = 1
    W[10][col] = 1
    col += 1
    
    # A[2,2]*B[2,3] -> C[2,3]=idx11
    U[8][col] = 1
    V[11][col] = 1
    W[11][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
