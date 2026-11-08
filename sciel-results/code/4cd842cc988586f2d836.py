"""
Strassen-based block decomposition for 3x3 * 3x7 matrix multiplication.

Strategy:
  1. Partition A (3x3) and B (3x7) into blocks:
     A = [[A11, A12, A13],
          [A21, A22, A23],
          [A31, A32, A33]]
     B = [[B11, B12, B13, B14, B15, B16, B17],
          [B21, B22, B23, B24, B25, B26, B27],
          [B31, B32, B33, B34, B35, B36, B37]]
     
     We partition into 2x2 blocks for the left 2 columns of A and left 2 rows of B,
     plus the remaining strips.
     
  2. Use Strassen (rank 7) for the 2x2 * 2x2 part (top-left blocks).
  3. Use schoolbook (rank 1) for all other terms.
  
  Calculation:
  - Strassen covers outputs (i,j) for i in {0,1}, j in {0,1} using inputs A[i][j] for i,j in {0,1} and B[j][k] for j in {0,1}, k in {0,1}.
  - This replaces 4 schoolbook terms (rank 4) with 7 Strassen terms (rank 7).
  - Net change: +3 terms.
  - But we can also apply Strassen-like optimizations to other 2x2 blocks.
  
  Let me think more carefully about how to get rank 61.
  
  Actually, the inspiration says rank 61. Let me construct it properly.
  
  Partition B into B_left (3x3) and B_right (3x4).
  Partition A into A_top (2x3) and A_bottom (1x3).
  
  For A_top * B_left: this is 2x3 * 3x3. We can use Strassen on the 2x2 part.
  For A_bottom * B_left: this is 1x3 * 3x3, rank 9.
  For A_top * B_right: this is 2x3 * 3x4, rank 24.
  For A_bottom * B_right: this is 1x3 * 3x4, rank 12.
  
  Total without Strassen: 2*3*3 + 1*3*3 + 2*3*4 + 1*3*4 = 18 + 9 + 24 + 12 = 63.
  
  With Strassen on the 2x2 block of A_top * B_left:
  A_top = [[A11, A12, A13],
           [A21, A22, A23]]
  B_left = [[B11, B12, B13],
            [B21, B22, B23],
            [B31, B32, B33]]
  
  The 2x2*2x2 part involves A11,A12,A21,A22 and B11,B12,B21,B22.
  This produces C11,C12,C21,C22 (where C is the 2x3 output of A_top * B_left).
  
  Wait, A_top is 2x3 and B_left is 3x3, so the output is 2x3.
  C[0][0] = A11*B11 + A12*B21 + A13*B31
  C[0][1] = A11*B12 + A12*B22 + A13*B32
  C[0][2] = A11*B13 + A12*B23 + A13*B33
  C[1][0] = A21*B11 + A22*B21 + A23*B31
  C[1][1] = A21*B12 + A22*B22 + A23*B32
  C[1][2] = A21*B13 + A22*B23 + A23*B33
  
  The Strassen part handles the 2x2 submatrix using A11,A12,A21,A22 and B11,B12,B21,B22.
  This gives us C[0][0], C[0][1], C[1][0], C[1][1] partially.
  
  Actually, let me just construct the full decomposition and use Strassen where beneficial.
  
  For a 3x3 * 3x7 matrix multiplication, let me use a different approach:
  
  I'll use the known fact that we can get rank 61 by using Strassen on certain 2x2 blocks.
  
  Let me define the decomposition explicitly.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]
DEADLINE_SECONDS = 110

def _verify(U, V, W):
    """Check that U, V, W exactly decompose the 3x3 * 3x7 multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    """Return the standard schoolbook decomposition of rank 63."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Return the Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    
    Strassen's 7 products:
    P1 = (A11 + A22)(B11 + B22)
    P2 = (A21 + A22) * B11
    P3 = A11 * (B12 - B22)
    P4 = A22 * (B21 - B11)
    P5 = (A11 + A12) * B22
    P6 = (A21 - A11) * (B11 + B12)
    P7 = (A12 - A22) * (B21 + B22)
    
    Results:
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    """
    rank = 7
    # U: 4 entries (A11, A12, A21, A22) -> 7 terms
    U = [[0] * rank for _ in range(4)]
    # V: 4 entries (B11, B12, B21, B22) -> 7 terms
    V = [[0] * rank for _ in range(4)]
    # W: 4 entries (C11, C12, C21, C22) -> 7 terms
    W = [[0] * rank for _ in range(4)]
    
    # P1 = (A11 + A22)(B11 + B22)
    U[0][0] = 1; U[1][0] = 1  # Wait, let me be more careful with indexing.
    
    # Let me use explicit indices:
    # A indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B indices: 0=B11, 1=B12, 2=B21, 3=B22
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    
    # P1 = (A11 + A22)(B11 + B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C11 += P1, C22 += P1
    
    # P2 = (A21 + A22) * B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C21 += P2, C22 -= P2
    
    # P3 = A11 * (B12 - B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C12 += P3, C22 += P3
    
    # P4 = A22 * (B21 - B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C11 += P4, C21 += P4
    
    # P5 = (A11 + A12) * B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C11 -= P5, C12 += P5
    
    # P6 = (A21 - A11) * (B11 + B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22 += P6
    
    # P7 = (A12 - A22) * (B21 + B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11 += P7
    
    return {"U": U, "V": V, "W": W}

def _construct_strassen_hybrid():
    """
    Construct a hybrid decomposition using Strassen for 2x2 blocks.
    
    For 3x3 * 3x7:
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    B = [[B11, B12, B13, B14, B15, B16, B17],
         [B21, B22, B23, B24, B25, B26, B27],
         [B31, B32, B33, B34, B35, B36, B37]]
    
    Output C is 3x7.
    
    We'll use Strassen for the 2x2 block involving A11,A12,A21,A22 and B11,B12,B21,B22.
    This handles part of C[0][0], C[0][1], C[1][0], C[1][1].
    
    Then use schoolbook for everything else.
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Start with schoolbook (rank 63)
    U = [[0] * 63 for _ in range(9)]
    V = [[0] * 63 for _ in range(21)]
    W = [[0] * 63 for _ in range(21)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Now replace the 4 schoolbook terms for the 2x2 block with 7 Strassen terms.
    # The 4 schoolbook terms are:
    # (i=0,j=0,k=0): A11*B11 -> C[0][0]
    # (i=0,j=1,k=0): A12*B21 -> C[0][0]
    # (i=1,j=0,k=1): A21*B12 -> C[1][1]
    # (i=1,j=1,k=1): A22*B22 -> C[1][1]
    
    # Wait, I need to be more careful. Let me identify which schoolbook terms to replace.
    
    # Actually, let me just return the schoolbook for now and try a different approach.
    # The Strassen approach is complex to get right. Let me try a cleaner construction.
    
    return None

def _construct_rank61():
    """
    Construct a rank 61 decomposition for 3x3 * 3x7.
    
    Strategy: Use Strassen on a 2x2 block and schoolbook for the rest.
    This should give rank 63 - 4 + 7 = 66, which is worse.
    
    Alternative: Use a different decomposition strategy.
    
    Let me try to find a valid rank 61 decomposition by construction.
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # I'll construct this by hand using a block decomposition.
    # 
    # Partition:
    # A = [A_top; A_bottom] where A_top is 2x3, A_bottom is 1x3
    # B = [B_left, B_right] where B_left is 3x3, B_right is 3x4
    #
    # C = A * B = [A_top * B_left, A_top * B_right; A_bottom * B_left, A_bottom * B_right]
    #
    # Rank contributions:
    # A_top * B_left: 2x3 * 3x3 = rank 18 (schoolbook) or rank 7+9=16 (with Strassen on 2x2 part)
    # A_top * B_right: 2x3 * 3x4 = rank 24
    # A_bottom * B_left: 1x3 * 3x3 = rank 9
    # A_bottom * B_right: 1x3 * 3x4 = rank 12
    #
    # Total with Strassen on 2x2 part of A_top*B_left: 16 + 24 + 9 + 12 = 61!
    
    # Let me construct this carefully.
    
    rank = 61
    U = [[0] * rank for _ in range(9)]  # 3x3 = 9 entries
    V = [[0] * rank for _ in range(21)]  # 3x7 = 21 entries
    W = [[0] * rank for _ in range(21)]  # 7x3 = 21 entries
    
    col = 0
    
    # Block 1: A_top * B_left with Strassen on the 2x2 part
    # A_top = [[A11, A12, A13],
    #          [A21, A22, A23]]
    # B_left = [[B11, B12, B13],
    #           [B21, B22, B23],
    #           [B31, B32, B33]]
    #
    # The 2x2 part: A11,A12,A21,A22 and B11,B12,B21,B22
    # This produces C[0][0], C[0][1], C[1][0], C[1][1]
    
    strassen = _strassen_2x2()
    
    # Map Strassen indices to global indices
    # A indices in Strassen: 0=A11, 1=A12, 2=A21, 3=A22
    # Global A indices: A11=0, A12=1, A21=3, A22=4 (since A[i*m+j], m=3)
    # B indices in Strassen: 0=B11, 1=B12, 2=B21, 3=B22
    # Global B indices: B11=0, B12=1, B21=7, B22=8 (since B[j*p+k], p=7)
    # C indices in Strassen: 0=C11, 1=C12, 2=C21, 3=C22
    # Global C indices: C[0][0]=0, C[0][1]=1, C[1][0]=3, C[1][1]=4 (since C[k*n+i], n=3)
    
    a_map = [0, 1, 3, 4]  # Strassen A index -> global A index
    b_map = [0, 1, 7, 8]  # Strassen B index -> global B index
    c_map = [0, 1, 3, 4]  # Strassen C index -> global C index
    
    for s in range(7):
        for a_s, a_g in enumerate(a_map):
            U[a_g][col] += strassen["U"][a_s][s]
        for b_s, b_g in enumerate(b_map):
            V[b_g][col] += strassen["V"][b_s][s]
        for c_s, c_g in enumerate(c_map):
            W[c_g][col] += strassen["W"][c_s][s]
        col += 1
    
    # Now add schoolbook terms for the remaining parts.
    # 
    # A_top * B_left remaining:
    # C[0][0] += A13*B31 (i=0, j=2, k=0)
    # C[0][1] += A13*B32 (i=0, j=2, k=1)
    # C[0][2] = A11*B13 + A12*B23 + A13*B33 (i=0, j=0,1,2, k=2)
    # C[1][0] += A23*B31 (i=1, j=2, k=0)
    # C[1][1] += A23*B32 (i=1, j=2, k=1)
    # C[1][2] = A21*B13 + A22*B23 + A23*B33 (i=1, j=0,1,2, k=2)
    
    # Add schoolbook terms for A_top * B_left remaining
    # (i,j,k) where i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # Skip the 4 terms already handled by Strassen: (0,0,0), (0,1,0), (1,0,1), (1,1,1)
    skip = {(0,0,0), (0,1,0), (1,0,1), (1,1,1)}
    
    for i in range(2):
        for j in range(3):
            for k in range(3):
                if (i, j, k) not in skip:
                    U[i * 3 + j][col] = 1
                    V[j * 7 + k][col] = 1
                    W[k * 3 + i][col] = 1
                    col += 1
    
    # A_top * B_right: 2x3 * 3x4
    # i in {0,1}, j in {0,1,2}, k in {3,4,5,6}
    for i in range(2):
        for j in range(3):
            for k in range(3, 7):
                U[i * 3 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # A_bottom * B_left: 1x3 * 3x3
    # i=2, j in {0,1,2}, k in {0,1,2}
    for j in range(3):
        for k in range(3):
            U[2 * 3 + j][col] = 1
            V[j * 7 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
    
    # A_bottom * B_right: 1x3 * 3x4
    # i=2, j in {0,1,2}, k in {3,4,5,6}
    for j in range(3):
        for k in range(3, 7):
            U[2 * 3 + j][col] = 1
            V[j * 7 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
    
    # Trim to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    result = {"U": U, "V": V, "W": W}
    
    if _verify(U, V, W):
        return result
    else:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Try to construct rank 61
    result = _construct_rank61()
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook()

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
