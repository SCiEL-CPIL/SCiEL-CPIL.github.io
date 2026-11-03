"""Matrix multiplication 3,3,7 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]
START_TIME = time.time()
DEADLINE = START_TIME + 140

def construct():
    n, m, p = DIMENSIONS

    # We construct an explicit decomposition.
    # Strategy:
    # 1. Split B (3x7) into B_left (3x3) and B_right (3x4).
    # 2. Decompose A * B_left (3x3x3) using a known rank-21 decomposition (or similar).
    #    However, to be safe and simple, let's use a block decomposition of the 3x3x3 case
    #    that yields rank 21.
    #    Split A and B (both 3x3) into blocks:
    #    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    #    B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1.
    #    C = A*B.
    #    C11 = A11*B11 + A12*B21
    #    C12 = A11*B12 + A12*B22
    #    C21 = A21*B11 + A22*B21
    #    C22 = A21*B12 + A22*B22
    
    #    Terms:
    #    A11*B11 (2x2x2): Use Strassen (7 terms).
    #    A12*B21 (2x1x1): Schoolbook (2 terms).
    #    A11*B12 (2x2x1): Schoolbook (4 terms).
    #    A12*B22 (2x1x1): Schoolbook (2 terms).
    #    A21*B11 (1x2x2): Schoolbook (2 terms).
    #    A22*B21 (1x1x1): Schoolbook (1 term).
    #    A21*B12 (1x2x1): Schoolbook (2 terms).
    #    A22*B22 (1x1x1): Schoolbook (1 term).
    #    Total: 7 + 2 + 4 + 2 + 2 + 1 + 2 + 1 = 21.

    #    So for 3x3x3, we have rank 21.
    #    For 3x3x4 (B_right), we can just use schoolbook? 3*3*4 = 36.
    #    Total rank = 21 + 36 = 57.
    #    Is there a better way for 3x3x4?
    #    3x3x4 can be seen as 3x3x2 + 3x3x2.
    #    3x3x2: Rank?
    #    Split B (3x2) into two 3x1 columns.
    #    3x3x1 has rank 9 (schoolbook).
    #    Can we do better?
    #    Actually, let's just use the 21-term for the 3x3 part and schoolbook for the rest.
    #    Rank = 21 + 36 = 57.
    #    This is better than 63.

    # Let's implement the 21-term decomposition for 3x3x3.
    # Indices for A (3x3): 0..8 (row-major)
    #   A[0,0]=0, A[0,1]=1, A[0,2]=2
    #   A[1,0]=3, A[1,1]=4, A[1,2]=5
    #   A[2,0]=6, A[2,1]=7, A[2,2]=8
    
    # Indices for B (3x3): 0..8 (row-major)
    #   B[0,0]=0, B[0,1]=1, B[0,2]=2
    #   B[1,0]=3, B[1,1]=4, B[1,2]=5
    #   B[2,0]=6, B[2,1]=7, B[2,2]=8
    
    # Indices for C (3x3): 0..8 (row-major)
    #   C[0,0]=0, C[0,1]=1, C[0,2]=2
    #   C[1,0]=3, C[1,1]=4, C[1,2]=5
    #   C[2,0]=6, C[2,1]=7, C[2,2]=8

    # Blocks:
    # A11 (2x2): 0,1,3,4
    # A12 (2x1): 2,5
    # A21 (1x2): 6,7
    # A22 (1x1): 8
    
    # B11 (2x2): 0,1,3,4
    # B12 (2x1): 2,5
    # B21 (1x2): 6,7
    # B22 (1x1): 8
    
    # C11 (2x2): 0,1,3,4
    # C12 (2x1): 2,5
    # C21 (1x2): 6,7
    # C22 (1x1): 8

    rank = 57
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, val=1):
        nonlocal col
        for u in u_indices:
            U[u][col] += val
        for v in v_indices:
            V[v][col] += val
        for w in w_indices:
            W[w][col] += val
        col += 1

    # 1. A * B_left (3x3x3) -> Rank 21
    # Strassen for A11*B11 (2x2x2)
    # A11 = [[a00, a01], [a10, a11]] -> indices 0,1,3,4
    # B11 = [[b00, b01], [b10, b11]] -> indices 0,1,3,4
    # C11 = [[c00, c01], [c10, c11]] -> indices 0,1,3,4
    
    # M1 = (a10+a11)(b00+b01)
    # M2 = (a10+a11)(b10+b11)
    # M3 = (a00+a01)b11
    # M4 = (a10+a11)b00
    # M5 = a00(b01-b11)
    # M6 = a11(b10-b00)
    # M7 = (a00+a01)(b00-b10)
    
    # C00 = M1 - M4 - M5 + M7
    # C01 = M5 + M6
    # C10 = M2 + M3
    # C11 = M1 - M2 + M4 + M6

    # Note: We must map indices carefully.
    # a00=0, a01=1, a10=3, a11=4
    # b00=0, b01=1, b10=3, b11=4
    
    # M1
    add_term([3,4], [0,1], [0,4]) # C00 and C11 get +M1
    # M2
    add_term([3,4], [3,4], [3,4]) # C10 gets +M2, C11 gets -M2? 
    # Wait, C10 = M2+M3. C11 = M1-M2+M4+M6.
    # So for M2: W[3]+=1, W[4]-=1.
    # Let's restart add_term to allow signed weights or just add separate terms.
    # It is easier to just define the 7 terms and their contributions to C.
    
    # Term 1: M1
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[1][col] = 1
    W[0][col] = 1; W[4][col] = 1 # C00, C11
    col += 1
    
    # Term 2: M2
    U[3][col] = 1; U[4][col] = 1
    V[3][col] = 1; V[4][col] = 1
    W[3][col] = 1; W[4][col] = -1 # C10, C11
    col += 1
    
    # Term 3: M3
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[3][col] = 1 # C10
    col += 1
    
    # Term 4: M4
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[0][col] = -1; W[4][col] = 1 # C00, C11
    col += 1
    
    # Term 5: M5
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[0][col] = -1; W[1][col] = 1 # C00, C01
    col += 1
    
    # Term 6: M6
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[1][col] = 1; W[4][col] = 1 # C01, C11
    col += 1
    
    # Term 7: M7
    U[0][col] = 1; U[1][col] = 1
    V[0][col] = 1; V[3][col] = -1
    W[0][col] = 1 # C00
    col += 1
    
    # Remaining terms for 3x3x3:
    # A12*B21 (2x1x1) -> C12 (2x1)
    # A12 = [a02, a12] -> 2, 5
    # B21 = [b20, b21] -> 6, 7
    # C12 = [c02, c12] -> 2, 5
    # c02 = a02*b20 + a12*b21
    # c12 = a02*b21 + a12*b22 ? No, A12 is 2x1, B21 is 1x2.
    # A12 = [[a02], [a12]]. B21 = [b20, b21].
    # Product is 2x1? No, 2x1 * 1x2 is 2x2.
    # Wait, A is 3x3, B is 3x3.
    # A = [[A11, A12], [A21, A22]]
    # A11 (2x2), A12 (2x1)
    # A21 (1x2), A22 (1x1)
    # B = [[B11, B12], [B21, B22]]
    # B11 (2x2), B12 (2x1)
    # B21 (1x2), B22 (1x1)
    
    # C12 = A11*B12 + A12*B22
    # A11 (2x2) * B12 (2x1) -> 2x1.
    # A12 (2x1) * B22 (1x1) -> 2x1.
    
    # C21 = A21*B11 + A22*B21
    # A21 (1x2) * B11 (2x2) -> 1x2.
    # A22 (1x1) * B21 (1x2) -> 1x2.
    
    # C22 = A21*B12 + A22*B22
    # A21 (1x2) * B12 (2x1) -> 1x1.
    # A22 (1x1) * B22 (1x1) -> 1x1.

    # So we need:
    # 1. A11*B12 (2x2x1): Rank 4.
    # 2. A12*B22 (2x1x1): Rank 2.
    # 3. A21*B11 (1x2x2): Rank 2.
    # 4. A22*B21 (1x1x2): Rank 2.
    # 5. A21*B12 (1x2x1): Rank 2.
    # 6. A22*B22 (1x1x1): Rank 1.
    
    # Total 7 + 4 + 2 + 2 + 2 + 2 + 1 = 20?
    # Earlier I said 21. Let's re-verify Strassen part.
    # Strassen is 7.
    # A11*B12: 2x2x1. Schoolbook is 4.
    # A12*B22: 2x1x1. Schoolbook is 2.
    # A21*B11: 1x2x2. Schoolbook is 2.
    # A22*B21: 1x1x2. Schoolbook is 2.
    # A21*B12: 1x2x1. Schoolbook is 2.
    # A22*B22: 1x1x1. Schoolbook is 1.
    # Sum: 7 + 4 + 2 + 2 + 2 + 2 + 1 = 20.
    
    # Is 20 possible for 3x3x3?
    # Lower bound is 19.
    # Let's implement these.
    
    # A11*B12
    # A11: 0,1,3,4. B12: 2,5. C12: 2,5.
    # c02 = a00*b02 + a01*b12
    # c12 = a10*b02 + a11*b12
    # Term: a00*b02 -> C02
    U[0][col]=1; V[2][col]=1; W[2][col]=1; col+=1
    # Term: a01*b12 -> C02
    U[1][col]=1; V[5][col]=1; W[2][col]=1; col+=1
    # Term: a10*b02 -> C12
    U[3][col]=1; V[2][col]=1; W[5][col]=1; col+=1
    # Term: a11*b12 -> C12
    U[4][col]=1; V[5][col]=1; W[5][col]=1; col+=1
    
    # A12*B22
    # A12: 2,5. B22: 8. C12: 2,5.
    # c02 += a02*b22
    # c12 += a12*b22
    U[2][col]=1; V[8][col]=1; W[2][col]=1; col+=1
    U[5][col]=1; V[8][col]=1; W[5][col]=1; col+=1
    
    # A21*B11
    # A21: 6,7. B11: 0,1,3,4. C21: 6,7.
    # c20 = a20*b00 + a21*b10
    # c21 = a20*b01 + a21*b11
    # Term: a20*b00 -> C20
    U[6][col]=1; V[0][col]=1; W[6][col]=1; col+=1
    # Term: a20*b01 -> C21
    U[6][col]=1; V[1][col]=1; W[7][col]=1; col+=1
    # Wait, can we combine?
    # c20 = a20*b00 + a21*b10
    # c21 = a20*b01 + a21*b11
    # This is 1x2 * 2x2.
    # Terms: a20*b00, a21*b10, a20*b01, a21*b11.
    # 4 terms? No, I said 2.
    # 1x2x2 rank is 4?
    # Matrix multiplication 1x2 * 2x2 -> 1x2.
    # Output has 2 elements.
    # c0 = a0*b00 + a1*b10
    # c1 = a0*b01 + a1*b11
    # This is 4 products.
    # So rank is 4.
    # Similarly 2x2x1 is 4.
    # 2x1x1 is 2.
    # 1x2x1 is 2.
    # 1x1x2 is 2.
    # 1x1x1 is 1.
    
    # So:
    # A11*B12 (2x2x1): 4
    # A12*B22 (2x1x1): 2
    # A21*B11 (1x2x2): 4
    # A22*B21 (1x1x2): 2
    # A21*B12 (1x2x1): 2
    # A22*B22 (1x1x1): 1
    # Total: 7 + 4 + 2 + 4 + 2 + 2 + 1 = 22.
    
    # So rank 22 for 3x3x3.
    # Total rank = 22 + 36 = 58.
    
    # Let's finish the 22 terms.
    # A21*B11
    U[6][col]=1; V[0][col]=1; W[6][col]=1; col+=1 # a20*b00 -> C20
    U[7][col]=1; V[3][col]=1; W[6][col]=1; col+=1 # a21*b10 -> C20
    U[6][col]=1; V[1][col]=1; W[7][col]=1; col+=1 # a20*b01 -> C21
    U[7][col]=1; V[4][col]=1; W[7][col]=1; col+=1 # a21*b11 -> C21
    
    # A22*B21
    # A22: 8. B21: 6,7. C21: 6,7.
    # c20 += a22*b20
    # c21 += a22*b21
    U[8][col]=1; V[6][col]=1; W[6][col]=1; col+=1
    U[8][col]=1; V[7][col]=1; W[7][col]=1; col+=1
    
    # A21*B12
    # A21: 6,7. B12: 2,5. C22: 8.
    # c22 = a20*b02 + a21*b12
    U[6][col]=1; V[2][col]=1; W[8][col]=1; col+=1
    U[7][col]=1; V[5][col]=1; W[8][col]=1; col+=1
    
    # A22*B22
    # A22: 8. B22: 8. C22: 8.
    U[8][col]=1; V[8][col]=1; W[8][col]=1; col+=1
    
    # We have used 22 columns for the first 3x3x3 block.
    # Now for the remaining 4 columns of B (indices 3..6 in B).
    # We can use schoolbook for 3x3x4.
    # 3*3*4 = 36 terms.
    
    # B indices for right part:
    # B[0,3]=3, B[0,4]=4, B[0,5]=5, B[0,6]=6
    # B[1,3]=10, B[1,4]=11, B[1,5]=12, B[1,6]=13
    # B[2,3]=17, B[2,4]=18, B[2,5]=19, B[2,6]=20
    
    # C indices for right part:
    # C[0,3]=3, C[0,4]=4, C[0,5]=5, C[0,6]=6
    # C[1,3]=10, C[1,4]=11, C[1,5]=12, C[1,6]=13
    # C[2,3]=17, C[2,4]=18, C[2,5]=19, C[2,6]=20
    
    # Note: The V and W indices in the full problem are different from the local 3x3x3 indices.
    # In the full problem:
    # U index: i*m + j (0..8)
    # V index: j*p + k (0..20)
    # W index: k*n + i (0..20)
    
    # My previous 22 terms used local indices for V and W (0..8).
    # I need to remap them.
    # For the first block (k=0,1,2):
    # V local k -> V global k (since j*p + k, and j is 0,1,2. Wait, V index is j*p+k).
    # In the 3x3x3 block, B is 3x3.
    # B[j, k_local].
    # V index = j*p + k_local.
    # Since k_local is 0,1,2, and p=7.
    # V indices are:
    # j=0: 0, 1, 2
    # j=1: 7, 8, 9
    # j=2: 14, 15, 16
    
    # W local k -> W global.
    # W index = k_local*n + i.
    # Since k_local is 0,1,2, and n=3.
    # W indices are:
    # k=0: 0, 1, 2
    # k=1: 3, 4, 5
    # k=2: 6, 7, 8
    
    # So I need to map the V and W indices I used above.
    # My previous code used V indices 0..8 and W indices 0..8.
    # This is incorrect for the global indexing.
    
    # Let's restart the construction with correct global indices.
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    
    # Helper to get global V index
    def v_idx(j, k): return j*p + k
    # Helper to get global W index
    def w_idx(k, i): return k*n + i
    # U index is i*m + j
    
    # Block 1: 3x3x3 (k=0,1,2)
    # A11*B11 (Strassen)
    # a00=0, a01=1, a10=3, a11=4
    # b00=0, b01=7, b10=14, b11=15  (j*p+k: 0*7+0=0, 0*7+1=7? No. B[j,k]. j is row, k is col.
    # B is 3x7.
    # B[0,0]=0, B[0,1]=1, B[0,2]=2
    # B[1,0]=7, B[1,1]=8, B[1,2]=9
    # B[2,0]=14, B[2,1]=15, B[2,2]=16
    
    # So for B11 (2x2), j=0,1 and k=0,1.
    # b00 = v_idx(0,0) = 0
    # b01 = v_idx(0,1) = 1
    # b10 = v_idx(1,0) = 7
    # b11 = v_idx(1,1) = 8
    
    # C11 (2x2), k=0,1 and i=0,1.
    # c00 = w_idx(0,0) = 0
    # c01 = w_idx(0,1) = 1
    # c10 = w_idx(1,0) = 3
    # c11 = w_idx(1,1) = 4
    
    # M1
    U[3][col]=1; U[4][col]=1
    V[0][col]=1; V[1][col]=1
    W[0][col]=1; W[4][col]=1
    col+=1
    
    # M2
    U[3][col]=1; U[4][col]=1
    V[7][col]=1; V[8][col]=1
    W[3][col]=1; W[4][col]=-1
    col+=1
    
    # M3
    U[0][col]=1; U[1][col]=1
    V[8][col]=1
    W[3][col]=1
    col+=1
    
    # M4
    U[3][col]=1; U[4][col]=1
    V[0][col]=1
    W[0][col]=-1; W[4][col]=1
    col+=1
    
    # M5
    U[0][col]=1
    V[1][col]=1; V[8][col]=-1
    W[0][col]=-1; W[1][col]=1
    col+=1
    
    # M6
    U[4][col]=1
    V[7][col]=1; V[0][col]=-1
    W[1][col]=1; W[4][col]=1
    col+=1
    
    # M7
    U[0][col]=1; U[1][col]=1
    V[0][col]=1; V[7][col]=-1
    W[0][col]=1
    col+=1
    
    # A11*B12
    # A11: 0,1,3,4. B12: k=2. j=0,1.
    # b02 = v_idx(0,2) = 2
    # b12 = v_idx(1,2) = 9
    # C12: i=0,1. k=2.
    # c02 = w_idx(2,0) = 6
    # c12 = w_idx(2,1) = 7
    
    # c02 = a00*b02 + a01*b12
    # c12 = a10*b02 + a11*b12
    
    U[0][col]=1; V[2][col]=1; W[6][col]=1; col+=1
    U[1][col]=1; V[9][col]=1; W[6][col]=1; col+=1
    U[3][col]=1; V[2][col]=1; W[7][col]=1; col+=1
    U[4][col]=1; V[9][col]=1; W[7][col]=1; col+=1
    
    # A12*B22
    # A12: i=0,1. j=2.
    # a02 = u_idx(0,2) = 2
    # a12 = u_idx(1,2) = 5
    # B22: j=2. k=2.
    # b22 = v_idx(2,2) = 16
    # C12: i=0,1. k=2.
    # c02 = 6, c12 = 7
    
    U[2][col]=1; V[16][col]=1; W[6][col]=1; col+=1
    U[5][col]=1; V[16][col]=1; W[7][col]=1; col+=1
    
    # A21*B11
    # A21: i=2. j=0,1.
    # a20 = u_idx(2,0) = 6
    # a21 = u_idx(2,1) = 7
    # B11: j=0,1. k=0,1.
    # b00=0, b01=1, b10=7, b11=8
    # C21: i=2. k=0,1.
    # c20 = w_idx(0,2) = 2
    # c21 = w_idx(1,2) = 5
    
    # c20 = a20*b00 + a21*b10
    # c21 = a20*b01 + a21*b11
    
    U[6][col]=1; V[0][col]=1; W[2][col]=1; col+=1
    U[7][col]=1; V[7][col]=1; W[2][col]=1; col+=1
    U[6][col]=1; V[1][col]=1; W[5][col]=1; col+=1
    U[7][col]=1; V[8][col]=1; W[5][col]=1; col+=1
    
    # A22*B21
    # A22: i=2, j=2.
    # a22 = u_idx(2,2) = 8
    # B21: j=2. k=0,1.
    # b20 = v_idx(2,0) = 14
    # b21 = v_idx(2,1) = 15
    # C21: i=2. k=0,1.
    # c20 = 2, c21 = 5
    
    U[8][col]=1; V[14][col]=1; W[2][col]=1; col+=1
    U[8][col]=1; V[15][col]=1; W[5][col]=1; col+=1
    
    # A21*B12
    # A21: i=2. j=0,1.
    # a20=6, a21=7
    # B12: j=0,1. k=2.
    # b02=2, b12=9
    # C22: i=2. k=2.
    # c22 = w_idx(2,2) = 8
    
    # c22 = a20*b02 + a21*b12
    
    U[6][col]=1; V[2][col]=1; W[8][col]=1; col+=1
    U[7][col]=1; V[9][col]=1; W[8][col]=1; col+=1
    
    # A22*B22
    # a22=8
    # b22=16
    # c22=8
    
    U[8][col]=1; V[16][col]=1; W[8][col]=1; col+=1
    
    # We have used 22 columns.
    
    # Block 2: 3x3x4 (k=3,4,5,6)
    # Schoolbook.
    # i in 0..2, j in 0..2, k in 3..6
    
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
