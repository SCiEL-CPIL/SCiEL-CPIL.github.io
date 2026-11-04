"""Find a rank-minimal exact tensor decomposition for 3x5 x 5x5 matrix multiplication."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
DEADLINE_SECONDS = 140.0

def _time_left():
    return DEADLINE_SECONDS - (time.time() - _START_TIME)

def _schoolbook(n, m, p):
    """Standard rank-nmp decomposition."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W, rank

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 x 2x2 matmul using rank-7 Strassen.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices:
    U: row = i*2 + j (i in 0..1, j in 0..1) -> A[i,j]
    V: row = j*2 + k (j in 0..1, k in 0..1) -> B[j,k]
    W: row = k*2 + i (k in 0..1, i in 0..1) -> C[i,k]
    
    Strassen formulas for C = A*B:
    Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-d)(f+g)
    M7 = (b-c)(e+h)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # r=0: M1 = (A00+A11)(B00+B11)
    # r=1: M2 = (A10+A11)B00
    # r=2: M3 = A00(B01-B11)
    # r=3: M4 = A11(B10-B00)
    # r=4: M5 = (A00+A01)B11
    # r=5: M6 = (A10-A11)(B01+B10)
    # r=6: M7 = (A01-A10)(B00+B11)
    
    U = [[Fraction(0)] * 7 for _ in range(4)]
    V = [[Fraction(0)] * 7 for _ in range(4)]
    W = [[Fraction(0)] * 7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1 # C00 and C11
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1 # C10
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1 # C01 and C11
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1 # C00 and C10
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1 # C00 and C01
    
    # M6
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[2][5] = 1
    W[3][5] = 1 # C11
    
    # M7
    U[1][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[3][6] = 1
    W[0][6] = 1 # C00
    
    return U, V, W

def _block_decompose_3x5x5():
    """
    Decompose 3x5 x 5x5 using block structure.
    Split A (3x5) into A1 (2x5) and A2 (1x5).
    Split B (5x5) into B1 (5x2) and B2 (5x3).
    
    C (3x5) = [C1; C2] where C1 (2x5), C2 (1x5).
    C1 = A1 * B = A1 * [B1, B2] = [A1*B1, A1*B2]
    C2 = A2 * B = A2 * [B1, B2] = [A2*B1, A2*B2]
    
    Blocks:
    1. A1(2x5) * B1(5x2) -> C11 (2x2). Use Strassen on 2x2 core? 
       A1 is 2x5, B1 is 5x2. This is not square.
       However, we can decompose the inner product.
       C11[i,k] = sum_j A1[i,j] B1[j,k].
       We can use a rank-11 decomposition for 2x5x2? 
       Or just schoolbook: 2*5*2 = 20.
       
       Let's look at the strategy hint: "isolate a 2x5x2 sub-tensor that can be decomposed using the rank-11 algorithm for 2x2x2 matrix multiplication applied to the 2x2 core with linear combinations for the remaining columns".
       
       Actually, let's just use schoolbook for the non-square blocks and Strassen for the 2x2x2 block if we can form one.
       
       Let's split dimensions differently to get square blocks.
       A (3x5), B (5x5).
       Split rows of A: 2 and 1.
       Split cols of B: 2 and 3.
       Split inner dim (cols of A, rows of B): 2 and 3.
       
       A = [[A11 (2x2), A12 (2x3)],
            [A21 (1x2), A22 (1x3)]]
       B = [[B11 (2x2), B12 (2x3)],
            [B21 (3x2), B22 (3x3)]]
       
       C = [[C11 (2x2), C12 (2x3)],
            [C21 (1x2), C22 (1x3)]]
       
       C11 = A11*B11 + A12*B21
       C12 = A11*B12 + A12*B22
       C21 = A21*B11 + A22*B21
       C22 = A21*B12 + A22*B22
       
       Block 1: A11 (2x2) * B11 (2x2) -> 2x2. Rank 7 (Strassen).
       Block 2: A12 (2x3) * B21 (3x2) -> 2x2. Rank 2*3*2 = 12 (Schoolbook).
       Block 3: A11 (2x2) * B12 (2x3) -> 2x3. Rank 2*2*3 = 12 (Schoolbook).
       Block 4: A12 (2x3) * B22 (3x3) -> 2x3. Rank 2*3*3 = 18 (Schoolbook).
       Block 5: A21 (1x2) * B11 (2x2) -> 1x2. Rank 1*2*2 = 4 (Schoolbook).
       Block 6: A22 (1x3) * B21 (3x2) -> 1x2. Rank 1*3*2 = 6 (Schoolbook).
       Block 7: A21 (1x2) * B12 (2x3) -> 1x3. Rank 1*2*3 = 6 (Schoolbook).
       Block 8: A22 (1x3) * B22 (3x3) -> 1x3. Rank 1*3*3 = 9 (Schoolbook).
       
       Total Rank = 7 + 12 + 12 + 18 + 4 + 6 + 6 + 9 = 74.
       Schoolbook is 75.
       This saves 1 rank.
       
       Can we do better?
       Block 2: 2x3 * 3x2. Can we use Strassen-like?
       Rank of 2x3x2 tensor.
       T[i,j,k] = 1. i in 0..1, j in 0..2, k in 0..1.
       Rank is at most 12. Is it lower?
       If we use rank 11, total is 73.
       
       Let's implement the rank 74 first, as it is guaranteed correct and simple.
       Then try to optimize Block 2 (2x3x2) and Block 5/6/7/8 if possible.
       
       For 2x3x2:
       C[i,k] = sum_j A[i,j] B[j,k].
       i=0,1; k=0,1; j=0,1,2.
       C00 = A00 B00 + A01 B10 + A02 B20
       C01 = A00 B01 + A01 B11 + A02 B21
       C10 = A10 B00 + A11 B10 + A12 B20
       C11 = A10 B01 + A11 B11 + A12 B21
       
       This is essentially 2 independent 1x3x2 problems? No, A and B are shared.
       But A is 2x3, B is 3x2.
       We can write:
       C00 = A00 B00 + A01 B10 + A02 B20
       C01 = A00 B01 + A01 B11 + A02 B21
       C10 = A10 B00 + A11 B10 + A12 B20
       C11 = A10 B01 + A11 B11 + A12 B21
       
       Let's try to find a rank-11 decomposition for 2x3x2.
       Or just use schoolbook 12.
       
       Let's implement the rank 74 block decomposition.
    """
    n, m, p = 3, 5, 5
    # Splits
    n1, n2 = 2, 1
    m1, m2 = 2, 3
    p1, p2 = 2, 3
    
    # Blocks:
    # 1: A11(n1,m1) * B11(m1,p1) -> C11(n1,p1) [2x2 * 2x2 -> 2x2] Rank 7
    # 2: A12(n1,m2) * B21(m2,p1) -> C11(n1,p1) [2x3 * 3x2 -> 2x2] Rank 12
    # 3: A11(n1,m1) * B12(m1,p2) -> C12(n1,p2) [2x2 * 2x3 -> 2x3] Rank 12
    # 4: A12(n1,m2) * B22(m2,p2) -> C12(n1,p2) [2x3 * 3x3 -> 2x3] Rank 18
    # 5: A21(n2,m1) * B11(m1,p1) -> C21(n2,p1) [1x2 * 2x2 -> 1x2] Rank 4
    # 6: A22(n2,m2) * B21(m2,p1) -> C21(n2,p1) [1x3 * 3x2 -> 1x2] Rank 6
    # 7: A21(n2,m1) * B12(m1,p2) -> C22(n2,p2) [1x2 * 2x3 -> 1x3] Rank 6
    # 8: A22(n2,m2) * B22(m2,p2) -> C22(n2,p2) [1x3 * 3x3 -> 1x3] Rank 9
    
    # Total Rank = 7 + 12 + 12 + 18 + 4 + 6 + 6 + 9 = 74
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(U_block, V_block, W_block, i_off, j_off, k_off):
        """
        Add a block decomposition.
        U_block shape: (n_b, m_b, rank_b)
        V_block shape: (m_b, p_b, rank_b)
        W_block shape: (p_b, n_b, rank_b)
        i_off, j_off, k_off are the starting indices for this block in the global tensor.
        """
        rank_b = U_block.shape[2]
        for r in range(rank_b):
            U_col = [Fraction(0)] * (n * m)
            V_col = [Fraction(0)] * (m * p)
            W_col = [Fraction(0)] * (p * n)
            
            for i in range(n_b):
                for j in range(m_b):
                    val = U_block[i, j, r]
                    if val != 0:
                        U_col[(i_off + i) * m + (j_off + j)] = Fraction(val)
                        
            for j in range(m_b):
                for k in range(p_b):
                    val = V_block[j, k, r]
                    if val != 0:
                        V_col[(j_off + j) * p + (k_off + k)] = Fraction(val)
                        
            for k in range(p_b):
                for i in range(n_b):
                    val = W_block[k, i, r]
                    if val != 0:
                        W_col[(k_off + k) * n + (i_off + i)] = Fraction(val)
                        
            U_list.append(U_col)
            V_list.append(V_col)
            W_list.append(W_col)
            
    # Block 1: 2x2 * 2x2 -> 2x2 (Strassen Rank 7)
    U1, V1, W1 = _strassen_2x2()
    # Convert to 3D arrays for the helper
    U1_3d = np.zeros((2, 2, 7))
    V1_3d = np.zeros((2, 2, 7))
    W1_3d = np.zeros((2, 2, 7))
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U1_3d[i, j, r] = U1[i*2+j][r]
                V1_3d[j, i, r] = 0 # Wait, V is (m, p). Here m=2, p=2.
                # In _strassen_2x2, V row is j*2+k. Here j is inner, k is col of B.
                # So V[j, k, r] = V1[j*2+k][r]
                # Let's re-map carefully.
                
    # Let's just use the helper with explicit mapping for Strassen.
    # Strassen U: (2,2,7), V: (2,2,7), W: (2,2,7)
    # U[i,j,r], V[j,k,r], W[k,i,r]
    for r in range(7):
        U_col = [Fraction(0)] * (n * m)
        V_col = [Fraction(0)] * (m * p)
        W_col = [Fraction(0)] * (p * n)
        
        # i_off=0, j_off=0, k_off=0
        for i in range(2):
            for j in range(2):
                U_col[i * m + j] = U1[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V_col[j * p + k] = V1[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W_col[k * n + i] = W1[k*2+i][r]
                
        U_list.append(U_col)
        V_list.append(V_col)
        W_list.append(W_col)
        
    # Helper for Schoolbook blocks
    def add_schoolblock(n_b, m_b, p_b, i_off, j_off, k_off):
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U_col = [Fraction(0)] * (n * m)
                    V_col = [Fraction(0)] * (m * p)
                    W_col = [Fraction(0)] * (p * n)
                    
                    U_col[(i_off + i) * m + (j_off + j)] = Fraction(1)
                    V_col[(j_off + j) * p + (k_off + k)] = Fraction(1)
                    W_col[(k_off + k) * n + (i_off + i)] = Fraction(1)
                    
                    U_list.append(U_col)
                    V_list.append(V_col)
                    W_list.append(W_col)
                    
    # Block 2: A12(2x3) * B21(3x2) -> C11(2x2)
    # i_off=0, j_off=2, k_off=0
    add_schoolblock(2, 3, 2, 0, 2, 0)
    
    # Block 3: A11(2x2) * B12(2x3) -> C12(2x3)
    # i_off=0, j_off=0, k_off=2
    add_schoolblock(2, 2, 3, 0, 0, 2)
    
    # Block 4: A12(2x3) * B22(3x3) -> C12(2x3)
    # i_off=0, j_off=2, k_off=2
    add_schoolblock(2, 3, 3, 0, 2, 2)
    
    # Block 5: A21(1x2) * B11(2x2) -> C21(1x2)
    # i_off=2, j_off=0, k_off=0
    add_schoolblock(1, 2, 2, 2, 0, 0)
    
    # Block 6: A22(1x3) * B21(3x2) -> C21(1x2)
    # i_off=2, j_off=2, k_off=0
    add_schoolblock(1, 3, 2, 2, 2, 0)
    
    # Block 7: A21(1x2) * B12(2x3) -> C22(1x3)
    # i_off=2, j_off=0, k_off=2
    add_schoolblock(1, 2, 3, 2, 0, 2)
    
    # Block 8: A22(1x3) * B22(3x3) -> C22(1x3)
    # i_off=2, j_off=2, k_off=2
    add_schoolblock(1, 3, 3, 2, 2, 2)
    
    rank = len(U_list)
    # Transpose to (rows, rank)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            U[i][r] = U_list[r][i]
        for i in range(m*p):
            V[i][r] = V_list[r][i]
        for i in range(p*n):
            W[i][r] = W_list[r][i]
            
    return U, V, W, rank

def _validate_fast(U, V, W, n, m, p):
    """Check decomposition validity."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    uv = U[i*m+j][r]
                    if uv == 0: continue
                    vw = V[j*p+k][r]
                    if vw == 0: continue
                    s += uv * vw * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try block decomposition (Rank 74)
    U, V, W, rank = _block_decompose_3x5x5()
    
    # Verify
    if not _validate_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W, rank = _schoolbook(n, m, p)
        
    # Convert to float for output (task says integer/half-integer, but float is usually accepted if exact)
    # The task says "integer/half-integer entries". My entries are integers.
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
