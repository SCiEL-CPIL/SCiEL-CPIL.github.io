"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return Strassen's 7-rank decomposition for 2x2 * 2x2.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    where i,k in {0,1} (output row/col), j in {0,1} (inner dim).
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U[i*2+j, c]: A_{i,j} coefficient for product c
    # V[j*2+k, c]: B_{j,k} coefficient for product c
    # W[k*2+i, c]: C_{i,k} coefficient (note: W index is k*n+i = k*2+i)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00 (k=0,i=0)
    W[3][0] = 1  # C11 (k=1,i=1)
    
    # M2: (A10+A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10 (k=0,i=1)
    
    # M3: A00(B01-B11)
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1 # B11
    W[1][2] = 1  # C01 (k=1,i=0)
    
    # M4: A11(B10-B00)
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1 # B00
    W[0][3] = -1 # C00 (k=0,i=0)
    W[3][3] = 1  # C11 (k=1,i=1)
    
    # M5: (A00+A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[0][4] = 1  # C00 (k=0,i=0)
    W[1][4] = 1  # C01 (k=1,i=0)
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    W[3][5] = 1  # C11 (k=1,i=1)
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00 (k=0,i=0)
    
    return U, V, W

def _build_strassen_based_2x4x5():
    """
    Build a decomposition of 2x4 * 4x5 using Strassen on 2x2 blocks.
    
    A is 2x4: split columns into two 2x2 blocks: A = [A_left | A_right]
    B is 4x5: split rows into two 2x2 blocks: B = [B_top; B_bottom]
    
    C = A*B = A_left * B_top + A_right * B_bottom
    
    Each 2x2 * 2x2 product has rank 7 (Strassen), so total rank = 14.
    But we need to handle the 2x4 and 4x5 dimensions properly.
    
    Actually, let's think more carefully.
    A is 2x4, B is 4x5.
    Split A into A0 (2x2, cols 0-1) and A1 (2x2, cols 2-3).
    Split B into B0 (2x5, rows 0-1) and B1 (2x5, rows 2-3).
    C = A0*B0 + A1*B1.
    
    Now A0*B0 is 2x2 * 2x5. We can further split B0 into B00 (2x2, cols 0-1) and B01 (2x3, cols 2-4).
    This gets complicated. Let me use a simpler block approach.
    
    Alternative: Split the 4-dimension into two 2-dimensions.
    A = [A0 | A1] where A0, A1 are 2x2.
    B = [B0; B1] where B0, B1 are 2x5.
    C = A0*B0 + A1*B1.
    
    For A0*B0 (2x2 * 2x5): split B0 into B00 (2x2) and B01 (2x3).
    A0*B0 = A0*B00 + A0*B01.
    A0*B00 is 2x2*2x2, rank 7.
    A0*B01 is 2x2*2x3. Split B01 into B010 (2x1) and B011 (2x2)? Or just use schoolbook for the 2x2*2x3 part.
    
    This is getting messy. Let me just use the direct approach:
    
    For 2x4 * 4x5:
    - Split the inner dimension 4 into 2+2.
    - A = [A0 | A1], A0,A1 in R^{2x2}
    - B = [B0; B1], B0,B1 in R^{2x5}
    - C = A0*B0 + A1*B1
    
    Now A0*B0 is 2x2 * 2x5. We can write this as:
    Split B0 into B0[:,0:2] (2x2) and B0[:,2:5] (2x3).
    A0*B0 = A0*B0a + A0*B0b where B0a is 2x2, B0b is 2x3.
    A0*B0a: 2x2*2x2, rank 7 (Strassen)
    A0*B0b: 2x2*2x3, rank 6 (schoolbook) or can we do better?
    
    Similarly for A1*B1.
    
    Total rank = 7 + 6 + 7 + 6 = 26? That's worse than 40? No, 26 < 40.
    
    Actually wait, let me reconsider. The known best for 2x4*4x5...
    
    Let me just use a straightforward block decomposition with Strassen where possible.
    
    Simpler approach: Use Strassen on 2x2 blocks for the 4x4 part, and handle the extra column separately.
    
    Actually, let me just implement a clean block decomposition:
    
    A (2x4) = [A00 A01] where A00, A01 are 2x2
    B (4x5) = [B00 B01; B10 B11] where each block is 2x2 or 2x3... 
    
    Hmm, 5 doesn't split evenly into 2s. Let me do:
    A (2x4) = [A0 A1], A0,A1 in R^{2x2}
    B (4x5): rows 0,1 form B0 (2x5), rows 2,3 form B1 (2x5)
    C = A0*B0 + A1*B1
    
    For A0*B0 (2x2 * 2x5):
    B0 = [B0_0 B0_1 B0_2 B0_3 B0_4] (five 2x1 columns)
    We can group as B0 = [B0_a | B0_b] where B0_a is 2x2 (cols 0,1) and B0_b is 2x3 (cols 2,3,4)
    A0*B0 = A0*B0_a + A0*B0_b
    A0*B0_a: 2x2*2x2, rank 7
    A0*B0_b: 2x2*2x3, rank 6 (schoolbook)
    
    Similarly A1*B1: rank 7 + 6 = 13
    
    Total: 7+6+7+6 = 26
    
    That gives rank 26. Let me implement this.
    """
    # We'll build U (8 x R), V (20 x R), W (10 x R)
    # n=2, m=4, p=5
    # U: (n*m, R) = (8, R)
    # V: (m*p, R) = (20, R)  
    # W: (p*n, R) = (10, R)
    
    n, m, p = 2, 4, 5
    
    # We'll collect terms: each term is (u_vec, v_vec, w_vec)
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    # Strassen 2x2 decomposition
    def strassen_2x2_offset(a_row_off, b_col_off, c_row_off):
        """
        Add Strassen terms for a 2x2 * 2x2 block.
        a_row_off: offset for A rows (A is indexed by row, col)
        b_col_off: offset for B cols (B is indexed by row, col)
        c_row_off: offset for C rows (C is indexed by row, col)
        """
        # A block is at rows a_row_off..a_row_off+1, cols 0..1 (within the 2x2 sub-problem)
        # B block is at rows 0..1, cols b_col_off..b_col_off+1
        # C block is at rows c_row_off..c_row_off+1, cols 0..1
        
        sU, sV, sW = _strassen_2x2()
        for c in range(7):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            for idx in range(4):
                if sU[idx][c] != 0:
                    u[idx + a_row_off * 2] = sU[idx][c]
                if sV[idx][c] != 0:
                    v[idx + b_col_off * 2] = sV[idx][c]
                if sW[idx][c] != 0:
                    w[idx + c_row_off * 2] = sW[idx][c]
            if any(x != 0 for x in u) and any(x != 0 for x in v) and any(x != 0 for x in w):
                add_term(u, v, w)
    
    def schoolbook_2x2x2x3_offset(a_row_off, b_col_off, c_row_off):
        """
        Schoolbook for 2x2 * 2x3.
        A is 2x2 (rows a_row_off..a_row_off+1, cols 0..1 in local)
        B is 2x3 (rows 0..1, cols b_col_off..b_col_off+2)
        C is 2x3 (rows c_row_off..c_row_off+1, cols 0..2)
        """
        for i in range(2):  # A row
            for j in range(2):  # inner
                for k in range(3):  # B col
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    u[i * 2 + j + a_row_off * 2] = 1
                    v[j * 3 + k + b_col_off * 3] = 1
                    w[k * 2 + i + c_row_off * 2] = 1
                    add_term(u, v, w)
    
    # Block structure:
    # A (2x4) = [A0 | A1], A0 is 2x2 (cols 0-1), A1 is 2x2 (cols 2-3)
    # B (4x5): B0 = rows 0-1 (2x5), B1 = rows 2-3 (2x5)
    # C = A0*B0 + A1*B1
    
    # A0*B0: A0 is 2x2, B0 is 2x5
    # Split B0 into B0a (2x2, cols 0-1) and B0b (2x3, cols 2-4)
    # A0*B0a: Strassen, rank 7
    strassen_2x2_offset(a_row_off=0, b_col_off=0, c_row_off=0)
    # A0*B0b: schoolbook 2x2*2x3, rank 6
    schoolbook_2x2x2x3_offset(a_row_off=0, b_col_off=2, c_row_off=0)
    
    # A1*B1: A1 is 2x2 (cols 2-3 of A), B1 is 2x5 (rows 2-3 of B)
    # Split B1 into B1a (2x2, cols 0-1) and B1b (2x3, cols 2-4)
    # A1*B1a: Strassen, rank 7
    strassen_2x2_offset(a_row_off=2, b_col_off=0, c_row_off=0)
    # A1*B1b: schoolbook 2x2*2x3, rank 6
    schoolbook_2x2x2x3_offset(a_row_off=2, b_col_off=2, c_row_off=0)
    
    R = len(terms)
    
    # Build final matrices
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            U[i][c] = u[i]
        for i in range(m * p):
            V[i][c] = v[i]
        for i in range(p * n):
            W[i][c] = w[i]
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, num_tests=20):
    """Verify the decomposition with random matrices."""
    import numpy as np
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Actually: C[i,k] = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # But W depends on (k,i), so:
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for c in range(rank):
                        total += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                C_computed[i, k] = total
        
        if not np.allclose(C_expected, C_computed, atol=1e-10):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with the Strassen-based decomposition (rank 26)
    best = _build_strassen_based_2x4x5()
    best_rank = len(best['U'][0])
    
    # Verify
    n, m, p = DIMENSIONS
    if _verify(best['U'], best['V'], best['W'], n, m, p):
        pass  # Good
    else:
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
    
    # Try to reduce rank by searching for linear dependencies
    # or using a smarter decomposition
    # For now, return the rank-26 decomposition
    
    return best

# EVOLVE-BLOCK-END
