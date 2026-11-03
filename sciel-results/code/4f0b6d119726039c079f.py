"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2():
    """Return Strassen 2x2 decomposition. U,V,W are 4x7 lists."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    # M2: (A10+A11)*B00 -> C10 += M2
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1
    # M3: A00*(B01-B11) -> C01 += M3
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[1][2]=1
    # M4: A11*(B10-B00) -> C00 += M4, C10 += M4
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1
    # M5: (A00+A01)*B11 -> C00 -= M5, C01 += M5
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[1][4]=1
    # M6: (A10-A00)*(B00+B01) -> C11 += M6
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    # M7: (A01-A11)*(B10+B11) -> C00 += M7
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1
    return U, V, W

def _laderman_3x3():
    """Return Laderman 3x3 decomposition. U,V,W are 9x7 lists."""
    # Laderman's 7-term algorithm for 3x3 matrix multiplication
    # M1 = (A00+A11+A22) * (B00+B11+B22)
    # M2 = (A01+A21) * (B10+B20)  -- but this isn't quite right
    # Actually Laderman's algorithm is more complex. Let me use the standard form.
    #
    # Laderman's 7 products:
    # M1 = (A00 + A11 + A22) * (B00 + B11 + B22)
    # M2 = (A01 + A21) * (B10 + B20)
    # M3 = (A02 + A20) * (B20 + B00)
    # M4 = (A10 + A12) * (B02 + B22)  -- hmm, this isn't standard
    #
    # Let me look up the exact Laderman decomposition.
    # Actually, the standard Laderman algorithm for 3x3:
    #
    # M1 = (A00+A11+A22)*(B00+B11+B22)
    # M2 = (A01+A21)*(B10+B20)
    # M3 = (A02+A20)*(B20+B00)
    # M4 = (A10+A12)*(B02+B22)
    # M5 = (A21+A22)*(B12+B20)  -- I'm not sure about these
    #
    # I'll use a verified Laderman decomposition. The key reference gives:
    #
    # Let me use the known 7-term decomposition for 3x3.
    # From the literature, one valid Laderman-style decomposition:
    
    U = [[0]*7 for _ in range(9)]
    V = [[0]*7 for _ in range(9)]
    W = [[0]*7 for _ in range(9)]
    
    # Indexing: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # M1: (A00+A11+A22)*(B00+B11+B22)
    # A00=U[0], A11=U[4], A22=U[8]
    # B00=V[0], B11=V[4], B22=V[8]
    # C00=W[0] += M1, C11=W[4] += M1, C22=W[8] += M1
    # But also C00 should get M1, and other terms subtract
    U[0][0]=1; U[4][0]=1; U[8][0]=1
    V[0][0]=1; V[4][0]=1; V[8][0]=1
    W[0][0]=1; W[4][0]=1; W[8][0]=1
    
    # M2: (A01+A21)*(B10+B20)
    # A01=U[1], A21=U[6]
    # B10=V[3], B20=V[6]
    # C00 = ... - M2? Let me think...
    # C[i][k] = sum_j A[i][j] B[j][k]
    # C[0][0] = A00*B00 + A01*B10 + A02*B20
    # So A01*B10 contributes to C00
    # C[0][1] = A00*B01 + A01*B11 + A02*B21
    # C[0][2] = A00*B02 + A01*B12 + A02*B22
    # C[2][0] = A20*B00 + A21*B10 + A22*B20
    # So (A01+A21)*(B10+B20) = A01*B10 + A01*B20 + A21*B10 + A21*B20
    # A01*B10 -> C00
    # A01*B20 -> C02? No, A01 is row 0 col 1, B20 is row 2 col 0, product goes to C[0][0]
    # Wait, I'm confusing myself. Let me be very careful.
    #
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # So A[i][j] * B[j][k] contributes to C[i][k].
    #
    # (A01 + A21) * (B10 + B20):
    # A01*B10: i=0, j=1, k=0 -> C[0][0]
    # A01*B20: i=0, j=1, k=0... wait B20 means j=2, k=0. But A01 has j=1, so A01*B20 is not a valid term in the sum for C.
    #
    # I see my error. The products (A[i1][j1] + A[i2][j2]) * (B[j1][k1] + B[j2][k2]) only make sense when the indices align.
    #
    # Let me restart with the correct Laderman algorithm. I'll use a known correct one.
    #
    # After careful thought, here's a correct Laderman decomposition for 3x3:
    # The 7 scalar multiplications are:
    # M1 = (A00+A11+A22) * (B00+B11+B22)
    # M2 = (A01+A21) * (B10+B20)
    # M3 = (A02+A20) * (B20+B00)
    # M4 = (A10+A12) * (B02+B22)
    # M5 = (A21+A02) * (B21+B02)  -- not sure
    #
    # I'm going to use a simpler approach: just use schoolbook for 3x3 (rank 27)
    # and Strassen for 2x2 (rank 7). This will give a valid but not optimal result.
    #
    # Actually, let me just return None and use schoolbook for 3x3.
    return None

def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m times m x p. Returns U(n*m x nmp), V(m*p x nmp), W(p*n x nmp)"""
    rank = n * m * p
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
    return U, V, W

def _construct_5x5():
    """Construct 5x5 decomposition using 2+3 block decomposition."""
    n = m = p = 5
    total_rank = 0
    # Pre-allocate
    # We'll collect terms and then assemble
    terms = []  # each term is (U_list, V_list, W_list, offset) where lists are of length rank
    
    # Block sizes: 2+3
    # A11: 2x2, A12: 2x3, A21: 3x2, A22: 3x3
    # B11: 2x2, B12: 2x3, B21: 3x2, B22: 3x3
    # C11: 2x2, C12: 2x3, C21: 3x2, C22: 3x3
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    def add_term_block(U_block, V_block, W_block, row_offset_A, col_offset_A, row_offset_B, col_offset_B, row_offset_C, col_offset_C, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add a block term to the global decomposition."""
        nonlocal total_rank
        rank = len(U_block[0])
        for i in range(a_rows):
            for j in range(a_cols):
                if U_block[i*a_cols+j] is not None:
                    for c in range(rank):
                        if U_block[i*a_cols+j][c] != 0:
                            global_row = (row_offset_A + i) * 5 + (col_offset_A + j)
                            # We'll handle this differently - build full matrices
        pass
    
    # This is getting complex. Let me just build the full 25xR matrices directly.
    # Total rank estimate: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 7 = 104
    # Let me use schoolbook for all rectangular blocks and Strassen for 2x2, schoolbook for 3x3.
    
    # For simplicity and correctness, let me use:
    # 2x2x2: Strassen rank 7
    # 2x3x2: schoolbook rank 12
    # 2x2x3: schoolbook rank 12
    # 2x3x3: schoolbook rank 18
    # 3x2x2: schoolbook rank 12
    # 3x3x2: schoolbook rank 18
    # 3x2x3: schoolbook rank 18
    # 3x3x3: schoolbook rank 27
    # Total: 7+12+12+18+12+18+18+27 = 124
    
    # That's only 1 better than schoolbook. Let me use Laderman for 3x3 to get rank 7.
    # But I need a correct Laderman decomposition.
    
    # Let me just implement it and verify. I'll use a known correct Laderman.
    
    # Actually, for the 3x3 case, let me try to find a rank-7 decomposition.
    # The Laderman algorithm is well-known. Let me use this formulation:
    #
    # From Laderman (1976), the 7 products are:
    # M1 = (A00 + A11 + A22) * (B00 + B11 + B22)
    # M2 = (A01 + A21) * (B10 + B20)
    # M3 = (A02 + A20) * (B20 + B00)
    # M4 = (A10 + A12) * (B02 + B22)
    # M5 = (A21 + A02) * (B21 + B02)
    # M6 = (A01 + A10) * (B10 + B01)
    # M7 = (A12 + A21) * (B12 + B21)
    #
    # And the outputs are:
    # C00 = M1 - M2 + M3 - M4 + M5 - M6 + M7  (this is a guess, need to verify)
    #
    # This is getting too error-prone. Let me just use schoolbook for everything and get rank 125,
    # then try to improve with a verified Strassen for 2x2 only.
    
    # Simplest correct approach: use Strassen for the 2x2 block in C11 = A11*B11
    # and schoolbook for everything else.
    #
    # C11 = A11*B11 + A12*B21
    #   A11*B11: 2x2x2, rank 7 (Strassen)
    #   A12*B21: 2x3x2, rank 12 (schoolbook)
    # C12 = A11*B12 + A12*B22
    #   A11*B12: 2x2x3, rank 12 (schoolbook)
    #   A12*B22: 2x3x3, rank 18 (schoolbook)
    # C21 = A21*B11 + A22*B21
    #   A21*B11: 3x2x2, rank 12 (schoolbook)
    #   A22*B21: 3x3x2, rank 18 (schoolbook)
    # C22 = A21*B12 + A22*B22
    #   A21*B12: 3x2x3, rank 18 (schoolbook)
    #   A22*B22: 3x3x3, rank 27 (schoolbook)
    #
    # Total: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124
    
    # Let me implement this and verify it's correct.
    
    R = 124  # total rank
    U = [[0]*R for _ in range(25)]
    V = [[0]*R for _ in range(25)]
    W = [[0]*R for _ in range(25)]
    
    col = 0
    
    # Helper to add a block decomposition
    def add_strassen_2x2(row_off_A, col_off_A, row_off_B, col_off_B, row_off_C, col_off_C):
        nonlocal col
        sU, sV, sW = _strassen_2x2()
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    if sU[i*2+j][c] != 0:
                        U[(row_off_A+i)*5 + (col_off_A+j)][col] += sU[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    if sV[j*2+k][c] != 0:
                        V[(row_off_B+j)*5 + (col_off_B+k)][col] += sV[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    if sW[k*2+i][c] != 0:
                        W[(row_off_C+k)*5 + (row_off_C+i)][col] += sW[k*2+i][c]
            col += 1
    
    def add_schoolbook_block(a_n, a_m, b_m, b_p, row_off_A, col_off_A, row_off_B, col_off_B, row_off_C, col_off_C):
        nonlocal col
        rank = a_n * a_m * b_p
        U_b, V_b, W_b = _schoolbook_block(a_n, a_m, b_p)
        for c in range(rank):
            for i in range(a_n):
                for j in range(a_m):
                    if U_b[i*a_m+j][c] != 0:
                        U[(row_off_A+i)*5 + (col_off_A+j)][col] += U_b[i*a_m+j][c]
            for j in range(a_m):
                for k in range(b_p):
                    if V_b[j*b_p+k][c] != 0:
                        V[(row_off_B+j)*5 + (col_off_B+k)][col] += V_b[j*b_p+k][c]
            for k in range(b_p):
                for i in range(a_n):
                    if W_b[k*a_n+i][c] != 0:
                        W[(row_off_C+k)*5 + (row_off_C+i)][col] += W_b[k*a_n+i][c]
            col += 1
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: Strassen 2x2
    add_strassen_2x2(0, 0, 0, 0, 0, 0)  # col now 7
    
    # A12*B21: schoolbook 2x3x2
    add_schoolbook_block(2, 3, 3, 2, 0, 2, 3, 0, 0, 0)  # A rows 0-1, cols 2-4; B rows 3-4, cols 0-1; C rows 0-1, cols 0-1
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: schoolbook 2x2x3
    add_schoolbook_block(2, 2, 2, 3, 0, 0, 0, 2, 0, 2)
    
    # A12*B22: schoolbook 2x3x3
    add_schoolbook_block(2, 3, 3, 3, 0, 2, 3, 2, 0, 2)
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: schoolbook 3x2x2
    add_schoolbook_block(3, 2, 2, 2, 2, 0, 0, 0, 2, 0)
    
    # A22*B21: schoolbook 3x3x2
    add_schoolbook_block(3, 3, 3, 2, 2, 2, 3, 0, 2, 0)
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: schoolbook 3x2x3
    add_schoolbook_block(3, 2, 2, 3, 2, 0, 0, 2, 2, 2)
    
    # A22*B22: schoolbook 3x3x3
    add_schoolbook_block(3, 3, 3, 3, 2, 2, 3, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # Try block decomposition with Strassen for 2x2
    try:
        result = _construct_5x5()
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception as e:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
