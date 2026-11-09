import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition
    # A is 2x5, B is 5x5
    # Split A into A1 (2x3) and A2 (2x2)
    # Split B into B11 (3x3), B12 (3x2), B21 (2x3), B22 (2x2)
    # C = A1*B11 + A2*B21
    
    # We will use the rank-14 Strassen algorithm for 3x3x3
    # and embed the smaller blocks into 3x3x3
    
    # Rank 14 decomposition for 3x3x3
    # This is the standard Strassen algorithm for 3x3 matrices
    # There are multiple rank-14 algorithms. One is based on 2x2 blocks.
    # Let's use a known rank-14 algorithm.
    
    # For simplicity and correctness, I will use the schoolbook for the 3x3 parts 
    # if I cannot recall the exact rank-14 coefficients, but that would be rank 27+27=54.
    # Wait, schoolbook for 3x3 is 27. 27+27=54 > 50.
    # So I MUST use a low-rank algorithm for 3x3.
    
    # Let's use the rank-14 algorithm for 3x3.
    # Reference: "A rank 14 algorithm for 3x3 matrix multiplication"
    
    # Since I don't have the explicit coefficients memorized for a specific rank-14 
    # algorithm that is easy to code from scratch without error, 
    # I will use the rank-14 algorithm derived from 2x2 Strassen blocks.
    
    # Actually, a simpler approach:
    # Use the rank-7 algorithm for 2x2x2.
    # Can we decompose 2x5x5 using 2x2 blocks?
    # 5 = 2 + 2 + 1.
    # A = [A1, A2, A3] (2x2, 2x2, 2x1)
    # B = [B11, B12, B13; B21, B22, B23; B31, B32, B33]
    # C = A1*B11 + A2*B21 + A3*B31
    # A1*B11: 2x2x2 (rank 7)
    # A2*B21: 2x2x2 (rank 7)
    # A3*B31: 2x1x2 (rank 4)
    # Total rank 7 + 7 + 4 = 18.
    
    # This is much better and easier to implement!
    # We need the rank-7 Strassen algorithm for 2x2x2.
    # And the rank-4 schoolbook for 2x1x2.
    
    rank = 7 + 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_2x2(a_offset, b_offset, c_offset, start_col):
        # A is 2x2, B is 2x2, C is 2x2
        # a_offset: (i, j) index of top-left of A block
        # b_offset: (k, l) index of top-left of B block
        # c_offset: (l, i) index of top-left of C block (note: W[k*n+i, r])
        
        # Strassen 7 multiplications:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        terms = [
            # (a_mult, a_row1, a_col1, a_row2, a_col2, b_mult, b_row1, b_col1, b_row2, b_col2, c_mult, c_row1, c_col1)
            # M1
            (1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0),
            (1, 0, 0, 1, 1, 1, 0, 0, 1, 1, -1, 0, 1),
            (1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0),
            (1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1),
            # M2
            (1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0),
            (1, 1, 0, 1, 1, 1, 0, 0, 0, 0, -1, 1, 1),
            # M3
            (1, 0, 0, 0, 0, 1, 0, 1, -1, 1, 1, 0, 1),
            # M4
            (1, 1, 1, 0, 0, 1, 1, 0, -1, 0, 1, 0, 0),
            (1, 1, 1, 0, 0, 1, 1, 0, -1, 0, 1, 1, 0),
            # M5
            (1, 0, 0, 1, 0, 1, 1, 1, 0, 0, -1, 0, 0),
            (1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1),
            # M6
            (1, 1, 0, -1, 0, 0, 1, 0, 0, 0, 1, 0, 0),
            (1, 1, 0, -1, 0, 0, 1, 0, 0, 0, 1, 1, 1),
            # M7
            (1, 0, 1, -1, 1, 1, 0, 0, 1, 0, 1, 0, 0),
            (1, 0, 1, -1, 1, 1, 0, 0, 1, 0, 1, 1, 1),
        ]
        
        # This manual expansion is error-prone.
        # Let's just use the standard Strassen formulas directly in the loop.
        
        # M1 = (A00+A11)(B00+B11)
        U[a_offset[0]*2 + 0][start_col] += 1
        U[a_offset[0]*2 + 1][start_col] += 1
        U[a_offset[1]*2 + 0][start_col] += 1
        U[a_offset[1]*2 + 1][start_col] += 1
        V[b_offset[0]*2 + 0][start_col] += 1
        V[b_offset[0]*2 + 1][start_col] += 1
        V[b_offset[1]*2 + 0][start_col] += 1
        V[b_offset[1]*2 + 1][start_col] += 1
        W[c_offset[0]*2 + 0][start_col] += 1
        W[c_offset[0]*2 + 1][start_col] -= 1
        W[c_offset[1]*2 + 0][start_col] += 1
        W[c_offset[1]*2 + 1][start_col] += 1
        start_col += 1
        
        # M2 = (A10+A11)B00
        U[a_offset[0]*2 + 1][start_col] += 1
        U[a_offset[1]*2 + 1][start_col] += 1
        V[b_offset[0]*2 + 0][start_col] += 1
        V[b_offset[0]*2 + 1][start_col] += 1
        W[c_offset[0]*2 + 1][start_col] += 1
        W[c_offset[1]*2 + 1][start_col] -= 1
        start_col += 1
        
        # M3 = A00(B01-B11)
        U[a_offset[0]*2 + 0][start_col] += 1
        V[b_offset[0]*2 + 1][start_col] += 1
        V[b_offset[1]*2 + 1][start_col] -= 1
        W[c_offset[0]*2 + 1][start_col] += 1
        start_col += 1
        
        # M4 = A11(B10-B00)
        U[a_offset[1]*2 + 1][start_col] += 1
        V[b_offset[1]*2 + 0][start_col] += 1
        V[b_offset[0]*2 + 0][start_col] -= 1
        W[c_offset[0]*2 + 0][start_col] += 1
        W[c_offset[1]*2 + 0][start_col] += 1
        start_col += 1
        
        # M5 = (A00+A01)B11
        U[a_offset[0]*2 + 0][start_col] += 1
        U[a_offset[0]*2 + 1][start_col] += 1
        V[b_offset[1]*2 + 1][start_col] += 1
        V[b_offset[1]*2 + 0][start_col] += 1
        W[c_offset[0]*2 + 0][start_col] -= 1
        W[c_offset[0]*2 + 1][start_col] += 1
        start_col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a_offset[0]*2 + 1][start_col] += 1
        U[a_offset[1]*2 + 0][start_col] -= 1
        V[b_offset[0]*2 + 0][start_col] += 1
        V[b_offset[0]*2 + 1][start_col] += 1
        W[c_offset[1]*2 + 0][start_col] += 1
        W[c_offset[1]*2 + 1][start_col] += 1
        start_col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_offset[0]*2 + 1][start_col] += 1
        U[a_offset[1]*2 + 1][start_col] -= 1
        V[b_offset[1]*2 + 0][start_col] += 1
        V[b_offset[1]*2 + 1][start_col] += 1
        W[c_offset[0]*2 + 0][start_col] += 1
        W[c_offset[1]*2 + 1][start_col] += 1
        start_col += 1
        
        return start_col

    # Helper to add a 2x1x2 schoolbook block
    def add_schoolbook_2x1_2(a_offset, b_offset, c_offset, start_col):
        # A is 2x1, B is 1x2, C is 2x2
        # C00 = A00*B00
        # C01 = A00*B01
        # C10 = A10*B00
        # C11 = A10*B01
        
        # A00*B00
        U[a_offset[0]*2 + 0][start_col] += 1
        V[b_offset[0]*2 + 0][start_col] += 1
        W[c_offset[0]*2 + 0][start_col] += 1
        start_col += 1
        
        # A00*B01
        U[a_offset[0]*2 + 0][start_col] += 1
        V[b_offset[0]*2 + 1][start_col] += 1
        W[c_offset[0]*2 + 1][start_col] += 1
        start_col += 1
        
        # A10*B00
        U[a_offset[1]*2 + 0][start_col] += 1
        V[b_offset[0]*2 + 0][start_col] += 1
        W[c_offset[1]*2 + 0][start_col] += 1
        start_col += 1
        
        # A10*B01
        U[a_offset[1]*2 + 0][start_col] += 1
        V[b_offset[0]*2 + 1][start_col] += 1
        W[c_offset[1]*2 + 1][start_col] += 1
        start_col += 1
        
        return start_col

    # Block 1: A1 (2x2) * B11 (2x2) -> C1 (2x2)
    # A1 is at A[0:2, 0:2]
    # B11 is at B[0:2, 0:2]
    # C1 is at C[0:2, 0:2]
    col = add_strassen_2x2((0, 0), (0, 0), (0, 0), col)
    
    # Block 2: A2 (2x2) * B21 (2x2) -> C2 (2x2)
    # A2 is at A[0:2, 2:4]
    # B21 is at B[2:4, 0:2]
    # C2 is at C[0:2, 2:4]
    col = add_strassen_2x2((0, 2), (2, 0), (0, 2), col)
    
    # Block 3: A3 (2x1) * B31 (1x2) -> C3 (2x2)
    # A3 is at A[0:2, 4:5]
    # B31 is at B[4:5, 0:2]
    # C3 is at C[0:2, 4:5]
    col = add_schoolbook_2x1_2((0, 4), (4, 0), (0, 4), col)
    
    return {"U": U, "V": V, "W": W}
