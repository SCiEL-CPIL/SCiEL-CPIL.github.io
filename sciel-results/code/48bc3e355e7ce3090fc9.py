"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _build_rank14():
    """
    Construct the rank-14 tensor decomposition for 2x2 * 2x4 matrix multiplication.
    
    The multiplication is decomposed into two independent 2x2 * 2x2 Strassen instances.
    The first instance computes C[:, 0:2] = A * B[:, 0:2].
    The second instance computes C[:, 2:4] = A * B[:, 2:4].
    
    Indices:
    A (4 elements): 0:A00, 1:A01, 2:A10, 3:A11
    B (8 elements): 0:B00, 1:B01, 2:B02, 3:B03, 4:B10, 5:B11, 6:B12, 7:B13
    C (8 elements, W factor): 
       W index k corresponds to C output element.
       Convention: W[k*n+i, r] where k is output column, i is output row.
       Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
       And "W has shape (p*n, rank)".
       p=4, n=2. Shape (8, rank).
       Index in W: k*n + i.
       k is column of C (0..3), i is row of C (0..1).
       
       C[0,0] -> k=0, i=0 -> idx 0
       C[1,0] -> k=0, i=1 -> idx 1
       C[0,1] -> k=1, i=0 -> idx 2
       C[1,1] -> k=1, i=1 -> idx 3
       C[0,2] -> k=2, i=0 -> idx 4
       C[1,2] -> k=2, i=1 -> idx 5
       C[0,3] -> k=3, i=0 -> idx 6
       C[1,3] -> k=3, i=1 -> idx 7
       
    U (4 elements, A factor):
       Index i*m + j. i=row A, j=col A.
       0:00, 1:01, 2:10, 3:11
       
    V (8 elements, B factor):
       Index j*p + k. j=row B, k=col B.
       0:00, 1:01, 2:02, 3:03, 4:10, 5:11, 6:12, 7:13
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Block 1: C[:, 0:2] = A * B[:, 0:2] ---
    # Strassen for 2x2 * 2x2
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1  # W[0] is C[0,0], W[3] is C[1,1] -> Wait.
    
    # Let's re-verify W indices carefully.
    # W index = k*n + i.
    # C[i, k] is the element at row i, col k.
    # C[0,0] -> i=0, k=0 -> idx 0.
    # C[1,0] -> i=1, k=0 -> idx 1.
    # C[0,1] -> i=0, k=1 -> idx 2.
    # C[1,1] -> i=1, k=1 -> idx 3.
    
    # Strassen outputs:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M2 + M7
    # C10 = M3 + M4
    # C11 = M1 - M2 + M3 - M5
    
    # Let's map Strassen terms to W indices.
    # M1 contributes to C00 (idx 0) and C11 (idx 3).
    # M2 contributes to C01 (idx 2) and C11 (idx 3).
    # M3 contributes to C10 (idx 1) and C11 (idx 3).
    # M4 contributes to C00 (idx 0) and C10 (idx 1).
    # M5 contributes to C00 (idx 0) and C11 (idx 3).
    # M6 contributes to C10 (idx 1).
    # M7 contributes to C00 (idx 0) and C01 (idx 2).
    
    # Re-evaluating standard Strassen mapping:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M2 + M7
    # C10 = M3 + M4
    # C11 = M1 - M2 + M3 - M5
    
    # Applying to W indices for Block 1 (cols 0,1):
    # W[0] (C00): M1(+1), M4(-1), M5(+1), M7(-1)
    # W[1] (C10): M3(+1), M4(+1)
    # W[2] (C01): M2(+1), M7(+1)
    # W[3] (C11): M1(+1), M2(-1), M3(+1), M5(-1)
    
    # Term 0 (M1):
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1 (M2):
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 2 (M3):
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 3 (M4):
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[1][3] = 1
    
    # Term 4 (M5):
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = 1; W[3][4] = -1
    
    # Term 5 (M6):
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1
    
    # Term 6 (M7):
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = -1; W[2][6] = 1
    
    # --- Block 2: C[:, 2:4] = A * B[:, 2:4] ---
    # Same A forms, different B forms.
    # B indices for block 2:
    # B02 (idx 2), B03 (idx 3), B12 (idx 6), B13 (idx 7)
    
    # W indices for Block 2 (cols 2,3):
    # W[4] (C02), W[5] (C12), W[6] (C03), W[7] (C13)
    
    # Term 7 (N1 = M1 equivalent):
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # Term 8 (N2 = M2 equivalent):
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[6][8] = 1; W[7][8] = -1
    
    # Term 9 (N3 = M3 equivalent):
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[5][9] = 1; W[7][9] = 1
    
    # Term 10 (N4 = M4 equivalent):
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = -1; W[5][10] = 1
    
    # Term 11 (N5 = M5 equivalent):
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = 1; W[7][11] = -1
    
    # Term 12 (N6 = M6 equivalent):
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[5][12] = 1
    
    # Term 13 (N7 = M7 equivalent):
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = -1; W[6][13] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
    The minimal rank is 14, achieved by two independent Strassen decompositions.
    """
    return _build_rank14()
# EVOLVE-BLOCK-END
