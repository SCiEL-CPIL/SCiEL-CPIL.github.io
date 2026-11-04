"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 5
    # Rank calculation:
    # Block 1 (k=0,1): 7 (Strassen A1B1) + 7 (Strassen A2B2) + 4 (School A3B3) = 18
    # Block 2 (k=2,3): 7 (Strassen A1B2) + 7 (Strassen A2B2) + 4 (School A3B3) = 18
    # Block 3 (k=4):   4 (School A1B3)   + 4 (School A2B3)   + 2 (School A3B3) = 10
    # Total Rank = 46
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # A is 2x2 at A[ai_start:ai_start+2, aj_start:aj_start+2]
    # B is 2x2 at B[aj_start:aj_start+2, bk_start:bk_start+2]
    # C is 2x2 at C[bk_start:bk_start+2, ai_start:ai_start+2]
    def add_strassen(ai, aj, bk):
        nonlocal col
        # Strassen factors
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map these to U, V, W columns.
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        # M1
        U[ai*m + aj][col] += 1
        U[(ai+1)*m + (aj+1)][col] += 1
        V[aj*p + bk][col] += 1
        V[(aj+1)*p + (bk+1)][col] += 1
        W[bk*n + ai][col] += 1
        W[(bk+1)*n + (ai+1)][col] += 1
        col += 1
        
        # M2
        U[(ai+1)*m + aj][col] += 1
        U[(ai+1)*m + (aj+1)][col] += 1
        V[aj*p + bk][col] += 1
        W[(bk+1)*n + (ai+1)][col] += 1
        col += 1
        
        # M3
        U[ai*m + aj][col] += 1
        V[aj*p + (bk+1)][col] += 1
        V[(aj+1)*p + (bk+1)][col] -= 1
        W[bk*n + ai][col] += 1
        W[(bk+1)*n + ai][col] += 1
        col += 1
        
        # M4
        U[(ai+1)*m + (aj+1)][col] += 1
        V[(aj+1)*p + bk][col] += 1
        V[aj*p + bk][col] -= 1
        W[bk*n + ai][col] += 1
        W[(bk+1)*n + (ai+1)][col] += 1
        col += 1
        
        # M5
        U[ai*m + aj][col] += 1
        U[ai*m + (aj+1)][col] += 1
        V[(aj+1)*p + (bk+1)][col] += 1
        W[bk*n + ai][col] -= 1
        W[(bk+1)*n + ai][col] += 1
        col += 1
        
        # M6
        U[(ai+1)*m + aj][col] += 1
        U[ai*m + aj][col] -= 1
        V[aj*p + bk][col] += 1
        V[aj*p + (bk+1)][col] += 1
        W[(bk+1)*n + (ai+1)][col] += 1
        col += 1
        
        # M7
        U[ai*m + (aj+1)][col] += 1
        U[(ai+1)*m + (aj+1)][col] -= 1
        V[(aj+1)*p + bk][col] += 1
        V[(aj+1)*p + (bk+1)][col] += 1
        W[bk*n + ai][col] += 1
        W[(bk+1)*n + (ai+1)][col] -= 1
        col += 1

    # Helper to add Schoolbook M_{2,2,1}
    # A is 2x2, B is 2x1, C is 2x1
    def add_school_221(ai, aj, bk):
        nonlocal col
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # Factor 1: A00 B0
        U[ai*m + aj][col] += 1
        V[aj*p + bk][col] += 1
        W[bk*n + ai][col] += 1
        col += 1
        # Factor 2: A01 B1
        U[ai*m + (aj+1)][col] += 1
        V[(aj+1)*p + bk][col] += 1
        W[bk*n + ai][col] += 1
        col += 1
        # Factor 3: A10 B0
        U[(ai+1)*m + aj][col] += 1
        V[aj*p + bk][col] += 1
        W[bk*n + (ai+1)][col] += 1
        col += 1
        # Factor 4: A11 B1
        U[(ai+1)*m + (aj+1)][col] += 1
        V[(aj+1)*p + bk][col] += 1
        W[bk*n + (ai+1)][col] += 1
        col += 1

    # Helper to add Schoolbook M_{2,1,2}
    # A is 2x1, B is 1x2, C is 2x2
    def add_school_212(ai, aj, bk):
        nonlocal col
        # C00 = A0 B0
        U[ai*m + aj][col] += 1
        V[aj*p + bk][col] += 1
        W[bk*n + ai][col] += 1
        col += 1
        # C01 = A0 B1
        U[ai*m + aj][col] += 1
        V[aj*p + (bk+1)][col] += 1
        W[(bk+1)*n + ai][col] += 1
        col += 1
        # C10 = A1 B0
        U[(ai+1)*m + aj][col] += 1
        V[aj*p + bk][col] += 1
        W[bk*n + (ai+1)][col] += 1
        col += 1
        # C11 = A1 B1
        U[(ai+1)*m + aj][col] += 1
        V[aj*p + (bk+1)][col] += 1
        W[(bk+1)*n + (ai+1)][col] += 1
        col += 1

    # Helper to add Schoolbook M_{2,1,1}
    # A is 2x1, B is 1x1, C is 2x1
    def add_school_211(ai, aj, bk):
        nonlocal col
        # C0 = A0 B0
        U[ai*m + aj][col] += 1
        V[aj*p + bk][col] += 1
        W[bk*n + ai][col] += 1
        col += 1
        # C1 = A1 B0
        U[(ai+1)*m + aj][col] += 1
        V[aj*p + bk][col] += 1
        W[bk*n + (ai+1)][col] += 1
        col += 1

    # Block 1: k=0,1
    # A1 (j=0,1) * B1 (j=0,1, k=0,1) -> Strassen
    add_strassen(0, 0, 0)
    # A2 (j=2,3) * B2 (j=2,3, k=0,1) -> Strassen
    add_strassen(0, 2, 0)
    # A3 (j=4) * B3 (j=4, k=0,1) -> School 212
    add_school_212(0, 4, 0)

    # Block 2: k=2,3
    # A1 (j=0,1) * B1 (j=0,1, k=2,3) -> Strassen
    add_strassen(0, 0, 2)
    # A2 (j=2,3) * B2 (j=2,3, k=2,3) -> Strassen
    add_strassen(0, 2, 2)
    # A3 (j=4) * B3 (j=4, k=2,3) -> School 212
    add_school_212(0, 4, 2)

    # Block 3: k=4
    # A1 (j=0,1) * B1 (j=0,1, k=4) -> School 221
    add_school_221(0, 0, 4)
    # A2 (j=2,3) * B2 (j=2,3, k=4) -> School 221
    add_school_221(0, 2, 4)
    # A3 (j=4) * B3 (j=4, k=4) -> School 211
    add_school_211(0, 4, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
