"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 11 decomposition: Strassen (7) for first 2 cols + Schoolbook (4) for 3rd col
    
    U = []
    V = []
    W = []
    
    # Strassen for 2x2 x 2x2 (cols 0,1)
    # A is 2x2, B1 is 2x2 (cols 0,1 of B)
    # C1 is 2x2 (cols 0,1 of C)
    
    # Strassen factors for M2,2 x M2,2
    # U: 4x7, V: 4x7 (but V is 6x7 for M2,3, so we pad with 0 for col 2)
    # W: 4x7 (but W is 6x7 for M2,3, so we pad with 0 for col 2? No, W indices are k*n+i.
    # C1 has k in {0,1}. C2 has k=2.
    # So W for Strassen will have non-zeros only for k=0,1.
    
    strassen_U = [
        [1, 0, 0, 0, 0, 0, 0], # A00
        [0, 1, 0, 0, 0, 0, 0], # A01
        [0, 0, 1, 0, 0, 0, 0], # A10
        [0, 0, 0, 1, 0, 0, 0]  # A11
    ]
    # Standard Strassen U vectors (linear forms in A)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U vectors (rows are i*m+j = 0,1,2,3)
    U_s = [
        [1, 1, 0, 0, 0, 0, 0], # A00 in M1, M2? No.
        # Let's use the standard matrix representation
    ]
    
    # It is safer to generate Strassen factors programmatically or hardcode correctly.
    # Hardcoding Strassen factors for 2x2:
    # U (4x7)
    U_s = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 1, 1, 0, 1],  # A01
        [1, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, -1, 1, 0, 0, -1] # A11
    ]
    # V (4x7) for B1 (2x2). Indices j*p+k. j in {0,1}, k in {0,1}.
    # B1 indices: 00, 01, 10, 11 -> 0, 1, 2, 3 in V (since p=3, j*3+k? No, V is m*p = 2*3=6.
    # V indices: j*3+k.
    # B1 uses k=0,1. So V rows 0,1 (j=0) and 3,4 (j=1).
    # B10 is index 0*3+0=0. B11 is 0*3+1=1.
    # B20 is 1*3+0=3. B21 is 1*3+1=4.
    
    V_s = [
        [1, 1, 0, -1, 0, 1, 1], # B00 (idx 0)
        [1, 0, 1, 1, 1, 1, 0],  # B01 (idx 1)
        [0, 0, 0, 0, 0, 0, 0],  # B02 (idx 2) - unused in Strassen
        [0, 1, -1, 0, 0, 1, 1], # B10 (idx 3)
        [1, 0, 0, -1, 1, 0, 0], # B11 (idx 4)
        [0, 0, 0, 0, 0, 0, 0]   # B12 (idx 5) - unused
    ]
    # W (4x7) for C1 (2x2). Indices k*n+i. k in {0,1}, i in {0,1}.
    # C10 is 0*2+0=0. C11 is 0*2+1=1.
    # C20 is 1*2+0=2. C21 is 1*2+1=3.
    # W is p*n = 3*2=6.
    # W rows 0,1 (k=0) and 2,3 (k=1). Row 4,5 (k=2) are 0.
    
    W_s = [
        [0, -1, 1, 0, 0, 1, 0], # C00 (idx 0)
        [0, 0, 0, 1, 0, 0, 1],  # C01 (idx 1)
        [1, 0, 0, -1, 1, 0, 0], # C10 (idx 2)
        [1, 1, -1, 0, 0, -1, 1],# C11 (idx 3)
        [0, 0, 0, 0, 0, 0, 0],  # C20 (idx 4)
        [0, 0, 0, 0, 0, 0, 0]   # C21 (idx 5)
    ]
    
    # Schoolbook for 3rd column (k=2)
    # C02 = A00 B02 + A01 B12
    # C12 = A10 B02 + A11 B12
    # 4 terms.
    
    # Term 8: A00 B02 -> C02
    # Term 9: A01 B12 -> C02
    # Term 10: A10 B02 -> C12
    # Term 11: A11 B12 -> C12
    
    U_sb = [
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1]  # A11
    ]
    V_sb = [
        [1, 0, 1, 0], # B02 (idx 2)
        [0, 0, 0, 0], # B0? No.
        # V indices: j*3+k.
        # B02 is j=0, k=2 -> idx 2.
        # B12 is j=1, k=2 -> idx 5.
        # So V_sb rows:
        # 0: 0
        # 1: 0
        # 2: 1, 0, 1, 0 (B02 in terms 8, 10)
        # 3: 0
        # 4: 0
        # 5: 0, 1, 0, 1 (B12 in terms 9, 11)
    ]
    V_sb = [
        [0, 0, 0, 0], # idx 0
        [0, 0, 0, 0], # idx 1
        [1, 0, 1, 0], # idx 2 (B02)
        [0, 0, 0, 0], # idx 3
        [0, 0, 0, 0], # idx 4
        [0, 1, 0, 1]  # idx 5 (B12)
    ]
    W_sb = [
        [0, 0, 0, 0], # C00
        [0, 0, 0, 0], # C01
        [0, 0, 0, 0], # C10
        [0, 0, 0, 0], # C11
        [1, 1, 0, 0], # C20 (idx 4) -> C02? No, C index k*n+i.
        # C02 is k=0, i=2? No, i is row of C. C is 2x3.
        # i in {0,1}. k in {0,1,2}.
        # C02 is i=0, k=2. Index 2*2+0 = 4.
        # C12 is i=1, k=2. Index 2*2+1 = 5.
        [0, 0, 1, 1]  # C21 (idx 5)
    ]
    # Wait, W_sb indices:
    # k=2, i=0 -> 2*2+0=4.
    # k=2, i=1 -> 2*2+1=5.
    # So W_sb[4] gets 1 for terms 8,9.
    # W_sb[5] gets 1 for terms 10,11.
    
    W_sb = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Combine
    U = [row_s + row_sb for row_s, row_sb in zip(U_s, U_sb)]
    V = [row_s + row_sb for row_s, row_sb in zip(V_s, V_sb)]
    W = [row_s + row_sb for row_s, row_sb in zip(W_s, W_sb)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
