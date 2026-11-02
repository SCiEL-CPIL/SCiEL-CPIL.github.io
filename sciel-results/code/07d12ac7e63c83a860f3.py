"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank 43.
    # Split p=5 into p1=2 and p2=3.
    # C = A B1 + A B2
    # Rank(A B1) <= 17
    # Rank(A B2) <= 26
    
    # Helper to add terms to U, V, W
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        nonlocal rank
        U.append(u)
        V.append(v)
        W.append(w)
        rank += 1

    # --- Part 1: 3x3 x 2 multiplication ---
    # A is 3x3, B1 is 3x2.
    # Split A into 2x2, 2x1, 1x2, 1x1.
    # Split B1 into 2x2, 1x2.
    
    # Indices for A:
    # A00 (0,1 x 0,1), A01 (0,1 x 2), A10 (2 x 0,1), A11 (2 x 2)
    # Indices for B1:
    # B00 (0,1 x 0,1), B10 (2 x 0,1)
    
    # Output C1 is 3x2.
    # C00 = A00 B00 + A01 B10
    # C10 = A10 B00 + A11 B10
    
    # C00 (2x2)
    # A00 B00: 2x2x2 Strassen
    # Strassen 2x2x2 terms
    # M1 = (A11 + A00) * (B00 + B11) -> Here B is 2x2, so B00, B01, B10, B11
    # Wait, B00 in this context is the top-left 2x2 of B1.
    # Let's denote B1 blocks as B1_00 (2x2), B1_10 (1x2).
    # A00 is 2x2. B1_00 is 2x2.
    # Product is 2x2.
    # Use Strassen for A00 * B1_00.
    
    # Strassen 2x2x2:
    # M1 = (A00 + A11) (B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A10) B11
    # M6 = (A11 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # In our case, the "B" matrix for Strassen is B1_00 (2x2).
    # Let's call its elements b00, b01, b10, b11.
    # A00 elements a00, a01, a10, a11.
    
    # We need to map these to the global indices.
    # A indices: 0..8.
    # A00: (0,1) x (0,1) -> 0,1,3,4
    # A01: (0,1) x 2 -> 2,5
    # A10: 2 x (0,1) -> 6,7
    # A11: 2 x 2 -> 8
    
    # B1 indices: 0..5 (3x2).
    # B1_00: (0,1) x (0,1) -> 0,1,3,4
    # B1_10: 2 x (0,1) -> 6,7
    
    # C1 indices: 0..5 (3x2).
    # C00: (0,1) x (0,1) -> 0,1,3,4
    # C10: 2 x (0,1) -> 6,7
    
    # We will generate the 7 Strassen terms for A00 * B1_00.
    # Term 1: (A00 + A11) * (B00 + B11)
    # u: C00, C11? No, Strassen gives 4 outputs.
    # Here we are computing a 2x2 matrix product.
    # The output is 2x2.
    # The "W" vector for this sub-problem will have 4 non-zero entries.
    
    # This is getting complex to code generically.
    # I will hardcode the 17 terms for 3x3x2 and 26 terms for 3x3x3.
    
    # For brevity and correctness, I will use the schoolbook for the sub-blocks if I can't easily code Strassen.
    # Schoolbook for 2x2x2 is 4.
    # If I use schoolbook for all sub-blocks:
    # 3x3x2:
    # C00: 4 (A00B00) + 2 (A01B10) = 6
    # C10: 2 (A10B00) + 2 (A11B10) = 4
    # Total 10.
    
    # 3x3x3:
    # C00: 4 + 2 = 6
    # C01: 2 + 1 = 3
    # C10: 2 + 2 = 4
    # C11: 1 + 1 = 2
    # Total 15.
    
    # Total 10 + 15 = 25.
    # This is much better than 43!
    # Wait, did I calculate schoolbook correctly?
    # Schoolbook for 2x2x2 is 4 multiplications.
    # Schoolbook for 2x1x1x2 is 2 multiplications.
    # Yes.
    
    # So rank 25 is achievable with a simple block decomposition using schoolbook inside.
    # Let's verify.
    # 3x3x2:
    # A00 (2x2) * B00 (2x2) -> 4 mults.
    # A01 (2x1) * B10 (1x2) -> 2 mults.
    # A10 (1x2) * B00 (2x2) -> 2 mults.
    # A11 (1x1) * B10 (1x2) -> 2 mults.
    # Sum: 4+2+2+2 = 10.
    
    # 3x3x3:
    # A00 (2x2) * B00 (2x2) -> 4
    # A01 (2x1) * B01 (1x2) -> 2
    # A10 (1x2) * B10 (2x2) -> 2
    # A11 (1x1) * B11 (1x1) -> 1
    # A00 (2x2) * B01 (2x1) -> 2 (Wait, B01 is 2x1? No, B is 3x3. B01 is 2x1? No, B01 is 2x1?
    # B blocks:
    # B00 (2x2), B01 (2x1)
    # B10 (1x2), B11 (1x1)
    
    # C00 = A00 B00 + A01 B10
    # A00 B00: 4
    # A01 (2x1) * B10 (1x2): 2
    # Sum 6.
    
    # C01 = A00 B01 + A01 B11
    # A00 (2x2) * B01 (2x1): 2
    # A01 (2x1) * B11 (1x1): 2
    # Sum 4.
    
    # C10 = A10 B00 + A11 B10
    # A10 (1x2) * B00 (2x2): 2
    # A11 (1x1) * B10 (1x2): 2
    # Sum 4.
    
    # C11 = A10 B01 + A11 B11
    # A10 (1x2) * B01 (2x1): 2
    # A11 (1x1) * B11 (1x1): 1
    # Sum 3.
    
    # Total 3x3x3: 6+4+4+3 = 17.
    
    # Total 3x3x5: 10 + 17 = 27.
    
    # This is a huge improvement over 45.
    # And it's easy to implement.

    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term for sub-blocks
    # u_idx: index in U (0..8)
    # v_idx: index in V (0..14)
    # w_idx: index in W (0..14)
    def add_schoolbook(u_idx, v_idx, w_idx):
        nonlocal col
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # --- 3x3x2 Part (cols 0-1 of B) ---
    # B indices: 0,1 (row 0), 3,4 (row 1), 6,7 (row 2)
    # A indices: 0,1,2 (row 0), 3,4,5 (row 1), 6,7,8 (row 2)
    # C indices: 0,1 (row 0), 3,4 (row 1), 6,7 (row 2)
    
    # C00 (rows 0,1; cols 0,1) = A00 B00 + A01 B10
    # A00: A[0,1,3,4]. B00: B[0,1,3,4].
    # A00 B00:
    add_schoolbook(0, 0, 0) # A00_00 * B00_00 -> C00_00
    add_schoolbook(0, 1, 1) # A00_01 * B00_01 -> C00_01
    add_schoolbook(3, 0, 3) # A00_10 * B00_00 -> C00_10
    add_schoolbook(3, 1, 4) # A00_11 * B00_01 -> C00_11
    
    # A01 B10: A01 is A[2,5], B10 is B[6,7].
    # A01_0 * B10_0 -> C00_00
    add_schoolbook(2, 6, 0)
    # A01_0 * B10_1 -> C00_01
    add_schoolbook(2, 7, 1)
    # A01_1 * B10_0 -> C00_10
    add_schoolbook(5, 6, 3)
    # A01_1 * B10_1 -> C00_11
    add_schoolbook(5, 7, 4)
    
    # C10 (row 2; cols 0,1) = A10 B00 + A11 B10
    # A10: A[6,7]. B00: B[0,1,3,4].
    # A10_0 * B00_00 -> C10_00
    add_schoolbook(6, 0, 6)
    # A10_0 * B00_01 -> C10_01
    add_schoolbook(6, 1, 7)
    # A10_1 * B00_10 -> C10_00
    add_schoolbook(7, 3, 6)
    # A10_1 * B00_11 -> C10_01
    add_schoolbook(7, 4, 7)
    
    # A11: A[8]. B10: B[6,7].
    # A11 * B10_0 -> C10_00
    add_schoolbook(8, 6, 6)
    # A11 * B10_1 -> C10_01
    add_schoolbook(8, 7, 7)
    
    # --- 3x3x3 Part (cols 2-4 of B) ---
    # B indices: 2,5,8 (col 2), 4,7,10 (col 3), 9,12,15? No, B is 3x5.
    # B indices:
    # Row 0: 0,1,2,3,4
    # Row 1: 5,6,7,8,9
    # Row 2: 10,11,12,13,14
    # Cols 2,3,4 correspond to indices:
    # Col 2: 2, 7, 12
    # Col 3: 3, 8, 13
    # Col 4: 4, 9, 14
    
    # B2 (3x3) blocks:
    # B2_00 (rows 0,1; cols 2,3): B[2,3,7,8]
    # B2_01 (rows 0,1; col 4): B[4,9]
    # B2_10 (row 2; cols 2,3): B[12,13]
    # B2_11 (row 2; col 4): B[14]
    
    # C2 (3x3) blocks:
    # C2_00 (rows 0,1; cols 2,3): C[2,3,7,8]
    # C2_01 (rows 0,1; col 4): C[4,9]
    # C2_10 (row 2; cols 2,3): C[12,13]
    # C2_11 (row 2; col 4): C[14]
    
    # C2_00 = A00 B2_00 + A01 B2_10
    # A00 B2_00:
    add_schoolbook(0, 2, 2) # A00_00 * B2_00_00 -> C2_00_00
    add_schoolbook(0, 3, 3) # A00_01 * B2_00_01 -> C2_00_01
    add_schoolbook(3, 2, 7) # A00_10 * B2_00_00 -> C2_00_10
    add_schoolbook(3, 3, 8) # A00_11 * B2_00_01 -> C2_00_11
    
    # A01 B2_10:
    add_schoolbook(2, 12, 2) # A01_0 * B2_10_0 -> C2_00_00
    add_schoolbook(2, 13, 3) # A01_0 * B2_10_1 -> C2_00_01
    add_schoolbook(5, 12, 7) # A01_1 * B2_10_0 -> C2_00_10
    add_schoolbook(5, 13, 8) # A01_1 * B2_10_1 -> C2_00_11
    
    # C2_01 = A00 B2_01 + A01 B2_11
    # A00 B2_01:
    add_schoolbook(0, 4, 4) # A00_00 * B2_01_0 -> C2_01_00
    add_schoolbook(3, 4, 9) # A00_10 * B2_01_0 -> C2_01_10
    add_schoolbook(1, 9, 4) # A00_01 * B2_01_1 -> C2_01_00
    add_schoolbook(4, 9, 9) # A00_11 * B2_01_1 -> C2_01_10
    
    # A01 B2_11:
    add_schoolbook(2, 14, 4) # A01_0 * B2_11 -> C2_01_00
    add_schoolbook(5, 14, 9) # A01_1 * B2_11 -> C2_01_10
    
    # C2_10 = A10 B2_00 + A11 B2_10
    # A10 B2_00:
    add_schoolbook(6, 2, 12) # A10_0 * B2_00_00 -> C2_10_00
    add_schoolbook(6, 3, 13) # A10_0 * B2_00_01 -> C2_10_01
    add_schoolbook(7, 7, 12) # A10_1 * B2_00_10 -> C2_10_00
    add_schoolbook(7, 8, 13) # A10_1 * B2_00_11 -> C2_10_01
    
    # A11 B2_10:
    add_schoolbook(8, 12, 12) # A11 * B2_10_0 -> C2_10_00
    add_schoolbook(8, 13, 13) # A11 * B2_10_1 -> C2_10_01
    
    # C2_11 = A10 B2_01 + A11 B2_11
    # A10 B2_01:
    add_schoolbook(6, 4, 14) # A10_0 * B2_01_0 -> C2_11
    add_schoolbook(7, 9, 14) # A10_1 * B2_01_1 -> C2_11
    
    # A11 B2_11:
    add_schoolbook(8, 14, 14) # A11 * B2_11 -> C2_11

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
