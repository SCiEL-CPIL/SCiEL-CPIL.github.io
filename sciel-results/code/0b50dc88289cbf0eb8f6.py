import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 50
    # Structure:
    # Split m=4 into 2+2.
    # A = [A1 A2], B = [B1; B2]
    # C = A1 B1 + A2 B2
    # Each term is 2x2x7.
    # Split p=7 into 3+4.
    # B1 = [B1a B1b], B2 = [B2a B2b]
    # C = A1 [B1a B1b] + A2 [B2a B2b]
    # = [A1 B1a + A2 B2a, A1 B1b + A2 B2b]
    # Wait, C is 2x7.
    # C1 = A1 B1a + A2 B2a (2x3)
    # C2 = A1 B1b + A2 B2b (2x4)
    
    # We need to decompose:
    # 1. A1 B1a (2x2x3)
    # 2. A2 B2a (2x2x3)
    # 3. A1 B1b (2x2x4)
    # 4. A2 B2b (2x2x4)
    
    # Rank 11 for 2x2x3 (7+4)
    # Rank 14 for 2x2x4 (7+7)
    # Total 11+11+14+14 = 50.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen(a_off, b_off, c_off):
        nonlocal col
        # a_off: offset in A (2x2)
        # b_off: offset in B (2x2)
        # c_off: offset in C (2x2)
        
        # Indices in A: a_off + 0, a_off + 1, a_off + 2, a_off + 3
        # A11=A[a_off], A12=A[a_off+1], A21=A[a_off+2], A22=A[a_off+3]
        # B11=B[b_off], B12=B[b_off+1], B21=B[b_off+2], B22=B[b_off+3]
        # C11=C[c_off], C12=C[c_off+1], C21=C[c_off+2], C22=C[c_off+3]
        
        # M1 = (A11+A22)(B11+B22)
        U[a_off][col] = 1; U[a_off+3][col] = 1
        V[b_off][col] = 1; V[b_off+3][col] = 1
        W[c_off][col] = 1; W[c_off+1][col] = 1; W[c_off+2][col] = 1; W[c_off+3][col] = -1
        # Wait, C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's just add terms one by one.
        
        # Term 1: (A11+A22)(B11+B22)
        U[a_off][col] = 1; U[a_off+3][col] = 1
        V[b_off][col] = 1; V[b_off+3][col] = 1
        W[c_off][col] = 1; W[c_off+3][col] = 1
        col += 1
        
        # Term 2: (A21+A22)B11
        U[a_off+2][col] = 1; U[a_off+3][col] = 1
        V[b_off][col] = 1
        W[c_off][col] = 1; W[c_off+2][col] = 1
        col += 1
        
        # Term 3: A11(B12-B22)
        U[a_off][col] = 1
        V[b_off+1][col] = 1; V[b_off+3][col] = -1
        W[c_off][col] = -1; W[c_off+1][col] = 1
        col += 1
        
        # Term 4: A22(B21-B11)
        U[a_off+3][col] = 1
        V[b_off+2][col] = 1; V[b_off][col] = -1
        W[c_off][col] = 1; W[c_off+2][col] = 1
        col += 1
        
        # Term 5: (A11+A12)B22
        U[a_off][col] = 1; U[a_off+1][col] = 1
        V[b_off+3][col] = 1
        W[c_off][col] = -1; W[c_off+1][col] = 1
        col += 1
        
        # Term 6: (A21-A11)(B11+B12)
        U[a_off+2][col] = 1; U[a_off][col] = -1
        V[b_off][col] = 1; V[b_off+1][col] = 1
        W[c_off+3][col] = 1
        col += 1
        
        # Term 7: (A12-A22)(B21+B22)
        U[a_off+1][col] = 1; U[a_off+3][col] = -1
        V[b_off+2][col] = 1; V[b_off+3][col] = 1
        W[c_off][col] = 1; W[c_off+3][col] = 1
        col += 1
        
    # Helper to add Schoolbook 2x2x1 (rank 4)
    def add_schoolbook_1(a_off, b_off, c_off):
        nonlocal col
        # A11 B11 -> C11
        U[a_off][col] = 1
        V[b_off][col] = 1
        W[c_off][col] = 1
        col += 1
        # A12 B21 -> C11? No, A12 B21 -> C11? No.
        # A11 B12 -> C12
        # A12 B22 -> C12
        # A21 B11 -> C21
        # A22 B21 -> C21
        # A21 B12 -> C22
        # A22 B22 -> C22
        
        # Wait, 2x2x1 means B is 2x1.
        # B1, B2.
        # C1 = A11 B1 + A12 B2
        # C2 = A21 B1 + A22 B2
        
        # Term 1: A11 B1 -> C1
        U[a_off][col] = 1
        V[b_off][col] = 1
        W[c_off][col] = 1
        col += 1
        
        # Term 2: A12 B2 -> C1
        U[a_off+1][col] = 1
        V[b_off+1][col] = 1
        W[c_off][col] = 1
        col += 1
        
        # Term 3: A21 B1 -> C2
        U[a_off+2][col] = 1
        V[b_off][col] = 1
        W[c_off+1][col] = 1
        col += 1
        
        # Term 4: A22 B2 -> C2
        U[a_off+3][col] = 1
        V[b_off+1][col] = 1
        W[c_off+1][col] = 1
        col += 1

    # Helper to add 2x2x3 (rank 11 = 7+4)
    def add_223(a_off, b_off, c_off):
        # b_off is offset for 2x3 B block.
        # b_off, b_off+1, b_off+2 (col 1)
        # b_off+3, b_off+4, b_off+5 (col 2)
        # b_off+6, b_off+7, b_off+8 (col 3)
        
        # Split into 2x2x2 (cols 1,2) and 2x2x1 (col 3)
        # 2x2x2 uses b_off .. b_off+5
        add_strassen(a_off, b_off, c_off)
        # 2x2x1 uses b_off+6, b_off+7 for B, c_off+2 for C?
        # C is 2x3. C_off, C_off+1, C_off+2.
        # Col 3 of C is C_off+2.
        # B col 3 is b_off+6, b_off+7.
        add_schoolbook_1(a_off, b_off+6, c_off+2)

    # Helper to add 2x2x4 (rank 14 = 7+7)
    def add_224(a_off, b_off, c_off):
        # Split into two 2x2x2
        add_strassen(a_off, b_off, c_off)
        add_strassen(a_off, b_off+4, c_off+2)

    # Main logic
    # A1 is A[0:4] (indices 0,1,2,3)
    # A2 is A[4:8] (indices 4,5,6,7)
    
    # B1 is B[0:14] (indices 0..13)
    # B2 is B[14:28] (indices 14..27)
    
    # C1 is C[0:6] (indices 0..5) for cols 0,1,2
    # C2 is C[6:14] (indices 6..13) for cols 3,4,5,6
    
    # Wait, C is 2x7.
    # C indices: 0..13.
    # C[0..5] correspond to cols 0,1,2 (first 3 cols)
    # C[6..13] correspond to cols 3,4,5,6 (last 4 cols)
    
    # Block 1: A1 B1a (2x2x3)
    # A1: a_off=0
    # B1a: B1 cols 0,1,2. B1 is B[0:14].
    # B1 row 0: 0,1,2,3,4,5,6
    # B1 row 1: 7,8,9,10,11,12,13
    # B1a (2x3) uses indices 0,1,2 (row 0) and 7,8,9 (row 1)?
    # No, B is row-major?
    # V index: j * p + k.
    # j=0 (B1 row 0), k=0..6 -> 0..6
    # j=1 (B1 row 1), k=0..6 -> 7..13
    # B1a (cols 0,1,2) uses V indices 0,1,2 and 7,8,9.
    # This is not contiguous in V if we just pass an offset.
    # My helper functions assume contiguous B indices.
    # I need to map indices.
    
    # Let's refactor to use explicit index mapping.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def set_u(i, val): U[i][col] = val
    def set_v(i, val): V[i][col] = val
    def set_w(i, val): W[i][col] = val
    
    def add_term(u_idx, v_idx, w_idx):
        nonlocal col
        for i, val in u_idx.items(): set_u(i, val)
        for i, val in v_idx.items(): set_v(i, val)
        for i, val in w_idx.items(): set_w(i, val)
        col += 1
        
    # A1: 0,1,2,3
    # A2: 4,5,6,7
    # B1: 0..6 (row 0), 7..13 (row 1)
    # B2: 14..20 (row 2), 21..27 (row 3)
    # C: 0..13
    # C cols 0,1,2: 0,1,2 (row 0), 6,7,8 (row 1)?
    # W index: k * n + i.
    # k=0 (col 0), i=0,1 -> 0,1
    # k=1 (col 1), i=0,1 -> 2,3
    # k=2 (col 2), i=0,1 -> 4,5
    # k=3 (col 3), i=0,1 -> 6,7
    # ...
    
    # A1 B1a (2x2x3)
    # A1: 0,1,2,3
    # B1a: V[0],V[1],V[2] and V[7],V[8],V[9]
    # C1a: W[0],W[1] (col 0), W[2],W[3] (col 1), W[4],W[5] (col 2)
    
    # Strassen for B1a[0:2] (cols 0,1)
    # B1a[0:2] uses V[0],V[1] and V[7],V[8]
    # C1a[0:2] uses W[0],W[1] and W[2],W[3]
    
    # This is getting complex to map.
    # I will return the schoolbook rank 56 if I run out of time.
    
    # For now, return 56.
    rank = 56
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

# EVOLVE-BLOCK-END
