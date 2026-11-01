"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # We use a hybrid decomposition:
    # Split the inner dimension m=3 into two parts: m1=2, m2=1.
    # For the (n x m1) x (m1 x p) = (2x2) x (2x7) block, we use a Strassen-like 
    # decomposition for the 2x2x2 part combined with schoolbook for the remaining columns.
    # Actually, let's be more precise.
    # The multiplication is (2x3) @ (3x7) = 2x7.
    # We can split A (2x3) into A1 (2x2) and A2 (2x1).
    # We can split B (3x7) into B1 (2x7) and B2 (1x7).
    # C = A1 B1 + A2 B2.
    
    # For A2 B2: A2 is 2x1, B2 is 1x7. This is rank 2 (schoolbook: u^T v^T for each col).
    # Wait, A2 B2 is (2x1)(1x7) = 2x7. Rank is at most 1? No, rank of product is <= min(rank A2, rank B2) = 1.
    # But we need a bilinear decomposition.
    # A2 = [a11; a12], B2 = [b1, ..., b7].
    # (A2 B2)[i, j] = A2[i, 0] * B2[0, j].
    # This is a rank 1 tensor for each (i, j) pair? No.
    # The tensor T_{i, 0, j} = A_{i0} B_{0j}. This is rank 1 (u_i v_j).
    # So A2 B2 contributes 1 to the rank?
    # Let's check. U has row for (i, m_idx). V has row for (m_idx, j). W has row for (j, i).
    # For the term A_{i, 2} B_{2, j}, we have U[i, 2] * V[2, j] * W[j, i] = 1.
    # This requires 1 rank?
    # If we set U[i, 2, r] = 1, V[2, j, r] = 1, W[j, i, r] = 1 for all i, j, that's not possible with 1 rank because W[j, i] depends on both.
    # Actually, the bilinear form is sum_r U[i, m_idx, r] V[m_idx, j, r] W[j, i, r].
    # For fixed m_idx, it is (sum_r U[i, m_idx, r] W[j, i, r]) V[m_idx, j, r]? No.
    # It is sum_r (U[i, m_idx, r] V[m_idx, j, r]) W[j, i, r].
    # This is a tensor.
    # For A2 B2 (m_idx = 2), the tensor is T_{i, 2, j} = A_{i2} B_{2j}.
    # This tensor has rank 1? T_{i, 2, j} = u_i v_j.
    # In our factorization, T_{i, m, j} = sum_r U_{i, m, r} V_{m, j, r} W_{j, i, r}.
    # If we want T_{i, 2, j} = A_{i2} B_{2j}, we can just use 1 rank if we allow W to be arbitrary?
    # No, W is shared.
    # Actually, the standard schoolbook for (2x1)(1x7) uses 2*1*7 = 14 terms?
    # No, schoolbook for (n x m)(m x p) uses n*m*p terms.
    # For (2x1)(1x7), it is 2*1*7 = 14 terms?
    # Let's see. For each i in 0..1, j in 0..6, the term is A_{i0} B_{0j}.
    # In the factorization: sum_r U_{i, 0, r} V_{0, j, r} W_{j, i, r}.
    # To get A_{i0} B_{0j}, we need U_{i, 0, r} to pick A_{i0}, V_{0, j, r} to pick B_{0j}, and W to pass it through.
    # If we use 1 rank per (i, j), that's 14 ranks.
    # Can we do better?
    # T_{i, 0, j} = A_{i0} B_{0j}. This is a rank 1 tensor in (i, j) space, but it's a 3-way tensor.
    # The slice for m=0 is T_{i, 0, j}. This is a 2x7 matrix of rank 1.
    # The rank of a tensor with one mode fixed...
    # Actually, it's easier to just use schoolbook for the whole thing and then try to merge.
    
    # Let's stick to the known improvement: 39 rank.
    # How to get 39?
    # Schoolbook is 42.
    # Strassen on 2x2x2 blocks.
    # Split m=3 into 2 and 1.
    # Part 1: A1 (2x2) @ B1 (2x7). 
    # B1 has 7 columns. We can split B1 into B1a (2x2), B1b (2x2), B1c (2x2), B1d (2x1).
    # Then A1 @ B1 = A1 @ [B1a B1b B1c B1d].
    # A1 @ B1a: 2x2 @ 2x2. Use Strassen (rank 7).
    # A1 @ B1b: 2x2 @ 2x2. Use Strassen (rank 7).
    # A1 @ B1c: 2x2 @ 2x2. Use Strassen (rank 7).
    # A1 @ B1d: 2x2 @ 2x1. Schoolbook? Rank 2*2*1 = 4?
    # Or Strassen for 2x2x1? Rank is 4? (Actually 2x2x1 is rank 4, same as schoolbook).
    # Part 2: A2 (2x1) @ B2 (1x7).
    # A2 @ B2: 2x1 @ 1x7. Schoolbook rank 2*1*7 = 14?
    # Or can we do better?
    # Total rank: 7 + 7 + 7 + 4 + 14 = 39.
    
    # This matches the known result of 39.
    
    # Let's implement this.
    # Indices:
    # U: (n*m, rank) = (2*3, 39) = (6, 39)
    # V: (m*p, rank) = (3*7, 39) = (21, 39)
    # W: (p*n, rank) = (7*2, 39) = (14, 39)
    
    # U rows: (i, m_idx) where i in 0..1, m_idx in 0..2.
    # V rows: (m_idx, j) where m_idx in 0..2, j in 0..6.
    # W rows: (j, i) where j in 0..6, i in 0..1.
    
    rank = 39
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(21)]
    W = [[0] * rank for _ in range(14)]
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10
    # M3: A00(B01-B11) -> C01
    # M4: A11(B10-B00) -> C10, C11
    # M5: (A00+A01)B11 -> C00, C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00
    
    # For a 2x2 @ 2x2 block with global m_offset and j_offset:
    # A is rows i in 0..1, cols m_offset..m_offset+1
    # B is rows m_offset..m_offset+1, cols j_offset..j_offset+1
    # C is rows i in 0..1, cols j_offset..j_offset+1
    
    def add_strassen(m_off, j_off):
        nonlocal col
        # M1
        U[0*3 + m_off][col] += 1  # A00
        U[1*3 + m_off + 1][col] += 1  # A11
        V[m_off*7 + j_off][col] += 1  # B00
        V[(m_off+1)*7 + j_off + 1][col] += 1  # B11
        W[(j_off)*2 + 0][col] += 1  # C00
        W[(j_off+1)*2 + 1][col] += 1  # C11
        col += 1
        
        # M2
        U[1*3 + m_off][col] += 1  # A10
        U[1*3 + m_off + 1][col] += 1  # A11
        V[m_off*7 + j_off][col] += 1  # B00
        W[(j_off)*2 + 1][col] += 1  # C10
        col += 1
        
        # M3
        U[0*3 + m_off][col] += 1  # A00
        V[m_off*7 + j_off + 1][col] += 1  # B01
        V[(m_off+1)*7 + j_off + 1][col] += -1  # -B11
        W[(j_off+1)*2 + 0][col] += 1  # C01
        col += 1
        
        # M4
        U[1*3 + m_off + 1][col] += 1  # A11
        V[(m_off+1)*7 + j_off][col] += 1  # B10
        V[m_off*7 + j_off][col] += -1  # -B00
        W[(j_off)*2 + 1][col] += 1  # C10
        W[(j_off+1)*2 + 1][col] += 1  # C11
        col += 1
        
        # M5
        U[0*3 + m_off][col] += 1  # A00
        U[0*3 + m_off + 1][col] += 1  # A01
        V[(m_off+1)*7 + j_off + 1][col] += 1  # B11
        W[(j_off)*2 + 0][col] += 1  # C00
        W[(j_off+1)*2 + 0][col] += 1  # C01
        col += 1
        
        # M6
        U[1*3 + m_off][col] += 1  # A10
        U[0*3 + m_off][col] += -1  # -A00
        V[m_off*7 + j_off][col] += 1  # B00
        V[m_off*7 + j_off + 1][col] += 1  # B01
        W[(j_off+1)*2 + 1][col] += 1  # C11
        col += 1
        
        # M7
        U[0*3 + m_off + 1][col] += 1  # A01
        U[1*3 + m_off + 1][col] += -1  # -A11
        V[(m_off+1)*7 + j_off][col] += 1  # B10
        V[(m_off+1)*7 + j_off + 1][col] += 1  # B11
        W[(j_off)*2 + 0][col] += 1  # C00
        col += 1

    # Schoolbook 2x2x1
    def add_schoolbook_2x2x1(m_off, j_off):
        nonlocal col
        # A is 2x2 (cols m_off, m_off+1), B is 2x1 (row m_off, m_off+1, col j_off)
        # C is 2x1
        for i in range(2):
            for mm in range(2):
                U[i*3 + m_off + mm][col] = 1
                V[(m_off + mm)*7 + j_off][col] = 1
                W[j_off*2 + i][col] = 1
                col += 1

    # Schoolbook 2x1x7
    def add_schoolbook_2x1x7(m_off, j_off):
        nonlocal col
        # A is 2x1 (col m_off), B is 1x7 (row m_off, cols j_off..j_off+6)
        # C is 2x7
        for i in range(2):
            for j in range(7):
                U[i*3 + m_off][col] = 1
                V[m_off*7 + j_off + j][col] = 1
                W[(j_off + j)*2 + i][col] = 1
                col += 1

    # Part 1: A1 (2x2) @ B1 (2x7)
    # Split B1 into 4 parts: 2x2, 2x2, 2x2, 2x1
    add_strassen(0, 0)   # cols 0-1
    add_strassen(0, 2)   # cols 2-3
    add_strassen(0, 4)   # cols 4-5
    add_schoolbook_2x2x1(0, 6)  # col 6
    
    # Part 2: A2 (2x1) @ B2 (1x7)
    # A2 is col 2, B2 is row 2
    add_schoolbook_2x1x7(2, 0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

