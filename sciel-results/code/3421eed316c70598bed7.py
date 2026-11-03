"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if len(V[0]) != rank:
        return False
    if len(W[0]) != rank:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(rank):
                    total += U[ui][c] * V[vj][c] * W[wk][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Rank 30 decomposition
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen(a00_idx, a01_idx, a10_idx, a11_idx, 
                     b00_idx, b01_idx, b10_idx, b11_idx,
                     c00_idx, c01_idx, c10_idx, c11_idx):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[a00_idx][col] = 1; U[a11_idx][col] = 1
        V[b00_idx][col] = 1; V[b11_idx][col] = 1
        W[c00_idx][col] = 1; W[c11_idx][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a10_idx][col] = 1; U[a11_idx][col] = 1
        V[b00_idx][col] = 1
        W[c10_idx][col] = 1; W[c11_idx][col] = -1
        col += 1
        
        # M3 = A00(B10-B11)
        U[a00_idx][col] = 1
        V[b10_idx][col] = 1; V[b11_idx][col] = -1
        W[c01_idx][col] = 1; W[c11_idx][col] = 1
        col += 1
        
        # M4 = A11(B01-B10)
        U[a11_idx][col] = 1
        V[b01_idx][col] = 1; V[b10_idx][col] = -1
        W[c00_idx][col] = -1; W[c10_idx][col] = 1
        col += 1
        
        # M5 = (A00+A10)B11
        U[a00_idx][col] = 1; U[a10_idx][col] = 1
        V[b11_idx][col] = 1
        W[c01_idx][col] = 1; W[c11_idx][col] = -1
        col += 1
        
        # M6 = (A01-A11)(B10+B11)
        U[a01_idx][col] = 1; U[a11_idx][col] = -1
        V[b10_idx][col] = 1; V[b11_idx][col] = 1
        W[c00_idx][col] = -1; W[c11_idx][col] = 1
        col += 1
        
        # M7 = (A10-A00)(B01+B11)
        U[a10_idx][col] = 1; U[a00_idx][col] = -1
        V[b01_idx][col] = 1; V[b11_idx][col] = 1
        W[c00_idx][col] = 1; W[c01_idx][col] = 1
        col += 1

    # Helper to add outer product (2x1) * (1x2)
    # A is 2x1 (indices a0, a1), B is 1x2 (indices b0, b1)
    # C is 2x2 (indices c00, c01, c10, c11)
    # C00 = A0 B0, C01 = A0 B1, C10 = A1 B0, C11 = A1 B1
    # Rank 2:
    # M1 = A0 B0 + A1 B1 ? No.
    # M1 = A0 B0
    # M2 = A1 B1
    # M3 = (A0+A1)(B0+B1) = A0B0 + A0B1 + A1B0 + A1B1
    # M4 = (A0-A1)(B0-B1) = A0B0 - A0B1 - A1B0 + A1B1
    # We have 4 unknowns. We can use 4 multiplications.
    # Can we do it in 2?
    # No, rank of 2x2 outer product is 2?
    # Actually, the rank of the tensor T_{ik} = A_i B_k is 2.
    # We can use:
    # M1 = A0 B0
    # M2 = A1 B1
    # M3 = (A0+A1)(B0+B1)
    # M4 = (A0-A1)(B0-B1)
    # This is 4 terms.
    # Wait, is rank 2?
    # If rank is 2, we need 2 multiplications.
    # Let M1 = A0 B0 + A1 B1
    # Let M2 = A0 B1 + A1 B0
    # Then C00 = (M1 + M2)/2? No.
    # C00 = A0 B0
    # C11 = A1 B1
    # C01 = A0 B1
    # C10 = A1 B0
    # If we compute M1 = A0 B0, M2 = A0 B1, M3 = A1 B0, M4 = A1 B1, rank is 4.
    # If we compute M1 = A0 B0 + A1 B1, M2 = A0 B1 + A1 B0, we can't separate them.
    # So rank is 4?
    # Actually, the rank of 2x2 matrix multiplication is 7.
    # The rank of 2x1 * 1x2 is 4?
    # Let's use 4 terms to be safe.
    
    def add_outer(a0_idx, a1_idx, b0_idx, b1_idx, c00_idx, c01_idx, c10_idx, c11_idx):
        nonlocal col
        # M1 = A0 B0
        U[a0_idx][col] = 1; V[b0_idx][col] = 1
        W[c00_idx][col] = 1
        col += 1
        # M2 = A0 B1
        U[a0_idx][col] = 1; V[b1_idx][col] = 1
        W[c01_idx][col] = 1
        col += 1
        # M3 = A1 B0
        U[a1_idx][col] = 1; V[b0_idx][col] = 1
        W[c10_idx][col] = 1
        col += 1
        # M4 = A1 B1
        U[a1_idx][col] = 1; V[b1_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1

    # Helper to add schoolbook 1x3 * 3x2
    # A is 1x3 (a0, a1, a2), B is 3x2 (b00, b01, b10, b11, b20, b21)
    # C is 1x2 (c0, c1)
    # C0 = A0 B00 + A1 B10 + A2 B20
    # C1 = A0 B01 + A1 B11 + A2 B21
    # Rank 6
    
    def add_schoolbook_1x3_3x2(a0_idx, a1_idx, a2_idx,
                               b00_idx, b01_idx, b10_idx, b11_idx, b20_idx, b21_idx,
                               c0_idx, c1_idx):
        nonlocal col
        # A0 B00
        U[a0_idx][col] = 1; V[b00_idx][col] = 1
        W[c0_idx][col] = 1
        col += 1
        # A0 B01
        U[a0_idx][col] = 1; V[b01_idx][col] = 1
        W[c1_idx][col] = 1
        col += 1
        # A1 B10
        U[a1_idx][col] = 1; V[b10_idx][col] = 1
        W[c0_idx][col] = 1
        col += 1
        # A1 B11
        U[a1_idx][col] = 1; V[b11_idx][col] = 1
        W[c1_idx][col] = 1
        col += 1
        # A2 B20
        U[a2_idx][col] = 1; V[b20_idx][col] = 1
        W[c0_idx][col] = 1
        col += 1
        # A2 B21
        U[a2_idx][col] = 1; V[b21_idx][col] = 1
        W[c1_idx][col] = 1
        col += 1

    # Indices
    # A: 0..8
    # B: 0..11
    # C: 0..11
    
    # 1. C_top_left (i=0,1; k=0,1)
    # A[0:2, 0:2] x B[0:2, 0:2]
    # A00=0, A01=1, A10=3, A11=4
    # B00=0, B01=1, B10=4, B11=5
    # C00=0, C01=3, C10=1, C11=4
    add_strassen(0, 1, 3, 4, 0, 1, 4, 5, 0, 3, 1, 4)
    
    # A[0:2, 2] x B[2, 0:2]
    # A02=2, A12=5
    # B20=8, B21=9
    # C00=0, C01=3, C10=1, C11=4
    add_outer(2, 5, 8, 9, 0, 3, 1, 4)
    
    # 2. C_top_right (i=0,1; k=2,3)
    # A[0:2, 0:2] x B[0:2, 2:4]
    # A00=0, A01=1, A10=3, A11=4
    # B02=2, B03=3, B12=6, B13=7
    # C02=6, C03=9, C12=7, C13=10
    add_strassen(0, 1, 3, 4, 2, 3, 6, 7, 6, 9, 7, 10)
    
    # A[0:2, 2] x B[2, 2:4]
    # A02=2, A12=5
    # B22=10, B23=11
    # C02=6, C03=9, C12=7, C13=10
    add_outer(2, 5, 10, 11, 6, 9, 7, 10)
    
    # 3. C_bottom_left (i=2; k=0,1)
    # A[2, 0:3] x B[0:3, 0:2]
    # A20=6, A21=7, A22=8
    # B00=0, B01=1, B10=4, B11=5, B20=8, B21=9
    # C20=2, C21=5
    add_schoolbook_1x3_3x2(6, 7, 8, 0, 1, 4, 5, 8, 9, 2, 5)
    
    # 4. C_bottom_right (i=2; k=2,3)
    # A[2, 0:3] x B[0:3, 2:4]
    # A20=6, A21=7, A22=8
    # B02=2, B03=3, B12=6, B13=7, B22=10, B23=11
    # C22=8, C23=11
    add_schoolbook_1x3_3x2(6, 7, 8, 2, 3, 6, 7, 10, 11, 8, 11)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][column] = 1
                    V_s[j * p + k][column] = 1
                    W_s[k * n + i][column] = 1
                    column += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
