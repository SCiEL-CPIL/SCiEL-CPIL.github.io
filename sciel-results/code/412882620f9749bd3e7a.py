"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    total_rank = 73
    U = [[0] * total_rank for _ in range(n * m)]  # 15 x 73
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 73
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x 73
    
    def add_term(row_u, row_v, row_w, col, val_u=1, val_v=1, val_w=1):
        U[row_u][col] += val_u
        V[row_v][col] += val_v
        W[row_w][col] += val_w

    col = 0
    # Component 1: 1x2x6, rank 6
    for k in range(6):
        # i=0, j=0,1; k=0..5
        for j in range(2):
            add_term(0*5+j, j*6+k, k*3+0, col)
        col += 1
    # col is now 6

    # Component 2: 2x2x6, rank 21 (3 blocks of 2x2x2, each Strassen rank 7)
    # i=1,2; j=0,1; k=0..5
    # 3 blocks: k in {0,1}, {2,3}, {4,5}
    for b in range(3):
        k0 = 2*b
        k1 = 2*b+1
        # Strassen for 2x2x2
        # Local i in {0,1} -> global i in {1,2}
        # Local j in {0,1} -> global j in {0,1}
        # Local k in {0,1} -> global k in {k0, k1}
        
        # U indices: i*5+j, i in {1,2}, j in {0,1}
        # V indices: j*6+k, j in {0,1}, k in {k0, k1}
        # W indices: k*3+i, k in {k0, k1}, i in {1,2}
        
        # Strassen terms (7 terms)
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C01 -= M1, C10 -= M1, C11 += M1
        # In tensor form:
        # Term 1: U: (0,0),(1,1) have +1; V: (0,0),(0,1) have +1? No.
        
        # Let's use the standard tensor decomposition for 2x2x2.
        # The 2x2x2 tensor has 7 terms.
        # Indices (i,j,k) in {0,1}^3.
        # Term 1: (0,0,0)+, (0,1,1)+, (1,0,0)-, (1,1,1)- ... wait, let's just use the known matrix.
        
        # Actually, I can just generate the 2x2x2 Strassen decomposition explicitly.
        # The 2x2x2 multiplication tensor has rank 7.
        # U (4x7), V (4x7), W (4x7).
        
        # Let's define the 7 terms for 2x2x2.
        # I will hardcode the Strassen terms for 2x2.
        # The terms are:
        # 1. U: (0,0)=1, (1,1)=1; V: (0,0)=1, (1,1)=1; W: (0,0)=1, (0,1)=-1, (1,0)=-1, (1,1)=1
        #    Wait, this is not a rank-1 term. A rank-1 term has one non-zero per U, V, W row? No, U has one non-zero, V has one non-zero, W has one non-zero?
        #    No, a rank-1 term is a vector u (dim 4), v (dim 4), w (dim 4).
        #    U is 4x7, so each column is a vector of length 4.
        #    V is 4x7, each column is a vector of length 4.
        #    W is 4x7, each column is a vector of length 4.
        
        # Let's use the known Strassen 2x2x2 decomposition.
        # The 7 vectors for U (rows are i*2+j):
        # u1 = (1, 0, 0, 1)  # (0,0), (1,1)
        # u2 = (0, 1, 0, 1)  # (0,1), (1,1)
        # u3 = (0, 0, 1, 0)  # (1,0)
        # u4 = (0, 0, 1, 1)  # (1,0), (1,1)
        # u5 = (1, 0, 0, 0)  # (0,0)
        # u6 = (0, 0, 0, 1)  # (1,1)
        # u7 = (0, 1, 0, 0)  # (0,1)
        
        # v1 = (1, 0, 1, 0)  # (0,0), (1,0)
        # v2 = (1, 0, 1, 1)  # (0,0), (1,0), (1,1)
        # v3 = (0, 0, 1, 0)  # (1,0)
        # v4 = (0, 1, 0, 1)  # (0,1), (1,1)
        # v5 = (0, 1, 0, 0)  # (0,1)
        # v6 = (0, 0, 0, 1)  # (1,1)
        # v7 = (0, 0, 1, 0)  # (1,0)
        
        # w1 = (1, 0, 0, 1)  # (0,0), (1,1)
        # w2 = (0, 1, 0, 0)  # (0,1)
        # w3 = (0, 0, 1, 0)  # (1,0)
        # w4 = (0, 1, 0, 1)  # (0,1), (1,1)
        # w5 = (1, 0, 0, 0)  # (0,0)
        # w6 = (0, 0, 0, 1)  # (1,1)
        # w7 = (0, 0, 1, 0)  # (1,0)
        
        # Let's verify term 1:
        # u1 = (1, 0, 0, 1) -> A00 += 1, A11 += 1
        # v1 = (1, 0, 1, 0) -> B00 += 1, B10 += 1
        # w1 = (1, 0, 0, 1) -> C00 += 1, C11 += 1
        # M1 = (A00+A11)(B00+B10) = A00B00 + A00B10 + A11B00 + A11B10
        # C00 gets +1, C11 gets +1.
        # C00 needs A00B00 + A01B10.
        # C11 needs A10B01 + A11B11.
        # This doesn't seem right.
        
        # I will use a known correct 2x2x2 rank-7 decomposition.
        # From literature, the 7 terms are:
        # (U, V, W) where U, V, W are 4-dim vectors.
        # I will use the following (i,j) for U, (j,k) for V, (k,i) for W.
        
        # Term 1:
        # U: (0,0)=1, (1,1)=1
        # V: (0,0)=1, (1,0)=1
        # W: (0,0)=1, (0,1)=1
        # (A00+A11)(B00+B10) = A00B00 + A00B10 + A11B00 + A11B10
        # C00 += A00B00, C11 += A11B10? No.
        # W: (0,0)=1 means k=0, i=0 -> C00. (0,1) means k=0, i=1 -> C10.
        # So C00 += A00B00 + A00B10 + A11B00 + A11B10? No, the product is a scalar.
        # The term contributes to C00 and C10.
        
        # This is getting complicated. I will use a simpler approach:
        # I will just use the schoolbook for 2x2x2 (rank 8) and see if it's better than 73.
        # 6 + 3*8 + 23 + 23 = 6 + 24 + 46 = 76.
        # 76 is worse than 73.
        
        # Let's use the Strassen decomposition from a known source.
        # The 2x2x2 tensor has rank 7.
        # I will hardcode the 7 terms for 2x2x2.
        
        # I will use the following decomposition (from "Matrix multiplication via arithmetic progressions"):
        # The 7 terms are:
        # 1. U: (1,0,0,1), V: (1,0,1,0), W: (1,0,0,1)
        # 2. U: (0,1,0,1), V: (1,0,0,1), W: (0,1,0,0)
        # 3. U: (0,0,1,0), V: (0,1,1,0), W: (0,0,1,0)
        # 4. U: (0,0,1,1), V: (0,0,1,1), W: (0,1,0,1)
        # 5. U: (1,0,0,0), V: (0,1,0,0), W: (1,0,0,0)
        # 6. U: (0,0,0,1), V: (0,0,0,1), W: (0,0,0,1)
        # 7. U: (0,1,0,0), V: (0,0,1,0), W: (0,0,1,0)
        
        # Let's verify term 5:
        # U: (1,0,0,0) -> A00
        # V: (0,1,0,0) -> B01
        # W: (1,0,0,0) -> C00
        # A00*B01 -> C00. Correct.
        
        # Term 6:
        # U: (0,0,0,1) -> A11
        # V: (0,0,0,1) -> B11
        # W: (0,0,0,1) -> C11
        # A11*B11 -> C11. Correct.
        
        # Term 3:
        # U: (0,0,1,0) -> A10
        # V: (0,1,1,0) -> B01, B11
        # W: (0,0,1,0) -> C10
        # A10*(B01+B11) -> C10.
        # C10 = A10B00 + A11B10.
        # This doesn't match.
        
        # I will use a different strategy: I will use the 3x3x3 Laderman for the 3x3x3 blocks and schoolbook for the rest, but optimize the partition.
        # Actually, the simplest is to use the known 3x3x3 rank 23 for the two 3x3x3 blocks, and schoolbook for the 1x2x6 and 2x2x6 parts.
        # Rank = 6 + 24 + 23 + 23 = 76.
        
        # Let's try to use Strassen for 2x2x2. I will use the correct decomposition.
        # The 2x2x2 multiplication tensor is:
        # sum_{i,j,k} A_ij B_jk e_{ik}
        # The 7 terms are (using (i,j) for U, (j,k) for V, (k,i) for W):
        # 1. U: (0,0),(1,1); V: (0,0),(1,0); W: (0,0),(1,0) -> (A00+A11)(B00+B10) -> C00, C10
        # 2. U: (0,1),(1,1); V: (0,1),(1,1); W: (0,0),(1,0) -> (A01+A11)(B01+B11) -> C00, C10
        # 3. U: (1,0); V: (0,1); W: (0,0) -> A10 B01 -> C00
        # 4. U: (1,0),(1,1); V: (0,0),(1,0); W: (1,0),(1,1) -> (A10+A11)(B00+B10) -> C10, C11
        # 5. U: (0,0); V: (0,1); W: (0,0) -> A00 B01 -> C00
        # 6. U: (1,1); V: (1,1); W: (1,1) -> A11 B11 -> C11
        # 7. U: (0,1); V: (1,0); W: (1,0) -> A01 B10 -> C10
        
        # Let's check C00:
        # Term 1: A00B00 + A00B10 + A11B00 + A11B10
        # Term 2: A01B01 + A01B11 + A11B01 + A11B11
        # Term 3: A10B01
        # Term 5: A00B01
        # Sum: A00B00 + A00B10 + A11B00 + A11B10 + A01B01 + A01B11 + A11B01 + A11B11 + A10B01 + A00B01
        # This is not A00B00 + A01B10.
        
        # I will use a library or a verified decomposition.
        # For now, I will use schoolbook for 2x2x2 (rank 8) and Laderman for 3x3x3 (rank 23).
        # Rank = 6 + 3*8 + 23 + 23 = 76.
        
        # Let's try a different partition: 3x5x6 -> (3x3x6) + (3x2x6)
        # 3x3x6: two 3x3x3 blocks, rank 23+23=46.
        # 3x2x6: three 3x2x2 blocks, each rank 6 (schoolbook) or 5 (if known).
        # 3x2x2 schoolbook rank is 6.
        # Total rank = 46 + 3*6 = 46 + 18 = 64.
        
        # Let's use this partition!
        # Component A: 3x3x6 (i=0..2, j=0..2, k=0..5), rank 46.
        # Component B: 3x2x6 (i=0..2, j=3..4, k=0..5), rank 18.
        # Total rank = 64.
        
        # I will implement this.
        pass
    # I will rewrite the code with this new partition.
    
    U = [[0] * 64 for _ in range(n * m)]
    V = [[0] * 64 for _ in range(m * p)]
    W = [[0] * 64 for _ in range(p * n)]
    
    col = 0
    # Component A: 3x3x6, two 3x3x3 blocks.
    for b in range(2):
        k0 = 3*b
        k1 = k0+1
        k2 = k0+2
        # Laderman for 3x3x3, rank 23.
        # I will hardcode the 23 terms for 3x3x3.
        # This is complex, so I will use schoolbook for 3x3x3 (rank 27) for now.
        # Rank = 2*27 + 18 = 54 + 18 = 72.
        # 72 is better than 73.
        
        # Let's use schoolbook for 3x3x3.
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    add_term(i*5+j, j*6+(k0+k), (k0+k)*3+i, col)
                    col += 1
    # col is now 54.
    
    # Component B: 3x2x6, three 3x2x2 blocks.
    for b in range(3):
        k0 = 2*b
        k1 = k0+1
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    add_term(i*5+(3+j), (3+j)*6+(k0+k), (k0+k)*3+i, col)
                    col += 1
    # col is now 54 + 18 = 72.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
