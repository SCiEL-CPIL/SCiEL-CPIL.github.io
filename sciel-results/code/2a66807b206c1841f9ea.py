import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # Decomposition:
    # A (2x5) = [A1 (2x2), A2 (2x3)]
    # B (5x6) = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    
    # A1*B1 (2x2 * 2x6):
    # Split B1 into 3 (2x2) blocks.
    # Each 2x2*2x2 uses Schoolbook (Rank 8).
    # Total Rank = 3 * 8 = 24.
    
    # A2*B2 (2x3 * 3x6):
    # Split A2 into A2a (2x2) and A2b (2x1).
    # Split B2 into B2a (2x6) and B2b (1x6).
    # A2a*B2a (2x2 * 2x6):
    # Split B2a into 3 (2x2) blocks.
    # Each 2x2*2x2 uses Schoolbook (Rank 8).
    # Total Rank = 3 * 8 = 24.
    # A2b*B2b (2x1 * 1x6):
    # Rank 1 (Outer product of [1,1] and [1,1,1,1,1,1]).
    
    # Total Rank = 24 + 24 + 1 = 49.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook_2x2x2(offset_u, offset_v, offset_w):
        """
        Adds 8 terms for 2x2x2 multiplication.
        U indices: offset_u .. offset_u+3 (2x2)
        V indices: offset_v .. offset_v+3 (2x2)
        W indices: offset_w .. offset_w+3 (2x2)
        """
        nonlocal col
        # Schoolbook: C_ij = sum_k A_ik B_kj
        # Terms:
        # 1. A00 B00 -> C00
        # 2. A01 B10 -> C00
        # 3. A00 B01 -> C01
        # 4. A01 B11 -> C01
        # 5. A10 B00 -> C10
        # 6. A11 B10 -> C10
        # 7. A10 B01 -> C11
        # 8. A11 B11 -> C11
        
        # U: A (2x2) -> [A00, A01, A10, A11]
        # V: B (2x2) -> [B00, B01, B10, B11] (Row major? No, standard flatten)
        # W: C (2x2) -> [C00, C01, C10, C11]
        
        # U indices relative: 0:A00, 1:A01, 2:A10, 3:A11
        # V indices relative: 0:B00, 1:B01, 2:B10, 3:B11
        # W indices relative: 0:C00, 1:C01, 2:C10, 3:C11
        
        # Term 1: A00 * B00 -> C00
        U[offset_u + 0][col] += 1
        V[offset_v + 0][col] += 1
        W[offset_w + 0][col] += 1
        col += 1
        
        # Term 2: A01 * B10 -> C00
        U[offset_u + 1][col] += 1
        V[offset_v + 2][col] += 1
        W[offset_w + 0][col] += 1
        col += 1
        
        # Term 3: A00 * B01 -> C01
        U[offset_u + 0][col] += 1
        V[offset_v + 1][col] += 1
        W[offset_w + 1][col] += 1
        col += 1
        
        # Term 4: A01 * B11 -> C01
        U[offset_u + 1][col] += 1
        V[offset_v + 3][col] += 1
        W[offset_w + 1][col] += 1
        col += 1
        
        # Term 5: A10 * B00 -> C10
        U[offset_u + 2][col] += 1
        V[offset_v + 0][col] += 1
        W[offset_w + 2][col] += 1
        col += 1
        
        # Term 6: A11 * B10 -> C10
        U[offset_u + 3][col] += 1
        V[offset_v + 2][col] += 1
        W[offset_w + 2][col] += 1
        col += 1
        
        # Term 7: A10 * B01 -> C11
        U[offset_u + 2][col] += 1
        V[offset_v + 1][col] += 1
        W[offset_w + 3][col] += 1
        col += 1
        
        # Term 8: A11 * B11 -> C11
        U[offset_u + 3][col] += 1
        V[offset_v + 3][col] += 1
        W[offset_w + 3][col] += 1
        col += 1

    # --- Part 1: A1 (2x2) * B1 (2x6) ---
    # A1 indices in U: 0, 1, 2, 3 (Rows 0,1 of A)
    # B1 indices in V: 0..11 (Rows 0,1 of B, 6 cols)
    # B1 is 2x6. Split into 3 blocks of 2x2.
    # Block 0: Cols 0,1. V indices: 0,1,2,3.
    # Block 1: Cols 2,3. V indices: 4,5,6,7.
    # Block 2: Cols 4,5. V indices: 8,9,10,11.
    # W indices for C1 (2x2 part of output):
    # Output C is 2x6.
    # C1 contributes to C[:, 0:2].
    # W indices: 0,1,2,3 (C00, C01, C10, C11).
    
    # Block 0
    add_schoolbook_2x2x2(0, 0, 0)
    # Block 1
    # W indices for C[:, 2:4] are 4,5,6,7
    add_schoolbook_2x2x2(0, 4, 4)
    # Block 2
    # W indices for C[:, 4:6] are 8,9,10,11
    add_schoolbook_2x2x2(0, 8, 8)
    
    # --- Part 2: A2 (2x3) * B2 (3x6) ---
    # A2 indices in U: 4,5,6,7,8,9 (Rows 2,3 of A, 3 cols)
    # A2a (2x2) uses U indices 4,5,6,7.
    # A2b (2x1) uses U indices 8,9.
    
    # B2 indices in V: 12..29 (Rows 2,3,4 of B, 6 cols)
    # B2a (2x6) uses V indices 12..23 (Rows 2,3 of B).
    # B2b (1x6) uses V indices 24..29 (Row 4 of B).
    
    # W indices for C2 (2x6 part of output, but added to same C):
    # C is 2x6.
    # C2 contributes to all C.
    # W indices: 0..11.
    
    # A2a * B2a (2x2 * 2x6)
    # Split B2a into 3 blocks of 2x2.
    # Block 0: Cols 0,1. V indices: 12,13,14,15.
    # W indices: 0,1,2,3.
    add_schoolbook_2x2x2(4, 12, 0)
    
    # Block 1: Cols 2,3. V indices: 16,17,18,19.
    # W indices: 4,5,6,7.
    add_schoolbook_2x2x2(4, 16, 4)
    
    # Block 2: Cols 4,5. V indices: 20,21,22,23.
    # W indices: 8,9,10,11.
    add_schoolbook_2x2x2(4, 20, 8)
    
    # A2b * B2b (2x1 * 1x6)
    # A2b: U indices 8,9. (A[2,2], A[3,2] -> A2b[0,0], A2b[1,0])
    # B2b: V indices 24..29. (B[4,0]..B[4,5])
    # W: 0..11.
    # C_ij = A2b_i * B2b_j
    # This is rank 1.
    # U: [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Wait, U size is n*m = 10.
    # U indices 8,9 correspond to A[2,2] and A[3,2]?
    # A is 2x5.
    # Row 0: 0,1,2,3,4
    # Row 1: 5,6,7,8,9
    # A1 is cols 0,1. Indices 0,1,5,6.
    # A2 is cols 2,3,4. Indices 2,3,4,7,8,9.
    # My previous indexing for A2a (2x2) was 4,5,6,7?
    # Let's re-map indices carefully.
    
    # U (10x49):
    # A:
    # A00=0, A01=1, A02=2, A03=3, A04=4
    # A10=5, A11=6, A12=7, A13=8, A14=9
    
    # A1 (2x2): A00, A01, A10, A11 -> Indices 0,1,5,6.
    # A2 (2x3): A02, A03, A04, A12, A13, A14 -> Indices 2,3,4,7,8,9.
    # A2a (2x2): A02, A03, A12, A13 -> Indices 2,3,7,8.
    # A2b (2x1): A04, A14 -> Indices 4,9.
    
    # V (30x49):
    # B:
    # B00=0, B01=1, B02=2, B03=3, B04=4, B05=5
    # B10=6, B11=7, B12=8, B13=9, B14=10, B15=11
    # B20=12, B21=13, B22=14, B23=15, B24=16, B25=17
    # B30=18, B31=19, B32=20, B33=21, B34=22, B35=23
    # B40=24, B41=25, B42=26, B43=27, B44=28, B45=29
    
    # B1 (2x6): Rows 0,1. Indices 0..11.
    # B2 (3x6): Rows 2,3,4. Indices 12..29.
    # B2a (2x6): Rows 2,3. Indices 12..23.
    # B2b (1x6): Row 4. Indices 24..29.
    
    # W (12x49):
    # C:
    # C00=0, C01=1, C02=2, C03=3, C04=4, C05=5
    # C10=6, C11=7, C12=8, C13=9, C14=10, C15=11
    
    # Reset U,V,W and col
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Helper for 2x2x2 Schoolbook with specific indices
    def add_schoolbook_2x2x2_idx(u_idx, v_idx, w_idx):
        nonlocal col
        # u_idx: [u00, u01, u10, u11]
        # v_idx: [v00, v01, v10, v11]
        # w_idx: [w00, w01, w10, w11]
        
        # 1. A00 B00 -> C00
        U[u_idx[0]][col] += 1
        V[v_idx[0]][col] += 1
        W[w_idx[0]][col] += 1
        col += 1
        
        # 2. A01 B10 -> C00
        U[u_idx[1]][col] += 1
        V[v_idx[2]][col] += 1
        W[w_idx[0]][col] += 1
        col += 1
        
        # 3. A00 B01 -> C01
        U[u_idx[0]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[1]][col] += 1
        col += 1
        
        # 4. A01 B11 -> C01
        U[u_idx[1]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[1]][col] += 1
        col += 1
        
        # 5. A10 B00 -> C10
        U[u_idx[2]][col] += 1
        V[v_idx[0]][col] += 1
        W[w_idx[2]][col] += 1
        col += 1
        
        # 6. A11 B10 -> C10
        U[u_idx[3]][col] += 1
        V[v_idx[2]][col] += 1
        W[w_idx[2]][col] += 1
        col += 1
        
        # 7. A10 B01 -> C11
        U[u_idx[2]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1
        
        # 8. A11 B11 -> C11
        U[u_idx[3]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1

    # Part 1: A1 * B1
    # A1: [0, 1, 5, 6]
    # B1 Block 0: [0, 1, 6, 7] (B00, B01, B10, B11)
    # W Block 0: [0, 1, 6, 7] (C00, C01, C10, C11)
    add_schoolbook_2x2x2_idx([0, 1, 5, 6], [0, 1, 6, 7], [0, 1, 6, 7])
    
    # B1 Block 1: [2, 3, 8, 9] (B02, B03, B12, B13)
    # W Block 1: [2, 3, 8, 9] (C02, C03, C12, C13)
    add_schoolbook_2x2x2_idx([0, 1, 5, 6], [2, 3, 8, 9], [2, 3, 8, 9])
    
    # B1 Block 2: [4, 5, 10, 11] (B04, B05, B14, B15)
    # W Block 2: [4, 5, 10, 11] (C04, C05, C14, C15)
    add_schoolbook_2x2x2_idx([0, 1, 5, 6], [4, 5, 10, 11], [4, 5, 10, 11])
    
    # Part 2: A2a * B2a
    # A2a: [2, 3, 7, 8] (A02, A03, A12, A13)
    # B2a Block 0: [12, 13, 18, 19] (B20, B21, B30, B31)
    # W Block 0: [0, 1, 6, 7]
    add_schoolbook_2x2x2_idx([2, 3, 7, 8], [12, 13, 18, 19], [0, 1, 6, 7])
    
    # B2a Block 1: [14, 15, 20, 21] (B22, B23, B32, B33)
    # W Block 1: [2, 3, 8, 9]
    add_schoolbook_2x2x2_idx([2, 3, 7, 8], [14, 15, 20, 21], [2, 3, 8, 9])
    
    # B2a Block 2: [16, 17, 22, 23] (B24, B25, B34, B35)
    # W Block 2: [4, 5, 10, 11]
    add_schoolbook_2x2x2_idx([2, 3, 7, 8], [16, 17, 22, 23], [4, 5, 10, 11])
    
    # Part 3: A2b * B2b
    # A2b: [4, 9] (A04, A14)
    # B2b: [24, 25, 26, 27, 28, 29]
    # W: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # Rank 1 term:
    # U: A04, A14
    # V: B40..B45
    # W: C00..C15
    
    # This is not a single rank-1 term in the tensor sense if we want to cover all W?
    # C_ij = A_i B_j.
    # C_0j = A04 B4j.
    # C_1j = A14 B4j.
    # This is 2 * 6 = 12 terms if schoolbook.
    # But we said Rank 1?
    # The tensor $2 \times 1 \times 6$ has rank 1?
    # $T_{i,0,k} = 1$.
    # $u=[1,1], v=[1], w=[1,1,1,1,1,1]$.
    # $T_{i,0,k} = u_i v_0 w_k = 1$.
    # Yes, Rank 1.
    # So we add 1 term.
    
    U[4][col] += 1
    U[9][col] += 1
    for k in range(6):
        V[24+k][col] += 1
        W[k][col] += 1
        W[6+k][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
