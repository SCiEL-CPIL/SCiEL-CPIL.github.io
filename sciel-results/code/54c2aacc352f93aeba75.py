import time

def construct():
    n, m, p = 3, 3, 8
    # We use a block decomposition:
    # A = [[A00 (2x2), A01 (2x1)],
    #      [A10 (1x2), A11 (1x1)]]
    # B = [[B00 (2x8)],
    #      [B10 (1x8)]]
    #
    # C = A * B
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    
    # Ranks:
    # A00*B00: 2x2x8. Use Strassen (4 blocks of 2x2x2, rank 7 each) -> 28
    # A01*B10: 2x1x8. Rank 16 (Schoolbook)
    # A10*B00: 1x2x8. Rank 16 (Schoolbook)
    # A11*B10: 1x1x8. Rank 8 (Schoolbook)
    # Total Rank = 28 + 16 + 16 + 8 = 68
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen_2x2x2(u_offset, v_offset, w_offset, b_col_offset):
        nonlocal col
        # Strassen 7 products for 2x2 * 2x2
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        # C00 = P1+P4-P5+P7
        # C01 = P3+P5
        # C10 = P2+P4
        # C11 = P1-P2+P3+P6
        
        # Indices in U (A part):
        # A00: u_offset + 0
        # A01: u_offset + 1
        # A10: u_offset + 2
        # A11: u_offset + 3
        
        # Indices in V (B part):
        # B00: v_offset + 0 * 2 + 0
        # B01: v_offset + 0 * 2 + 1
        # B10: v_offset + 1 * 2 + 0
        # B11: v_offset + 1 * 2 + 1
        
        # Indices in W (C part):
        # C00: w_offset + 0
        # C01: w_offset + 1
        # C10: w_offset + 2
        # C11: w_offset + 3
        
        # P1
        U[u_offset+0][col] += 1
        U[u_offset+3][col] += 1
        V[v_offset+0][col] += 1
        V[v_offset+3][col] += 1
        W[w_offset+0][col] += 1
        W[w_offset+3][col] += 1
        col += 1
        
        # P2
        U[u_offset+2][col] += 1
        U[u_offset+3][col] += 1
        V[v_offset+0][col] += 1
        W[w_offset+2][col] += 1
        col += 1
        
        # P3
        U[u_offset+0][col] += 1
        V[v_offset+1][col] += 1
        V[v_offset+3][col] -= 1
        W[w_offset+1][col] += 1
        col += 1
        
        # P4
        U[u_offset+3][col] += 1
        V[v_offset+2][col] += 1
        V[v_offset+0][col] -= 1
        W[w_offset+0][col] += 1
        W[w_offset+3][col] += 1
        col += 1
        
        # P5
        U[u_offset+0][col] += 1
        U[u_offset+1][col] += 1
        V[v_offset+3][col] += 1
        W[w_offset+0][col] -= 1
        W[w_offset+1][col] += 1
        col += 1
        
        # P6
        U[u_offset+2][col] += 1
        U[u_offset+0][col] -= 1
        V[v_offset+0][col] += 1
        V[v_offset+1][col] += 1
        W[w_offset+3][col] += 1
        col += 1
        
        # P7
        U[u_offset+1][col] += 1
        U[u_offset+3][col] -= 1
        V[v_offset+2][col] += 1
        V[v_offset+3][col] += 1
        W[w_offset+0][col] += 1
        col += 1

    # Block A00 (2x2) is at A[0:2, 0:2]
    # U indices: 0,1,2,3
    # V indices for B00 (2x8):
    # B is 3x8.
    # B00 is rows 0,1 of B. Cols 0-7.
    # V index = j * p + k.
    # j=0, k=0..7 -> 0..7
    # j=1, k=0..7 -> 8..15
    
    # We need 4 Strassen blocks for the 8 columns.
    # Block 0: B cols 0,1. V indices 0,1,8,9.
    # Block 1: B cols 2,3. V indices 2,3,10,11.
    # Block 2: B cols 4,5. V indices 4,5,12,13.
    # Block 3: B cols 6,7. V indices 6,7,14,15.
    
    # W indices for C00 (2x8):
    # C is 3x8.
    # C00 is rows 0,1 of C. Cols 0-7.
    # W index = k * n + i.
    # k=0, i=0,1 -> 0,1
    # k=1, i=0,1 -> 3,4
    # k=2, i=0,1 -> 6,7
    # k=3, i=0,1 -> 9,10
    # k=4, i=0,1 -> 12,13
    # k=5, i=0,1 -> 15,16
    # k=6, i=0,1 -> 18,19
    # k=7, i=0,1 -> 21,22
    
    for b_block in range(4):
        # V offset for this block
        v_off = b_block * 2
        # W offset for this block
        w_off = b_block * 6 # k increases by 2, so k*n increases by 6
        
        # Local V indices: 0,1,2,3 correspond to global v_off+0, v_off+1, v_off+8, v_off+9?
        # No, V index is j*p+k.
        # For j=0: k=2*b_block, 2*b_block+1. Indices: 2*b_block, 2*b_block+1.
        # For j=1: k=2*b_block, 2*b_block+1. Indices: 8 + 2*b_block, 8 + 2*b_block+1.
        
        # My add_strassen_2x2x2 assumes V indices are contiguous 0,1,2,3.
        # I need to map them.
        # Let's just write the terms manually for each block to avoid offset bugs.
        
        # P1
        U[0][col] += 1; U[3][col] += 1
        V[v_off][col] += 1; V[8+v_off][col] += 1
        W[w_off][col] += 1; W[w_off+3][col] += 1
        col += 1
        # P2
        U[2][col] += 1; U[3][col] += 1
        V[v_off][col] += 1
        W[w_off+2][col] += 1
        col += 1
        # P3
        U[0][col] += 1
        V[v_off+1][col] += 1; V[8+v_off+1][col] -= 1
        W[w_off+1][col] += 1
        col += 1
        # P4
        U[3][col] += 1
        V[8+v_off][col] += 1; V[v_off][col] -= 1
        W[w_off][col] += 1; W[w_off+3][col] += 1
        col += 1
        # P5
        U[0][col] += 1; U[1][col] += 1
        V[8+v_off+1][col] += 1
        W[w_off][col] -= 1; W[w_off+1][col] += 1
        col += 1
        # P6
        U[2][col] += 1; U[0][col] -= 1
        V[v_off][col] += 1; V[v_off+1][col] += 1
        W[w_off+3][col] += 1
        col += 1
        # P7
        U[1][col] += 1; U[3][col] -= 1
        V[8+v_off][col] += 1; V[8+v_off+1][col] += 1
        W[w_off][col] += 1
        col += 1

    # Next: A01*B10 (2x1x8)
    # A01 is A[0:2, 2]. U indices: 2, 3? No.
    # A is 3x3.
    # A00: rows 0,1 cols 0,1. U: 0,1,2,3.
    # A01: rows 0,1 col 2. U: 2, 3?
    # U index = i*m + j.
    # i=0, j=2 -> 2.
    # i=1, j=2 -> 5.
    # So U indices are 2 and 5.
    
    # B10 is B[2, 0:8]. V indices: 2*8 + k = 16+k. k=0..7.
    
    # C01 is C[0:2, 0:8]. W indices: k*3 + i.
    # i=0 -> k*3.
    # i=1 -> k*3 + 1.
    
    for k in range(8):
        # C[0,k] += A[0,2] * B[2,k]
        U[2][col] += 1
        V[16+k][col] += 1
        W[k*3][col] += 1
        col += 1
        # C[1,k] += A[1,2] * B[2,k]
        U[5][col] += 1
        V[16+k][col] += 1
        W[k*3+1][col] += 1
        col += 1

    # Next: A10*B00 (1x2x8)
    # A10 is A[2, 0:2]. U indices: 2*3+0=6, 2*3+1=7.
    # B00 is B[0:2, 0:8]. V indices: 0..15.
    # C10 is C[2, 0:8]. W indices: k*3 + 2.
    
    for k in range(8):
        # C[2,k] += A[2,0] * B[0,k]
        U[6][col] += 1
        V[k][col] += 1
        W[k*3+2][col] += 1
        col += 1
        # C[2,k] += A[2,1] * B[1,k]
        U[7][col] += 1
        V[8+k][col] += 1
        W[k*3+2][col] += 1
        col += 1

    # Next: A11*B10 (1x1x8)
    # A11 is A[2,2]. U index: 8.
    # B10 is B[2, 0:8]. V indices: 16..23.
    # C11 is C[2, 0:8]. W indices: k*3 + 2.
    
    for k in range(8):
        U[8][col] += 1
        V[16+k][col] += 1
        W[k*3+2][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
