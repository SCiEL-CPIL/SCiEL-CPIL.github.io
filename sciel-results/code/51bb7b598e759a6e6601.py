import numpy as np

def construct():
    n, m, p = 2, 4, 8
    # U: (8, rank), V: (32, rank), W: (16, rank)
    
    # Define Strassen 2x2 @ 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6
    
    # For each Strassen block, we need:
    # - The 4 A indices (in the 2x2 submatrix)
    # - The 4 B indices (in the 2x2 submatrix)
    # - The 4 C indices (in the 2x2 submatrix)
    
    # A indices in the full 2x4 matrix: (i, j) -> i*4+j
    # B indices in the full 4x8 matrix: (j, k) -> j*8+k
    # C indices in the full 2x8 matrix: (i, k) -> k*2+i
    
    # The 8 Strassen blocks:
    # Block 0: A1*B11a -> A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A1*B11b -> A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 2: A1*B12a -> A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    # Block 3: A1*B12b -> A[0:2, 0:2] * B[0:2, 6:8] -> C[0:2, 6:8]
    # Block 4: A2*B21a -> A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 5: A2*B21b -> A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # Block 6: A2*B22a -> A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    # Block 7: A2*B22b -> A[0:2, 2:4] * B[2:4, 6:8] -> C[0:2, 6:8]
    
    # Wait, I need to be more careful.
    # A1 = A[:, 0:2], so A1[i,j] = A[i, j] for j=0,1
    # A2 = A[:, 2:4], so A2[i,j] = A[i, j+2] for j=0,1
    
    # B11 = B[0:2, 0:4], so B11[j,k] = B[j, k] for k=0,1,2,3
    # B11a = B11[:, 0:2] = B[0:2, 0:2]
    # B11b = B11[:, 2:4] = B[0:2, 2:4]
    
    # B12 = B[0:2, 4:8], so B12[j,k] = B[j, k+4] for k=0,1,2,3
    # B12a = B12[:, 0:2] = B[0:2, 4:6]
    # B12b = B12[:, 2:4] = B[0:2, 6:8]
    
    # B21 = B[2:4, 0:4], so B21[j,k] = B[j+2, k] for k=0,1,2,3
    # B21a = B21[:, 0:2] = B[2:4, 0:2]
    # B21b = B21[:, 2:4] = B[2:4, 2:4]
    
    # B22 = B[2:4, 4:8], so B22[j,k] = B[j+2, k+4] for k=0,1,2,3
    # B22a = B22[:, 0:2] = B[2:4, 4:6]
    # B22b = B22[:, 2:4] = B[2:4, 6:8]
    
    # C[:, 0:2] = A1*B11a + A2*B21a
    # C[:, 2:4] = A1*B11b + A2*B21b
    # C[:, 4:6] = A1*B12a + A2*B22a
    # C[:, 6:8] = A1*B12b + A2*B22b
    
    # So the 8 blocks are:
    # (A_sub, B_sub, C_col_offset)
    # A_sub: which 2x2 submatrix of A (0 for A1, 1 for A2)
    # B_sub: which 2x4 submatrix of B and which 2x2 half
    # C_col_offset: which 2 columns of C
    
    # Let me define the blocks explicitly:
    # Each block is (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # where the 2x2 A submatrix is A[a_row_offset:a_row_offset+2, a_col_offset:a_col_offset+2]
    # and the 2x2 B submatrix is B[b_row_offset:b_row_offset+2, b_col_offset:b_col_offset+2]
    # and the 2x2 C submatrix is C[0:2, c_col_offset:c_col_offset+2]
    
    blocks = [
        # (a_col_off, b_row_off, b_col_off, c_col_off)
        (0, 0, 0, 0),  # A1 * B[0:2, 0:2] -> C[:, 0:2]
        (0, 0, 2, 2),  # A1 * B[0:2, 2:4] -> C[:, 2:4]
        (0, 0, 4, 4),  # A1 * B[0:2, 4:6] -> C[:, 4:6]
        (0, 0, 6, 6),  # A1 * B[0:2, 6:8] -> C[:, 6:8]
        (2, 2, 0, 0),  # A2 * B[2:4, 0:2] -> C[:, 0:2]
        (2, 2, 2, 2),  # A2 * B[2:4, 2:4] -> C[:, 2:4]
        (2, 2, 4, 4),  # A2 * B[2:4, 4:6] -> C[:, 4:6]
        (2, 2, 6, 6),  # A2 * B[2:4, 6:8] -> C[:, 6:8]
    ]
    
    # For each block, generate 7 Strassen mults
    # Strassen mults for 2x2 @ 2x2:
    # A = [[a, b], [c, d]] where a=A[ar, ac], b=A[ar, ac+1], c=A[ar+1, ac], d=A[ar+1, ac+1]
    # B = [[e, f], [g, h]] where e=B[br, bc], f=B[br, bc+1], g=B[br+1, bc], h=B[br+1, bc+1]
    
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # C00 = P1+P4-P5+P7 -> C[ar, cc]
    # C01 = P3+P5 -> C[ar, cc+1]
    # C10 = P2+P4 -> C[ar+1, cc]
    # C11 = P1-P2+P3+P6 -> C[ar+1, cc+1]
    
    # U index for A[i,j]: i*4+j
    # V index for B[j,k]: j*8+k
    # W index for C[i,k]: k*2+i
    
    rank = 56
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        # A submatrix indices
        a00 = 0*4 + a_col_off      # A[0, a_col_off]
        a01 = 0*4 + a_col_off + 1  # A[0, a_col_off+1]
        a10 = 1*4 + a_col_off      # A[1, a_col_off]
        a11 = 1*4 + a_col_off + 1  # A[1, a_col_off+1]
        
        # B submatrix indices
        b00 = b_row_off*8 + b_col_off
        b01 = b_row_off*8 + b_col_off + 1
        b10 = (b_row_off+1)*8 + b_col_off
        b11 = (b_row_off+1)*8 + b_col_off + 1
        
        # C submatrix indices
        c00 = c_col_off*2 + 0  # C[0, c_col_off]
        c01 = (c_col_off+1)*2 + 0  # C[0, c_col_off+1]
        c10 = c_col_off*2 + 1  # C[1, c_col_off]
        c11 = (c_col_off+1)*2 + 1  # C[1, c_col_off+1]
        
        # P1 = (a00+a11)(b00+b11)
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1  # C00 += P1
        W[c11, col] += 1  # C11 += P1
        col += 1
        
        # P2 = (a10+a11)*b00
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1  # C10 += P2
        W[c11, col] += -1  # C11 -= P2
        col += 1
        
        # P3 = a00*(b01-b11)
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1  # C01 += P3
        W[c11, col] += 1  # C11 += P3
        col += 1
        
        # P4 = a11*(b10-b00)
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1  # C00 += P4
        W[c10, col] += 1  # C10 += P4
        col += 1
        
        # P5 = (a00+a01)*b11
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1  # C00 -= P5
        W[c01, col] += 1  # C01 += P5
        col += 1
        
        # P6 = (a10-a00)*(b00+b01)
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1  # C11 += P6
        col += 1
        
        # P7 = (a01-a11)*(b10+b11)
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1  # C00 += P7
        col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
