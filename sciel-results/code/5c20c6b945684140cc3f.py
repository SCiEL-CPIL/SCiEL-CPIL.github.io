import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # Strassen decomposition for 2x4 @ 4x8
    # Split A into A1 (2x2) and A2 (2x2)
    # Split B into B11, B12, B13, B14 (2x2) for A1
    # and B21, B22, B23, B24 (2x2) for A2
    # C = A1 @ B1 + A2 @ B2 where B1 = [B11 B12 B13 B14] and B2 = [B21 B22 B23 B24]
    # Actually, B is 4x8.
    # B = [[B11, B12, B13, B14],
    #      [B21, B22, B23, B24]]
    # where B_ij are 2x2 blocks.
    # A = [A1, A2] where A1, A2 are 2x2.
    # C = A1 @ B1 + A2 @ B2
    # B1 is 2x8, B2 is 2x8.
    # We compute A1 @ B1 and A2 @ B2 separately.
    # Each is a 2x2 @ 2x8 multiplication.
    # 2x2 @ 2x8 can be done with 4 Strassen 2x2 @ 2x2 multiplications?
    # No, 2x2 @ 2x8 is equivalent to 4 independent 2x2 @ 2x2 multiplications if we split B into 4 2x2 blocks.
    # So each 2x2 @ 2x8 uses 4 * 7 = 28 multiplications.
    # Total 2 * 28 = 56.
    
    rank = 56
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    # Indices:
    # U[i*m + j, r] -> A[i, j]
    # V[j*p + k, r] -> B[j, k]
    # W[k*n + i, r] -> C[i, k]
    
    # Strassen 2x2 @ 2x2:
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
    
    col = 0
    # Blocks:
    # A1 (cols 0,1) interacts with B rows 0,1 (B11, B12, B13, B14)
    # A2 (cols 2,3) interacts with B rows 2,3 (B21, B22, B23, B24)
    
    # For A1:
    # A1 indices: U[0, :], U[1, :], U[4, :], U[5, :] (A[0,0], A[0,1], A[1,0], A[1,1])
    # Wait, A is 2x4. A[0,0], A[0,1], A[1,0], A[1,1] are A1.
    # A[0,2], A[0,3], A[1,2], A[1,3] are A2.
    # U indices:
    # A[0,0] -> 0*4+0 = 0
    # A[0,1] -> 0*4+1 = 1
    # A[1,0] -> 1*4+0 = 4
    # A[1,1] -> 1*4+1 = 5
    # A[0,2] -> 0*4+2 = 2
    # A[0,3] -> 0*4+3 = 3
    # A[1,2] -> 1*4+2 = 6
    # A[1,3] -> 1*4+3 = 7
    
    # B indices:
    # B[j, k] -> j*8 + k
    # B11: j=0,1; k=0,1 -> V[0], V[1], V[8], V[9]
    # B12: j=0,1; k=2,3 -> V[2], V[3], V[10], V[11]
    # B13: j=0,1; k=4,5 -> V[4], V[5], V[12], V[13]
    # B14: j=0,1; k=6,7 -> V[6], V[7], V[14], V[15]
    # B21: j=2,3; k=0,1 -> V[16], V[17], V[24], V[25]
    # B22: j=2,3; k=2,3 -> V[18], V[19], V[26], V[27]
    # B23: j=2,3; k=4,5 -> V[20], V[21], V[28], V[29]
    # B24: j=2,3; k=6,7 -> V[22], V[23], V[30], V[31]
    
    # C indices:
    # C[i, k] -> k*2 + i
    # C[0,0] -> 0
    # C[0,1] -> 2
    # C[1,0] -> 1
    # C[1,1] -> 3
    # ...
    # C[0, k] -> 2k
    # C[1, k] -> 2k+1
    
    def add_strassen_block(a_indices, b_indices, c_indices):
        nonlocal col
        a00, a01, a10, a11 = a_indices
        b00, b01, b10, b11 = b_indices
        c00, c01, c10, c11 = c_indices
        
        # P1 = (a00+a11)(b00+b11)
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P2 = (a10+a11)*b00
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] += -1
        col += 1
        
        # P3 = a00*(b01-b11)
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P4 = a11*(b10-b00)
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # P5 = (a00+a01)*b11
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        # P6 = (a10-a00)*(b00+b01)
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P7 = (a01-a11)*(b10+b11)
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
    # A1 indices: 0, 1, 4, 5
    a1 = [0, 1, 4, 5]
    # A2 indices: 2, 3, 6, 7
    a2 = [2, 3, 6, 7]
    
    # B11: 0, 1, 8, 9
    b11 = [0, 1, 8, 9]
    # B12: 2, 3, 10, 11
    b12 = [2, 3, 10, 11]
    # B13: 4, 5, 12, 13
    b13 = [4, 5, 12, 13]
    # B14: 6, 7, 14, 15
    b14 = [6, 7, 14, 15]
    
    # B21: 16, 17, 24, 25
    b21 = [16, 17, 24, 25]
    # B22: 18, 19, 26, 27
    b22 = [18, 19, 26, 27]
    # B23: 20, 21, 28, 29
    b23 = [20, 21, 28, 29]
    # B24: 22, 23, 30, 31
    b24 = [22, 23, 30, 31]
    
    # C indices for k=0: 0, 2, 1, 3
    # C indices for k=1: 2, 4, 3, 5
    # C indices for k=2: 4, 6, 5, 7
    # C indices for k=3: 6, 8, 7, 9
    # C indices for k=4: 8, 10, 9, 11
    # C indices for k=5: 10, 12, 11, 13
    # C indices for k=6: 12, 14, 13, 15
    # C indices for k=7: 14, 16, 15, 17
    
    # Block 1: A1 * B11 -> C[:, 0:2]
    c0 = [0, 2, 1, 3] # C[0,0], C[0,1], C[1,0], C[1,1]
    add_strassen_block(a1, b11, c0)
    
    # Block 2: A1 * B12 -> C[:, 2:4]
    c1 = [4, 6, 5, 7] # C[0,2], C[0,3], C[1,2], C[1,3]
    add_strassen_block(a1, b12, c1)
    
    # Block 3: A1 * B13 -> C[:, 4:6]
    c2 = [8, 10, 9, 11] # C[0,4], C[0,5], C[1,4], C[1,5]
    add_strassen_block(a1, b13, c2)
    
    # Block 4: A1 * B14 -> C[:, 6:8]
    c3 = [12, 14, 13, 15] # C[0,6], C[0,7], C[1,6], C[1,7]
    add_strassen_block(a1, b14, c3)
    
    # Block 5: A2 * B21 -> C[:, 0:2]
    add_strassen_block(a2, b21, c0)
    
    # Block 6: A2 * B22 -> C[:, 2:4]
    add_strassen_block(a2, b22, c1)
    
    # Block 7: A2 * B23 -> C[:, 4:6]
    add_strassen_block(a2, b23, c2)
    
    # Block 8: A2 * B24 -> C[:, 6:8]
    add_strassen_block(a2, b24, c3)
    
    assert col == 56
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
