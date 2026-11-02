import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 71
    rank = 71
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col_offset = 0
    
    # Split m=5 into 4+1, p=5 into 4+1
    # A is 3x5, B is 5x5
    # A1 (3x4), A2 (3x1)
    # B11 (4x4), B12 (4x1)
    # B21 (1x4), B22 (1x1)
    
    # Term 1: A1 * B11 (3x4 * 4x4)
    # Decomposed into 4 blocks of 3x2 * 2x2
    # A1 = [A1a (3x2), A1b (3x2)]
    # B11 = [[B11a (2x2), B11b (2x2)], [B11c (2x2), B11d (2x2)]]
    
    # We need a rank 11 decomposition for 3x2x2
    def get_3x2x2():
        # Returns U (3*2, r), V (2*2, r), W (2*3, r)
        # Split 3x2 into 2x2 and 1x2
        # U1 (2x2, 7), V1 (2x2, 7), W1 (2x2, 7) for 2x2x2
        # U2 (1x2, 4), V2 (2x2, 4), W2 (2x1, 4) for 1x2x2
        r1 = 7
        r2 = 4
        r = r1 + r2
        u = np.zeros((6, r))
        v = np.zeros((4, r))
        w = np.zeros((6, r)) # 2x3 -> 6 elements
        
        # 2x2x2 Strassen-like or just schoolbook?
        # Schoolbook 2x2x2 is 4. Strassen is 7.
        # Wait, Strassen is 7 mults.
        # Let's use Strassen for 2x2x2.
        # But Strassen is for 2x2 * 2x2.
        # The tensor indices are i (2), j (2), k (2).
        # U[i,j, c], V[j,k, c], W[k,i, c].
        
        # Strassen decomposition for 2x2x2:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # We need to map these to U, V, W.
        # U is for A (2x2). V is for B (2x2). W is for C (2x2).
        # Note: W index is k*n + i. Here n=2.
        
        u1 = np.zeros((4, r1))
        v1 = np.zeros((4, r1))
        w1 = np.zeros((4, r1))
        
        # M1
        u1[0][0] = 1; u1[3][0] = 1 # a11, a22
        v1[0][0] = 1; v1[3][0] = 1 # b11, b22
        w1[0][0] = 1; w1[3][0] = 1 # c11, c22 (k=0,i=0 and k=1,i=1)
        # Wait, c11 is k=0,i=0. c22 is k=1,i=1.
        # W index: k*n + i.
        # c11: 0*2+0=0. c22: 1*2+1=3.
        
        # M2
        u1[2][1] = 1; u1[3][1] = 1 # a21, a22
        v1[0][1] = 1 # b11
        w1[2][1] = 1 # c21 (k=1,i=0) -> 1*2+0=2
        
        # M3
        u1[0][2] = 1 # a11
        v1[1][2] = 1; v1[3][2] = -1 # b12, b22
        w1[1][2] = 1 # c12 (k=0,i=1) -> 0*2+1=1
        
        # M4
        u1[3][3] = 1 # a22
        v1[2][3] = 1; v1[0][3] = -1 # b21, b11
        w1[0][3] = 1; w1[2][3] = 1 # c11, c21
        
        # M5
        u1[0][4] = 1; u1[1][4] = 1 # a11, a12
        v1[3][4] = 1 # b22
        w1[0][4] = -1; w1[1][4] = 1 # c11, c12
        
        # M6
        u1[2][5] = 1; u1[0][5] = -1 # a21, a11
        v1[0][5] = 1; v1[1][5] = 1 # b11, b12
        w1[3][5] = 1 # c22
        
        # M7
        u1[1][6] = 1; u1[3][6] = -1 # a12, a22
        v1[2][6] = 1; v1[3][6] = 1 # b21, b22
        w1[1][6] = 1; w1[3][6] = -1 # c12, c22
        
        u[:, :r1] = u1
        v[:, :r1] = v1
        w[:, :r1] = w1
        
        # 1x2x2 part
        # A (1x2), B (2x2), C (1x2)
        # u2 (2, 4), v2 (4, 4), w2 (2, 4)
        # C_j = A_1 B_1j + A_2 B_2j
        # 4 mults.
        u2 = np.zeros((2, r2))
        v2 = np.zeros((4, r2))
        w2 = np.zeros((2, r2))
        
        # c1 = a1 b11 + a2 b21
        u2[0][0] = 1
        v2[0][0] = 1; v2[2][0] = 1 # b11, b21
        w2[0][0] = 1 # c1 (k=0, i=0? No, C is 1x2. Index is just col?
        # Wait, W shape is (p*n, rank). Here p=2, n=1.
        # W index: k*n + i. k is 0..1, i is 0.
        # c1 corresponds to k=0? No.
        # C is 1x2. Rows=1, Cols=2.
        # In our convention W[k*n+i, r].
        # Here n=1, p=2.
        # C_{i,k}. i=0, k=0,1.
        # W index: k*1 + 0 = k.
        # So w2[0] is c1, w2[1] is c2.
        
        u2[0][1] = 1
        v2[1][1] = 1; v2[3][1] = 1 # b12, b22
        w2[1][1] = 1
        
        u2[1][2] = 1
        v2[0][2] = 1; v2[2][2] = 1
        w2[0][2] = 1
        
        u2[1][3] = 1
        v2[1][3] = 1; v2[3][3] = 1
        w2[1][3] = 1
        
        # Map to u, v, w
        # u2 corresponds to A rows 2,3 (0-indexed 2,3 in 3x2 block)
        u[2:4, r1:r1+r2] = u2
        v[:, r1:r1+r2] = v2
        w[0:2, r1:r1+r2] = w2 # C is 2x1? No, C is 1x2.
        # Wait, W is (p*n, rank). p=2, n=1.
        # The output C is 1x2.
        # The indices in W are k*n + i.
        # k=0, i=0 -> 0.
        # k=1, i=0 -> 1.
        # So w2 should be 2x4.
        # My w2 is 2x4.
        # But w in get_3x2x2 is 6x11.
        # The first 4 rows of w correspond to the 2x2 part (indices 0,1,2,3).
        # The last 2 rows of w correspond to the 1x2 part?
        # No, the 1x2 part output is 1x2.
        # In the 3x2x2 block, the output is 3x2.
        # Rows 0,1 are from 2x2x2.
        # Row 2 is from 1x2x2.
        # Cols 0,1 are the output columns.
        # So W index for row 2, col 0 is k=0, i=2 -> 0*3+2=2?
        # No, n=3 in the full problem.
        # In the sub-problem 3x2x2, n=3, m=2, p=2.
        # W shape (2*3, 11) = (6, 11).
        # W[k*n+i, c].
        # For the 2x2 part (rows 0,1 of A), i=0,1.
        # For the 1x2 part (row 2 of A), i=2.
        # So w2 indices should be k*3 + 2.
        # k=0 -> 2. k=1 -> 5.
        
        w[2, r1:r1+r2] = w2[0]
        w[5, r1:r1+r2] = w2[1]
        
        return u, v, w
    
    u_322, v_322, w_322 = get_322()
    
    # Place 4 blocks of 3x2x2 into A1*B11
    # A1 indices: 0-3 (m), B11 indices: 0-3 (m) x 0-3 (p)
    # A1a: m 0-1. A1b: m 2-3.
    # B11a: m 0-1, p 0-1. B11b: m 0-1, p 2-3.
    # B11c: m 2-3, p 0-1. B11d: m 2-3, p 2-3.
    
    # Block 1: A1a (m0-1) * B11a (m0-1, p0-1)
    # U: rows 0-1 (m0-1, n0-2).
    # V: rows 0-3 (m0-1, p0-1).
    # W: cols 0-1 (p0-1, n0-2).
    
    U[0:3, col_offset:col_offset+11] = u_322
    V[0:4, col_offset:col_offset+11] = v_322
    W[0:6, col_offset:col_offset+11] = w_322 # W is 6x11. Indices 0..5.
    # W indices: k*3+i.
    # k=0,1. i=0,1,2.
    # So rows 0,1,2,3,4,5.
    col_offset += 11
    
    # Block 2: A1a (m0-1) * B11b (m0-1, p2-3)
    U[0:3, col_offset:col_offset+11] = u_322
    V[0:4, col_offset:col_offset+11] = v_322
    # W: p2-3, n0-2.
    # k=2,3. i=0,1,2.
    # Indices: 2*3+0=6, 2*3+1=7, 2*3+2=8, 3*3+0=9, 3*3+1=10, 3*3+2=11.
    W[6:12, col_offset:col_offset+11] = w_322
    col_offset += 11
    
    # Block 3: A1b (m2-3) * B11c (m2-3, p0-1)
    # U: rows 3-5 (m2-3, n0-2).
    U[3:6, col_offset:col_offset+11] = u_322
    V[4:8, col_offset:col_offset+11] = v_322
    W[0:6, col_offset:col_offset+11] = w_322
    col_offset += 11
    
    # Block 4: A1b (m2-3) * B11d (m2-3, p2-3)
    U[3:6, col_offset:col_offset+11] = u_322
    V[8:12, col_offset:col_offset+11] = v_322
    W[6:12, col_offset:col_offset+11] = w_322
    col_offset += 11
    
    # Term 2: A1 (3x4) * B12 (4x1)
    # Rank 12
    # U: 3x4. V: 4x1. W: 1x3.
    # Schoolbook.
    for i in range(3):
        for j in range(4):
            U[i*4+j, col_offset] = 1 # A index: i*m + j. m=4.
            V[j*1+0, col_offset] = 1 # B index: j*p + k. p=1. k=0.
            W[0*3+i, col_offset] = 1 # C index: k*n + i. k=0, n=3.
            col_offset += 1
    
    # Term 3: A2 (3x1) * B21 (1x4)
    # Rank 12
    # U: 3x1. V: 1x4. W: 4x3.
    for i in range(3):
        for k in range(4):
            U[i*4+4, col_offset] = 1 # A index: i*m + j. m=4, j=4.
            V[4*1+k, col_offset] = 1 # B index: j*p + k. j=4, p=1.
            W[k*3+i, col_offset] = 1 # C index: k*n + i.
            col_offset += 1
            
    # Term 4: A2 (3x1) * B22 (1x1)
    # Rank 3
    for i in range(3):
        U[i*4+4, col_offset] = 1
        V[4*1+4, col_offset] = 1
        W[4*3+i, col_offset] = 1
        col_offset += 1
        
    return {"U": U, "V": V, "W": W}
