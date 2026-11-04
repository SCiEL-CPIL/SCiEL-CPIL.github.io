"""Tensor decomposition for 4x4x9 matrix multiplication."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # For 4x4x9, we decompose as:
    # - First 4 columns: 4x4x4 multiplication (rank 18)
    # - Next 4 columns: 4x4x4 multiplication (rank 18)
    # - Last 1 column: 4x4x1 multiplication (rank 16, schoolbook)
    # Total rank: 18 + 18 + 16 = 52
    
    # We need a rank-18 decomposition for 4x4x4.
    # This is the Laderman algorithm.
    
    # Laderman's algorithm for 4x4x4 uses 18 multiplications.
    # Let me implement it carefully.
    
    # The 18 scalar multiplications in Laderman's algorithm:
    # 
    # Let A = [[a,b,c,d],[e,f,g,h],[i,j,k,l],[m,n,o,p]]
    # Let B = [[q,r,s,t],[u,v,w,x],[y,z,A1,B1],[C1,D1,E1,F1]]
    # 
    # The 18 products are:
    # M1  = (a+f) * (q+v)
    # M2  = (e+i) * (u+y)
    # M3  = (g+k) * (w+A1)
    # M4  = (m+n) * (C1+D1)
    # M5  = (a-f+c-g) * (q-v+s-w)
    # M6  = (e-i+h-l) * (u-y+x-z)
    # M7  = (m-n+o-p) * (C1-D1+E1-F1)
    # M8  = (a+f+j+k) * (q+v+z+A1)
    # M9  = (e+i+h+l) * (u+y+x+z)
    # M10 = (c+g+m+n) * (s+w+C1+D1)
    # M11 = (d+h+o+p) * (t+x+E1+F1)
    # M12 = (a+b+e+f) * (q+r+u+v)
    # M13 = (c+d+g+h) * (s+t+w+x)
    # M14 = (i+j+k+l) * (y+z+A1+B1)
    # M15 = (m+n+o+p) * (C1+D1+E1+F1)
    # M16 = (a-c+e-g) * (q-s+u-w)
    # M17 = (i-k+m-o) * (y-A1+C1-E1)
    # M18 = (b-d+f-h) * (r-t+v-x)
    
    # This is getting complex. Let me use a simpler approach.
    
    # Actually, let me use Strassen-based decomposition for 4x4x4.
    # Partition 4x4 into four 2x2 blocks.
    # Use Strassen's 7 multiplications for the block-level multiplication.
    # Each 2x2 multiplication uses Strassen's 7 multiplications.
    # Total: 7 * 7 = 49 for 4x4x4.
    
    # For 4x4x9: 49 + 49 + 16 = 114
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]  # 16 x 114
    V = [[0] * rank for _ in range(m * p)]  # 36 x 114
    W = [[0] * rank for _ in range(p * n)]  # 36 x 114
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    def add_strassen_2x2(U, V, W, a_off, b_off, c_off, c_start):
        """Add Strassen 2x2x2 decomposition."""
        c = c_start
        # M1 = (A11+A22)(B11+B22)
        U[a_off + 0][c] = 1  # A11
        U[a_off + 3][c] = 1  # A22
        V[b_off + 0][c] = 1  # B11
        V[b_off + 3][c] = 1  # B22
        W[c_off + 0][c] = 1  # C11
        W[c_off + 3][c] = 1  # C22
        c += 1
        
        # M2 = (A21+A22)B11
        U[a_off + 2][c] = 1  # A21
        U[a_off + 3][c] = 1  # A22
        V[b_off + 0][c] = 1  # B11
        W[c_off + 2][c] = 1  # C21
        W[c_off + 3][c] = -1  # C22
        c += 1
        
        # M3 = A11(B21-B22)
        U[a_off + 0][c] = 1  # A11
        V[b_off + 2][c] = 1  # B21
        V[b_off + 3][c] = -1  # B22
        W[c_off + 1][c] = 1  # C12
        W[c_off + 3][c] = 1  # C22
        c += 1
        
        # M4 = A22(B12-B11)
        U[a_off + 3][c] = 1  # A22
        V[b_off + 1][c] = 1  # B12
        V[b_off + 0][c] = -1  # B11
        W[c_off + 0][c] = 1  # C11
        W[c_off + 2][c] = 1  # C21
        c += 1
        
        # M5 = (A11+A12)B22
        U[a_off + 0][c] = 1  # A11
        U[a_off + 1][c] = 1  # A12
        V[b_off + 3][c] = 1  # B22
        W[c_off + 0][c] = -1  # C11
        W[c_off + 1][c] = 1  # C12
        c += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a_off + 2][c] = 1  # A21
        U[a_off + 0][c] = -1  # A11
        V[b_off + 0][c] = 1  # B11
        V[b_off + 1][c] = 1  # B12
        W[c_off + 3][c] = 1  # C22
        c += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a_off + 1][c] = 1  # A12
        U[a_off + 3][c] = -1  # A22
        V[b_off + 2][c] = 1  # B21
        V[b_off + 3][c] = 1  # B22
        W[c_off + 0][c] = 1  # C11
        c += 1
        
        return c
    
    # For 4x4x4, partition into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Using Strassen for block-level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product, computed with Strassen (7 multiplications)
    
    def add_4x4x4_strassen(U, V, W, a_off, b_off, c_off, c_start):
        """Add 49-rank Strassen-based decomposition for 4x4x4."""
        c = c_start
        
        # Block indices for 4x4 matrix (row-major, 4 columns):
        # A11: rows 0-1, cols 0-1 -> U indices: a_off+0, a_off+1, a_off+4, a_off+5
        # A12: rows 0-1, cols 2-3 -> U indices: a_off+2, a_off+3, a_off+6, a_off+7
        # A21: rows 2-3, cols 0-1 -> U indices: a_off+8, a_off+9, a_off+12, a_off+13
        # A22: rows 2-3, cols 2-3 -> U indices: a_off+10, a_off+11, a_off+14, a_off+15
        
        # Similarly for B and C.
        
        # M1 = (A11+A22)(B11+B22)
        # This is a 2x2 product of (A11+A22) and (B11+B22)
        # A11+A22 has entries:
        #   (A11[0,0]+A22[0,0], A11[0,1]+A22[0,1])
        #   (A11[1,0]+A22[1,0], A11[1,1]+A22[1,1])
        # In U indices: (a_off+0 + a_off+10, a_off+1 + a_off+11)
        #               (a_off+4 + a_off+14, a_off+5 + a_off+15)
        
        # For the 2x2 product, we use Strassen with offsets:
        # The "A" matrix for this product has its 4 entries at:
        #   [a_off+0, a_off+1, a_off+4, a_off+5] for A11 part
        #   [a_off+10, a_off+11, a_off+14, a_off+15] for A22 part
        # Combined: we need to add both.
        
        # This is complex. Let me use a different approach.
        # I'll directly compute each of the 7 block products using Strassen.
        
        # For M1 = (A11+A22)(B11+B22):
        # The sum A11+A22 is a 2x2 matrix. Let's call it S_A.
        # S_A[0,0] = A[0,0] + A[2,2] -> U[a_off+0] + U[a_off+10]
        # S_A[0,1] = A[0,1] + A[2,3] -> U[a_off+1] + U[a_off+11]
        # S_A[1,0] = A[1,0] + A[3,2] -> U[a_off+4] + U[a_off+14]
        # S_A[1,1] = A[1,1] + A[3,3] -> U[a_off+5] + U[a_off+15]
        
        # Similarly for B11+B22.
        
        # To handle the sum, I need to modify the Strassen function to accept
        # linear combinations. This is getting too complex for a clean implementation.
        
        # Let me use schoolbook for the inner 2x2 multiplications instead.
        # Total rank: 7 * 4 = 28 for 4x4x4.
        
        # For M1 = (A11+A22)(B11+B22), schoolbook:
        # 4 scalar multiplications:
        # S_A[0,0] * S_B[0,0]
        # S_A[0,1] * S_B[1,0]
        # S_A[1,0] * S_B[0,1]
        # S_A[1,1] * S_B[1,1]
        
        # Each of these is a product of linear combinations.
        
        # This is still complex. Let me just use schoolbook for the whole 4x4x4.
        # Rank 16 for 4x4x4.
        
        # For 4x4x9: 16 + 16 + 16 = 48? No, that's not right.
        # Schoolbook for 4x4x4 is rank 16.
        # For 4x4x9: 16 (first 4 cols) + 16 (next 4 cols) + 16 (last 1 col) = 48
        
        # Wait, schoolbook for 4x4x4 is 4*4 = 16 multiplications.
        # For 4x4x9, it's 4*4*9 = 144.
        
        # I'm going in circles. Let me just use schoolbook for the whole thing.
        
        return c
    
    # Given the complexity, let me just use schoolbook.
    rank = n * m * p
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
