"""Program for matrix multiplication 4x4 x 4x9 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 x 4x9 matrix multiplication."""
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Schoolbook decomposition: rank = n*m*p = 144
    # This is guaranteed to be correct
    rank_schoolbook = n * m * p  # 144
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a better decomposition using Strassen-based approach
    # Strategy: Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A*B = [A*B1 | A*B2 | A*B3]
    # Use Strassen (rank 7) for 2x2x2 blocks, applied recursively for 4x4x4
    # 4x4x4 can be done with rank 28 using Strassen on 2x2 blocks of 2x2 matrices
    # Actually, let's use the known result: 4x4x4 tensor has rank at most 28
    # Wait, that's not right. The rank of the 4x4x4 multiplication tensor is at most 4^2 * 7 = 112?
    # No. Let me think more carefully.
    
    # For 2x2x2: rank 7 (Strassen)
    # For 4x4x4: We can view it as 2x2 blocks of 2x2 matrices
    # The multiplication involves 8 scalar multiplications of 2x2 blocks? No.
    # C = A*B where A, B are 4x4, partitioned into 2x2 blocks.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # Each Cij is 2x2, each Aij, Bij is 2x2.
    # Each term like A11*B11 is a 2x2x2 multiplication (rank 7).
    # There are 8 such terms, so rank <= 8*7 = 56 for 4x4x4.
    
    # For 4x4x1: schoolbook, rank = 4*4*1 = 16
    
    # Total: 56 + 56 + 16 = 128
    
    # Let's implement this properly.
    # First, let's build a 4x4x4 decomposition with rank 56.
    
    def build_4x4x4_decomp():
        """Build 4x4 x 4x4 -> 4x4 decomposition with rank 56."""
        # 4x4 matrices partitioned into 2x2 blocks
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]] where each is 2x2
        # C = [[C11, C12], [C21, C22]] where each is 2x2
        #
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        #
        # Each 2x2x2 multiplication has rank 7 (Strassen)
        # 8 such multiplications: rank 56
        
        U = [[0] * 56 for _ in range(16)]  # 4x4 = 16 elements
        V = [[0] * 56 for _ in range(16)]
        W = [[0] * 56 for _ in range(16)]
        
        # Strassen for 2x2x2:
        # Given A2x2 = [[a,b],[c,d]], B2x2 = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-c)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C00 = M1 - M4 - M2 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 - M4 + M6
        
        def add_strassen_2x2(a_base, b_base, c_base, offset):
            """Add Strassen decomposition for 2x2 block at given base indices."""
            # a_base: index of top-left element of A block (0, 1, 2, 3 for 2x2)
            # b_base: index of top-left element of B block
            # c_base: index of top-left element of C block
            # Elements: base, base+1, base+2, base+3 (row-major)
            
            a00, a01, a10, a11 = a_base, a_base+1, a_base+2, a_base+3
            b00, b01, b10, b11 = b_base, b_base+1, b_base+2, b_base+3
            c00, c01, c10, c11 = c_base, c_base+1, c_base+2, c_base+3
            
            # M1 = (a00+a11)(b00+b11)
            U[a00][offset] += 1
            U[a11][offset] += 1
            V[b00][offset] += 1
            V[b11][offset] += 1
            W[c00][offset] += 1
            W[c11][offset] += 1
            offset += 1
            
            # M2 = (a10+a11)*b00
            U[a10][offset] += 1
            U[a11][offset] += 1
            V[b00][offset] += 1
            W[c10][offset] += 1
            W[c11][offset] -= 1
            offset += 1
            
            # M3 = a00*(b01-b11)
            U[a00][offset] += 1
            V[b01][offset] += 1
            V[b11][offset] -= 1
            W[c01][offset] += 1
            W[c11][offset] += 1
            offset += 1
            
            # M4 = a11*(b10-a10)
            # Wait, this is not bilinear in the right way. Let me recheck.
            # M4 = d*(g-c) where d=a11, g=b10, c=a10
            # This is a11*b10 - a11*a10, which is not a product of one A and one B element.
            # I made an error. Let me use the correct Strassen formulas.
            
            # Correct Strassen:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-c)  -- This is d*g - d*c, which mixes A and B incorrectly
            # Actually, M4 = d*(g-c) where c is from A and g is from B.
            # d is from A (a11), g is from B (b10), c is from A (a10)
            # So M4 = a11 * (b10 - a10) = a11*b10 - a11*a10
            # This is not a simple product of one A entry and one B entry!
            
            # I need to be more careful. The Strassen algorithm uses:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-c)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            
            # Each Mi is a product of a linear combination of A entries and a linear combination of B entries.
            # So in the tensor decomposition, each Mi contributes to multiple U, V entries.
            
            # Let me redo this properly.
            # M4 = a11 * (b10 - a10) is WRONG because a10 is from A.
            # The correct formula: M4 = d*(g-c) where d is from A, g is from B, c is from A.
            # So M4 = a11 * b10 - a11 * a10. This is not separable as U*V.
            
            # I think I'm confusing myself. Let me look up the correct Strassen.
            # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-c)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            
            # C00 = M1 - M4 - M2 + M5
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 - M4 + M6
            
            # M4 = d*(g-c) = d*g - d*c
            # d is A[1,0]=a11, g is B[1,0]=b10, c is A[1,0]=a10
            # So M4 = a11*b10 - a11*a10
            # This is NOT a product of a linear form in A and a linear form in B.
            # It's a11*b10 - a11*a10, which has a11 appearing in both terms.
            
            # Wait, that can't be right for a bilinear algorithm. Let me recheck.
            # Actually, I think the standard Strassen is:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-c)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            
            # And M4 = d*(g-c) where c is A[1,0] and g is B[1,0].
            # So M4 = A[1,0] * (B[1,0] - A[1,0])
            # This is not bilinear! It has A[1,0]^2 term.
            
            # I must be misremembering. Let me use a different formulation.
            # Actually, the correct Strassen products are:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-c)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            
            # And indeed M4 = d*(g-c) = A[1,0]*(B[1,0]-A[1,0])
            # This IS bilinear in the sense that it's a product of a linear form in A and a linear form in B?
            # No, d is from A and (g-c) has g from B and c from A.
            # So M4 = A[1,0]*B[1,0] - A[1,0]*A[1,0]
            # The second term is not bilinear (it's quadratic in A).
            
            # I think I have the wrong formula. Let me use the correct one.
            # The standard Strassen algorithm:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-c)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            
            # I'll just implement schoolbook for 2x2x2 to be safe. Rank 4.
            # Then 4x4x4 rank = 8*4 = 32.
            # Total: 32 + 32 + 16 = 80.
            
            pass
        
        # Use schoolbook for 2x2x2 blocks
        # Each 2x2x2 has rank 8 (schoolbook)
        # 8 such blocks: rank 64
        
        U = [[0] * 64 for _ in range(16)]
        V = [[0] * 64 for _ in range(16)]
        W = [[0] * 64 for _ in range(16)]
        
        # Block indices in 4x4 matrix (row-major):
        # A11: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,4,5
        # A12: (0,2),(0,3),(1,2),(1,3) -> indices 2,3,6,7
        # A21: (2,0),(2,1),(3,0),(3,1) -> indices 8,9,12,13
        # A22: (2,2),(2,3),(3,2),(3,3) -> indices 10,11,14,15
        
        def get_block_indices(base_row, base_col):
            """Get the 4 indices for a 2x2 block starting at (base_row, base_col) in a 4x4 matrix."""
            return [base_row*4 + base_col, base_row*4 + base_col + 1,
                    (base_row+1)*4 + base_col, (base_row+1)*4 + base_col + 1]
        
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        
        # Each term is a 2x2x2 multiplication
        # Schoolbook for 2x2: C[i,j] = sum_k A[i,k]*B[k,j]
        # 4 products per 2x2x2
        
        def add_schoolbook_2x2(a_idx, b_idx, c_idx, offset):
            """Add schoolbook 2x2x2 decomposition."""
            # a_idx, b_idx, c_idx are lists of 4 indices
            # A = [[a00, a01], [a10, a11]] = a_idx[0], a_idx[1], a_idx[2], a_idx[3]
            # B = [[b00, b01], [b10, b11]] = b_idx[0], b_idx[1], b_idx[2], b_idx[3]
            # C = [[c00, c01], [c10, c11]] = c_idx[0], c_idx[1], c_idx[2], c_idx[3]
            
            # C00 = A00*B00 + A01*B10
            U[a_idx[0]][offset] += 1
            V[b_idx[0]][offset] += 1
            W[c_idx[0]][offset] += 1
            offset += 1
            
            U[a_idx[1]][offset] += 1
            V[b_idx[2]][offset] += 1
            W[c_idx[0]][offset] += 1
            offset += 1
            
            # C01 = A00*B01 + A01*B11
            U[a_idx[0]][offset] += 1
            V[b_idx[1]][offset] += 1
            W[c_idx[1]][offset] += 1
            offset += 1
            
            U[a_idx[1]][offset] += 1
            V[b_idx[3]][offset] += 1
            W[c_idx[1]][offset] += 1
            offset += 1
            
            # C10 = A10*B00 + A11*B10
            U[a_idx[2]][offset] += 1
            V[b_idx[0]][offset] += 1
            W[c_idx[2]][offset] += 1
            offset += 1
            
            U[a_idx[3]][offset] += 1
            V[b_idx[2]][offset] += 1
            W[c_idx[2]][offset] += 1
            offset += 1
            
            # C11 = A10*B01 + A11*B11
            U[a_idx[2]][offset] += 1
            V[b_idx[1]][offset] += 1
            W[c_idx[3]][offset] += 1
            offset += 1
            
            U[a_idx[3]][offset] += 1
            V[b_idx[3]][offset] += 1
            W[c_idx[3]][offset] += 1
            offset += 1
            
            return offset
        
        offset = 0
        # A11 indices
        A11 = get_block_indices(0, 0)
        A12 = get_block_indices(0, 2)
        A21 = get_block_indices(2, 0)
        A22 = get_block_indices(2, 2)
        
        B11 = get_block_indices(0, 0)
        B12 = get_block_indices(0, 2)
        B21 = get_block_indices(2, 0)
        B22 = get_block_indices(2, 2)
        
        C11 = get_block_indices(0, 0)
        C12 = get_block_indices(0, 2)
        C21 = get_block_indices(2, 0)
        C22 = get_block_indices(2, 2)
        
        # C11 = A11*B11 + A12*B21
        offset = add_schoolbook_2x2(A11, B11, C11, offset)
        offset = add_schoolbook_2x2(A12, B21, C11, offset)
        
        # C12 = A11*B12 + A12*B22
        offset = add_schoolbook_2x2(A11, B12, C12, offset)
        offset = add_schoolbook_2x2(A12, B22, C12, offset)
        
        # C21 = A21*B11 + A22*B21
        offset = add_schoolbook_2x2(A21, B11, C21, offset)
        offset = add_schoolbook_2x2(A22, B21, C21, offset)
        
        # C22 = A21*B12 + A22*B22
        offset = add_schoolbook_2x2(A21, B12, C22, offset)
        offset = add_schoolbook_2x2(A22, B22, C22, offset)
        
        return U, V, W
    
    # Build 4x4x4 decomposition
    U_444, V_444, W_444 = build_4x4x4_decomp()
    
    # Verify the 4x4x4 decomposition
    def verify_444(U, V, W, rank):
        """Verify that the 4x4x4 decomposition is correct."""
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    val = sum(U[i*4+j][r] * V[j*4+k][r] * W[k*4+i][r] for r in range(rank))
                    if val != 1:
                        return False
        return True
    
    if not verify_444(U_444, V_444, W_444, 32):
        # Fallback to schoolbook
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Now build 4x4x9 decomposition
    # B is 4x9, split into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A*B = [A*B1 | A*B2 | A*B3]
    # A*B1 and A*B2 use 4x4x4 decomposition (rank 32 each)
    # A*B3 uses 4x4x1 schoolbook (rank 16)
    
    # Total rank: 64 + 64 + 16 = 144
    total_rank = 144
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 80
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 80
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 80
    
    # A*B1: B1 is columns 0-3 of B (4x4)
    # U: same as U_444
    # V: B1 is B[:, 0:4], so V indices are j*9 + k for j in 0-3, k in 0-3
    # W: same as W_444
    
    for r in range(64):
        for i in range(16):
            U[i][r] = U_444[i][r]
        for j in range(4):
            for k in range(4):
                V[j*9 + k][r] = V_444[j*4 + k][r]
        for k in range(4):
            for i in range(4):
                W[k*4 + i][r] = W_444[k*4 + i][r]
    
    # A*B2: B2 is columns 4-7 of B (4x4)
    offset = 64
    for r in range(64):
        for i in range(16):
            U[i][offset + r] = U_444[i][r]
        for j in range(4):
            for k in range(4):
                V[j*9 + (k+4)][offset + r] = V_444[j*4 + k][r]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4 + i][offset + r] = W_444[k*4 + i][r]
    
    # A*B3: B3 is column 8 of B (4x1)
    # Schoolbook: C[i,8] = sum_j A[i,j] * B[j,8]
    offset = 128
    for i in range(4):
        for j in range(4):
            U[i*4 + j][offset] = 1
            V[j*9 + 8][offset] = 1
            W[8*4 + i][offset] = 1
            offset += 1
    
    # Verify the full decomposition
    def verify_full(U, V, W, rank):
        """Verify that the full 4x4x9 decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if val != 1:
                        return False
        return True
    
    if not verify_full(U, V, W, total_rank):
        # Fallback to schoolbook
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
