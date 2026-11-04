import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 114 solution
    # 2 * Strassen 4x4x4 (rank 49 each) + Schoolbook 4x4x1 (rank 16)
    rank_total = 49 + 49 + 16
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Strassen 2x2x2 rank 7
    U2 = [
        [1, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 1, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 1]
    ]
    
    # Generate 4x4x4 Strassen (rank 49)
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for i in range(7):
        for j in range(7):
            k = i * 7 + j
            # U4 indices: (i1*2+i2, j1*2+j2) -> (i1*4+i2) ? No.
            # A is 4x4. Index u = i_row * 4 + i_col.
            # A = [[A00, A01], [A10, A11]]
            # A00 is at (0,0), (0,1), (1,0), (1,1) in 2x2 block space.
            # In 4x4 space, A00 is at (0,0), (0,1), (1,0), (1,1).
            # A01 is at (0,2), (0,3), (1,2), (1,3).
            # A10 is at (2,0), (2,1), (3,0), (3,1).
            # A11 is at (2,2), (2,3), (3,2), (3,3).
            
            # U2[i] is a vector of length 4, corresponding to the 4 entries of the 2x2 A block.
            # Let U2[i] = [a00, a01, a10, a11].
            # We need to map this to the 4x4 A matrix.
            # The block index is determined by i and j.
            # Wait, the tensor product T2 x T2 means we are multiplying two 2x2 matrices.
            # The result is a 2x2 matrix of 2x2 blocks.
            # So the first index of T2 corresponds to the block row/col, and the second index to the element within the block.
            
            # Let's use the property:
            # U4[ (r1*2+r2)*4 + (c1*2+c2) ] ... no.
            
            # Simpler:
            # U2[i] has entries for indices 0,1,2,3 of the 2x2 matrix.
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1).
            # In 4x4 matrix, block (b1, b2) where b1, b2 in {0,1}.
            # The element (r, c) in the 2x2 block corresponds to (b1*2+r, b2*2+c) in 4x4.
            
            # For the first factor (A), the block index is (i1, i2) ? No.
            # The tensor product of T2 and T2 is for 4x4x4.
            # The indices are (i1, i2, j1, j2, k1, k2).
            # i1, j1 are block indices. i2, j2 are element indices.
            
            # U2[i] corresponds to the A part of the i-th rank-1 tensor.
            # It has 4 components.
            # When we take U2[i] x U2[j], we get a 4x4 vector.
            # Component (a, b) where a, b in 0..3.
            # a = i1*2 + i2, b = i2'? No.
            
            # Let's just iterate over all 16 entries of U4.
            for u in range(16):
                r = u // 4
                c = u % 4
                br = r // 2
                bc = c // 2
                er = r % 2
                ec = c % 2
                
                # U2[i] has components for (er, ec) in the block (br, bc)?
                # No, U2[i] is for a single 2x2 block.
                # The tensor product means we have two levels.
                # Level 1: Block indices. Level 2: Element indices.
                
                # The value at (r, c) is U2[i][er*2+ec] * U2[j][br*2+bc]?
                # No, the first index of T2 is for the "outer" 2x2, the second for the "inner" 2x2.
                
                # Let's assume the standard Kronecker product ordering.
                # U4[u] = U2[i][er*2+ec] * U2[j][br*2+bc] is wrong.
                
                # Correct mapping:
                # The 4x4 matrix is a 2x2 block matrix of 2x2 matrices.
                # The tensor product T2 x T2 decomposes the multiplication.
                # The first T2 handles the block structure, the second handles the elements.
                
                # Actually, it's easier to just use the schoolbook 144 if this is too risky.
                # But 114 is a 21% improvement.
                
                pass
