import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 Strassen with rank 7.
    U: (4,7), V: (4,7), W: (4,7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 multiplications
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # U maps A elements to M inputs
    # A = [[a11, a12], [a21, a22]]
    # U row for a11: index 0*2+1 = 1? Wait, let me be careful.
    # U[i*m+j, r] where m=2. So U[0, r] for a11, U[1, r] for a12, U[2, r] for a21, U[3, r] for a22
    
    # M1: a11+a22 -> U[0][0]=1, U[3][0]=1; b11+b22 -> V[0][0]=1, V[3][0]=1
    # M2: a21+a22 -> U[2][1]=1, U[3][1]=1; b11 -> V[0][1]=1
    # M3: a11 -> U[0][2]=1; b21-b22 -> V[2][2]=1, V[3][2]=-1
    # M4: a22 -> U[3][3]=1; b12-b11 -> V[1][3]=1, V[0][3]=-1
    # M5: a11+a12 -> U[0][4]=1, U[1][4]=1; b22 -> V[3][4]=1
    # M6: a21-a11 -> U[2][5]=1, U[0][5]=-1; b11+b12 -> V[0][5]=1, V[1][5]=1
    # M7: a12-a22 -> U[1][6]=1, U[3][6]=-1; b21+b22 -> V[2][6]=1, V[3][6]=1
    
    U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1
    U[2][1] = 1; U[3][1] = 1; V[0][1] = 1
    U[0][2] = 1; V[2][2] = 1; V[3][2] = -1
    U[3][3] = 1; V[1][3] = 1; V[0][3] = -1
    U[0][4] = 1; U[1][4] = 1; V[3][4] = 1
    U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1
    U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1
    
    # W maps M outputs to C
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # W[k*2+i, r] where k is row of C, i is col of C
    # C11: k=0,i=0 -> W[0][r]; C12: k=0,i=1 -> W[1][r]; C21: k=1,i=0 -> W[2][r]; C22: k=1,i=1 -> W[3][r]
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1  # C11 = M1 + M4 - M5 + M7
    W[1][2] = 1; W[1][4] = 1                              # C12 = M3 + M5
    W[2][1] = 1; W[2][3] = 1                              # C21 = M2 + M4
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1  # C22 = M1 - M2 + M3 + M6
    
    return np.array(U), np.array(V), np.array(W)

def _build_4x4x4_tensors():
    """
    Build rank-49 tensors for 4x4x4 using 2x2x2 Strassen blocks.
    This uses the fact that 4x4x4 can be decomposed using 2x2x2 Strassen on 2x2 blocks.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # For 4x4x4, we have 4 blocks of 2x2x2 in the A and B matrices
    # But Strassen's algorithm for 4x4 is not simply 4 independent 2x2 multiplications
    # The rank-49 decomposition uses a more sophisticated combination
    
    # Actually, the standard way to get rank 49 for 4x4x4 is to use the 
    # recursive Strassen algorithm, which gives rank 7^log2(4) = 7^2 = 49
    
    # Let me construct this properly using Kronecker products
    # The 4x4x4 tensor can be seen as a 2x2 block matrix of 2x2 blocks
    
    # U: 16x49, V: 16x49, W: 16x449
    n = 4
    U = np.zeros((n*n, 49))
    V = np.zeros((n*n, 49))
    W = np.zeros((n*n, 49))
    
    # The rank-49 decomposition for 4x4x4 using Strassen recursion
    # This is complex, so let me use a known valid decomposition
    
    # Alternative: Use the fact that 4x4x4 rank is at most 49 (Strassen)
    # I'll construct it using the 2x2x2 factors
    
    # For simplicity and correctness, let me use a different approach:
    # Build 4x4x4 by combining 2x2x2 Strassen factors
    
    # The proper Strassen 4x4x4 decomposition:
    # We treat A, B, C as 2x2 block matrices with 2x2 blocks
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # C = A*B = [[C11, C12], [C21, C22]]
    
    # Using Strassen on the block level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mj is a 2x2 matrix product, so each requires rank 7 (Strassen 2x2x2)
    # Total rank = 7 * 7 = 49
    
    # Now I need to construct U, V, W of size 16x49
    
    # U: maps A elements to M inputs
    # A has 16 elements: A11 (4 elements), A12 (4), A21 (4), A22 (4)
    # A11 is at rows 0-3, A12 at rows 4-7, A21 at rows 8-11, A22 at rows 12-15
    
    # For each of the 7 block-level Strassen multiplications, we need a 2x2x2 Strassen decomposition
    # So we have 7 groups of 7 rank-1 terms = 49 total
    
    for block_mult in range(7):
        base = block_mult * 7
        for sub in range(7):
            col = base + sub
            
            # Determine which A blocks contribute to this block multiplication
            # and how they're combined
            
            # This is getting complex. Let me use a verified rank-49 decomposition.
            # Since I can't easily derive it, let me fall back to a valid but higher rank
            # or use a known implementation.
    
    # Given the complexity, let me use a simpler valid decomposition
    # For 4x4x8, I can use two independent 4x4x4 multiplications
    # If I have a rank-49 for 4x4x4, then rank-98 for 4x4x8
    
    # Let me construct rank-98 for 4x4x8 using schoolbook for the 4x4 parts
    # but with some optimization
    
    # Actually, let me just use the schoolbook for 4x4x8 which is rank 128
    # and see if I can improve
    
    pass

def construct():
    n, m, p = DIMENSIONS
    
    # Use Strassen-based decomposition for better rank
    # For 4x4x8, we can view it as two 4x4x4 multiplications
    # Rank = 2 * rank(4x4x4)
    
    # Known: rank(4x4x4) <= 49 (Strassen)
    # So rank(4x4x8) <= 98
    
    # Let me construct a valid rank-98 decomposition
    # by using the 2x2x2 Strassen factors
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # Build 4x4x4 tensors using Kronecker product structure
    # A 4x4 matrix can be seen as 2x2 blocks of 2x2 matrices
    # The tensor for 4x4x4 is a combination of 4x4=16 2x2x2 tensors
    
    # Actually, the cleanest way: use the fact that matmul(4,4,8) = matmul(4,4,4) + matmul(4,4,4)
    # where the second handles columns 4-7
    
    # For each 4x4x4, I need rank-49 factors
    # Let me construct rank-49 for 4x4x4 properly
    
    # The rank-49 decomposition for 4x4x4:
    # U: 16x49, V: 16x49, W: 16x49
    
    # I'll build it using the block Strassen approach
    n4 = 4
    U4 = np.zeros((n4*n4, 49))
    V4 = np.zeros((n4*n4, 49))
    W4 = np.zeros((n4*n4, 49))
    
    # Block indices for 4x4:
    # A11: rows 0-3, cols 0-1 -> elements at U indices 0,1,4,5 (i*4+j for i=0,1 j=0,1)
    # Wait, U[i*m+j, r] with m=4. So A[i,j] is at U[i*4+j, r]
    # A11 (top-left 2x2): i=0,1; j=0,1 -> U[0], U[1], U[4], U[5]
    # A12 (top-right 2x2): i=0,1; j=2,3 -> U[2], U[3], U[6], U[7]
    # A21 (bottom-left 2x2): i=2,3; j=0,1 -> U[8], U[9], U[12], U[13]
    # A22 (bottom-right 2x2): i=2,3; j=2,3 -> U[10], U[11], U[14], U[15]
    
    # Similarly for B and C
    
    # For each of the 7 Strassen block multiplications, apply 2x2x2 Strassen
    # M1 = (A11+A22)(B11+B22): A11+A22 is 2x2, B11+B22 is 2x2
    # The 2x2x2 Strassen for this uses U2, V2, W2
    
    # A11 elements: U[0], U[1], U[4], U[5] -> map to 2x2 indices 0,1,2,3
    # A22 elements: U[10], U[11], U[14], U[15] -> map to 2x2 indices 0,1,2,3
    # A11+A22: element (i,j) of the sum = A11[i,j] + A22[i,j]
    # In 2x2 indexing: A11[i,j] is at U2 index i*2+j, A22[i,j] is at U2 index i*2+j
    
    # So for M1, the A-side input to 2x2x2 Strassen is A11+A22
    # U2[i*2+j, sub] gives the coefficient for the (i,j) element of the 2x2 A-matrix
    # We need to map this to the 4x4 U matrix
    
    # Let me define the mapping
    # For a 2x2 block at position (bi, bj) in the 4x4 matrix:
    # bi=0: rows 0-1, bi=1: rows 2-3
    # bj=0: cols 0-1, bj=1: cols 2-3
    # Element (i,j) in the 2x2 block maps to 4x4 index (bi*2+i)*4 + (bj*2+j)
    
    def block_index(bi, bj, i, j):
        return (bi*2 + i) * 4 + (bj*2 + j)
    
    # A blocks:
    # A11: bi=0, bj=0
    # A12: bi=0, bj=1
    # A21: bi=1, bj=0
    # A22: bi=1, bj=1
    
    # B blocks:
    # B11: bi=0, bj=0
    # B12: bi=0, bj=1
    # B21: bi=1, bj=0
    # B22: bi=1, bj=1
    
    # C blocks:
    # C11: bi=0, bj=0
    # C12: bi=0, bj=1
    # C21: bi=1, bj=0
    # C22: bi=1, bj=1
    
    # Strassen block multipliers:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each Mj, the A-side is a linear combination of A blocks
    # and the B-side is a linear combination of B blocks
    
    # Let me define for each Mj:
    # a_comb: dict from (bi,bj) to coefficient
    # b_comb: dict from (bi,bj) to coefficient
    
    strassen_defs = [
        # M1
        {'a': {(0,0): 1, (1,1): 1}, 'b': {(0,0): 1, (1,1): 1}},
        # M2
        {'a': {(1,0): 1, (1,1): 1}, 'b': {(0,0): 1}},
        # M3
        {'a': {(0,0): 1}, 'b': {(1,0): 1, (1,1): -1}},
        # M4
        {'a': {(1,1): 1}, 'b': {(0,1): 1, (0,0): -1}},
        # M5
        {'a': {(0,0): 1, (0,1): 1}, 'b': {(1,1): 1}},
        # M6
        {'a': {(1,0): 1, (0,0): -1}, 'b': {(0,0): 1, (0,1): 1}},
        # M7
        {'a': {(0,1): 1, (1,1): -1}, 'b': {(1,0): 1, (1,1): 1}},
    ]
    
    for mj, definition in enumerate(strassen_defs):
        base = mj * 7
        for sub in range(7):
            col = base + sub
            
            # A-side: for each (i,j) in 2x2, coefficient from U2[i*2+j, sub]
            # The 2x2 A-matrix is sum of a_comb[(bi,bj)] * A_block(bi,bj)
            # So U4[index, col] += sum over (bi,bj) of a_comb[(bi,bj)] * U2[i*2+j, sub]
            
            for i in range(2):
                for j in range(2):
                    coeff_2x2 = U2[i*2 + j, sub]
                    if coeff_2x2 == 0:
                        continue
                    for (bi, bj), a_coeff in definition['a'].items():
                        idx = block_index(bi, bj, i, j)
                        U4[idx, col] += a_coeff * coeff_2x2
            
            # B-side: similar
            for i in range(2):
                for j in range(2):
                    coeff_2x2 = V2[i*2 + j, sub]
                    if coeff_2x2 == 0:
                        continue
                    for (bi, bj), b_coeff in definition['b'].items():
                        idx = block_index(bi, bj, i, j)
                        V4[idx, col] += b_coeff * coeff_2x2
            
            # C-side: W4[index, col] += sum over (bi,bj) of c_comb[(bi,bj)] * W2[i*2+j, sub]
            # The C block is at position (ci, cj) where ci, cj determine which C block
            # For Mj, the result contributes to specific C blocks
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So for each Mj, we need to know which C blocks it contributes to and with what coefficient
            c_contrib = {
                0: {(0,0): 1},      # M1 -> C11
                1: {(1,0): 1},      # M2 -> C21
                2: {(0,1): 1, (1,1): 1},  # M3 -> C12, C22
                3: {(0,0): 1, (1,0): 1},  # M4 -> C11, C21
                4: {(0,0): -1, (0,1): 1}, # M5 -> -C11, C12
                5: {(1,1): 1},      # M6 -> C22
                6: {(0,0): 1, (1,1): 1},  # M7 -> C11, C22
            }
            
            for i in range(2):
                for j in range(2):
                    coeff_2x2 = W2[i*2 + j, sub]
                    if coeff_2x2 == 0:
                        continue
                    for (ci, cj), c_coeff in c_contrib[mj].items():
                        idx = block_index(ci, cj, i, j)
                        W4[idx, col] += c_coeff * coeff_2x2
    
    # Now build 4x4x8 by using two copies of the 4x4x4 decomposition
    # First copy: columns 0-3 of B, columns 0-3 of C
    # Second copy: columns 4-7 of B, columns 4-7 of C
    
    U8 = np.zeros((n*m, 98))
    V8 = np.zeros((m*p, 98))
    W8 = np.zeros((p*n, 98))
    
    # Copy 1: B cols 0-3, C cols 0-3
    # U is the same for both copies (A is 4x4)
    U8[:, 0:49] = U4
    # V: B has 4x8, so B11 is rows 0-3, cols 0-3; B12 is rows 0-3, cols 4-7
    # For copy 1, we use B cols 0-3, which is the left half
    # V4 maps B elements to M inputs. B element (j,k) is at V4[j*4+k, r]
    # For copy 1, B[k] where k=0-3, so V8[j*8+k, 0:49] = V4[j*4+k, :]
    for j in range(4):
        for k in range(4):
            V8[j*8 + k, 0:49] = V4[j*4 + k, :]
    
    # W: C has 4x4 for copy 1, C element (k,i) is at W4[k*4+i, r]
    # For copy 1, C cols 0-3, so W8[k*4+i, 0:49] = W4[k*4+i, :]
    W8[:, 0:49] = W4
    
    # Copy 2: B cols 4-7, C cols 4-7
    U8[:, 49:98] = U4
    # V: B cols 4-7, so B element (j, k+4) for k=0-3
    # V8[j*8 + (k+4), 49:98] = V4[j*4+k, :]
    for j in range(4):
        for k in range(4):
            V8[j*8 + (k+4), 49:98] = V4[j*4 + k, :]
    
    # W: C cols 4-7, so C element (k, i+4) for i=0-3
    # W8[k*4 + (i+4), 49:98] = W4[k*4+i, :]
    for k in range(4):
        for i in range(4):
            W8[k*4 + (i+4), 49:98] = W4[k*4 + i, :]
    
    return {"U": U8.tolist(), "V": V8.tolist(), "W": W8.tolist()}
