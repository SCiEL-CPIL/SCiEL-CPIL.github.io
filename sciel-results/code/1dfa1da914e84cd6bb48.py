import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4 * 4x4 using Strassen (rank 49)
    # Part 2: 4x4 * 4x3 using Schoolbook (rank 48)
    
    # Helper to create Strassen decomposition for 2x2
    # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
    def strassen_2x2():
        # 7 multiplications
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1 # C00, C01, C10, C11 all get M1? 
        # C00 = M1 + ...
        # C01 = ...
        # C10 = ...
        # C11 = M1 - ...
        # So W[0][0]=1, W[1][0]=0, W[2][0]=0, W[3][0]=1? No.
        # Let's trace carefully.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 contributes to C00 (+1) and C11 (+1)
        W[0][0] = 1
        W[3][0] = 1
        
        # M2 contributes to C10 (+1) and C11 (-1)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        W[3][1] = -1
        
        # M3 contributes to C01 (+1) and C11 (+1)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1
        W[3][2] = 1
        
        # M4 contributes to C00 (+1) and C10 (+1)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1
        W[2][3] = 1
        
        # M5 contributes to C01 (+1) and C00 (-1)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = 1
        W[0][4] = -1
        
        # M6 contributes to C11 (+1)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[3][5] = 1
        
        # M7 contributes to C00 (+1)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # Build 4x4x4 decomposition from 2x2x2
    U_44, V_44, W_44 = strassen_2x2()
    
    U_4x4 = [[0]*49 for _ in range(16)]
    V_4x4 = [[0]*49 for _ in range(16)]
    W_4x4 = [[0]*49 for _ in range(16)]
    
    # A blocks:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # B blocks:
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # C blocks:
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 8,9,12,13
    # C11: 10,11,14,15
    
    for k in range(7):
        for i in range(4):
            val = U_44[i][k]
            if val != 0:
                # Map 2x2 index i to 4x4 index
                # i=0 -> A00 (0,1,4,5)
                # i=1 -> A01 (2,3,6,7)
                # i=2 -> A10 (8,9,12,13)
                # i=3 -> A11 (10,11,14,15)
                # This mapping is tricky because U_44[i] corresponds to a specific entry in the 2x2 block.
                # Actually, U_44 is for the whole 2x2 matrix A.
                # The indices in U_44 are 0,1,2,3 corresponding to A00, A01, A10, A11.
                # But in the 4x4 matrix, A00 is a 2x2 block.
                # So if U_44[0][k] = 1, it means A00 (the top-left entry of the top-left block) is involved.
                # Wait, the Strassen algorithm operates on the 2x2 blocks.
                # The "A00" in Strassen is a 2x2 matrix.
                # So when we say U_44[0][k] = 1, it refers to the entire 2x2 block A00?
                # No, the Strassen decomposition I wrote above is for 2x2 matrices where the entries are scalars.
                # So U_44[0] corresponds to the scalar A00.
                # But in the 4x4 case, A00 is a 2x2 block.
                # So we need to apply the Strassen decomposition to each 2x2 block of A and B.
                # This means we need 4 Strassen decompositions? No.
                # The Strassen algorithm replaces 8 scalar multiplications with 7.
                # If we apply it to 4x4 matrices, we view them as 2x2 blocks of 2x2 matrices.
                # The 7 "multiplications" are of 2x2 matrices.
                # Each 2x2 matrix multiplication can be done in rank 7 (using Strassen again) or 8 (schoolbook).
                # If we use Strassen again, we get 7*7=49.
                # So, for each of the 7 Strassen terms, we need to multiply two 2x2 matrices.
                # Let's say term k involves A_block_i and B_block_j.
                # We need to compute A_block_i * B_block_j.
                # This is a 2x2 * 2x2 = 2x2 multiplication.
                # We can use the strassen_2x2() decomposition for this.
                
                # So, for each k in 0..6:
                #   We have a specific pair of 2x2 blocks from A and B.
                #   We use the 7 rank-1 terms of strassen_2x2() to compute the product.
                #   This gives 7*7 = 49 terms total.
                
                pass # The loop above is incorrect. Let's restructure.

    # Correct approach for 4x4x4:
    # 1. Define the 7 Strassen combinations for 4x4 blocks.
    # 2. For each combination, we multiply two 2x2 matrices.
    # 3. Each 2x2 multiplication is decomposed into 7 rank-1 terms.
    
    # Let's define the 7 block operations.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # ...
    
    # We will create a list of (U_block_indices, V_block_indices, sign)
    # where U_block_indices are the 2x2 blocks of A involved, etc.
    # But since we are going down to scalar level, we can just use the strassen_2x2() result directly.
    
    # Let's create a function that takes U, V, W for 2x2 and expands them to 4x4.
    
    U_4x4 = [[0]*49 for _ in range(16)]
    V_4x4 = [[0]*49 for _ in range(16)]
    W_4x4 = [[0]*49 for _ in range(16)]
    
    # The 7 Strassen terms for 4x4:
    # Each term is a product of two 2x2 matrices.
    # Let's list the 2x2 block indices for A and B for each of the 7 terms.
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    # M1: (A00+A11) * (B00+B11)
    # This is not a single block multiplication. It's a sum of blocks multiplied.
    # (A00+A11) is a 2x2 matrix (sum of two 2x2 blocks).
    # So we need to compute (A00+A11) * (B00+B11).
    # This is a 2x2 * 2x2 multiplication.
    # We can use strassen_2x2() to decompose this multiplication.
    # The inputs to strassen_2x2() are the 2x2 matrices X = A00+A11 and Y = B00+B11.
    # The entries of X are sums of entries of A00 and A11.
    # So in the final U, V, W, we will have coefficients that are sums of entries from A00 and A11.
    
    # This is getting complicated to hardcode. Let's use a more systematic way.
    # We can generate the decomposition by composing the Strassen decomposition.
    
    # Let's use numpy to build the matrices.
    
    # strassen_2x2 returns U, V, W for 2x2 * 2x2 -> 2x2
    # U is 4x7, V is 4x7, W is 4x7
    
    # For 4x4, we have 4 blocks of A and 4 blocks of B.
    # The 7 Strassen terms involve linear combinations of these blocks.
    
    # Let's define the 7 terms as linear combinations of block multiplications.
    # Actually, it's easier to think of it as:
    # We want to compute C = A * B.
    # We use Strassen to express C in terms of 7 products of 2x2 matrices.
    # Let these products be P1, ..., P7.
    # Each Pk is a 2x2 matrix.
    # We then decompose each Pk = Xk * Yk using Strassen (rank 7).
    # So Pk = sum_{l=1}^7 u_kl * v_kl * w_kl.
    # Total rank = 7 * 7 = 49.
    
    # The matrices Xk and Yk are linear combinations of the 2x2 blocks of A and B.
    # For example, for M1 = (A00+A11)(B00+B11), X1 = A00+A11, Y1 = B00+B11.
    # The entries of X1 are:
    # X1[0,0] = A[0,0] + A[2,2]
    # X1[0,1] = A[0,1] + A[2,3]
    # X1[1,0] = A[1,0] + A[3,2]
    # X1[1,1] = A[1,1] + A[3,3]
    
    # So, when we use the strassen_2x2() decomposition for X1 * Y1,
    # the U vector for the first entry of X1 (which is X1[0,0]) will be combined with the U vector for A[0,0] and A[2,2].
    
    # Let's implement this composition.
    
    # First, get the base Strassen decomposition for 2x2.
    U_2, V_2, W_2 = strassen_2x2()
    
    # Now, for each of the 7 Strassen terms in the 4x4 case:
    # We need to define Xk and Yk as linear combinations of A and B blocks.
    
    # Let's define the linear combinations for X and Y for each of the 7 terms.
    # Term 1: X1 = A00 + A11, Y1 = B00 + B11
    # Term 2: X2 = A10 + A11, Y2 = B00
    # Term 3: X3 = A00, Y3 = B10 - B11
    # Term 4: X4 = A11, Y4 = B01 - B00
    # Term 5: X5 = A00 + A01, Y5 = B11
    # Term 6: X6 = A10 - A00, Y6 = B00 + B10
    # Term 7: X7 = A01 - A11, Y7 = B01 + B11
    
    # For each term k, we have Xk and Yk.
    # We use U_2, V_2, W_2 to decompose Xk * Yk.
    # U_2[i, l] is the coefficient of the i-th entry of X in the l-th rank-1 term.
    # So, if Xk[i] = sum_j c_ij A_j, then the U vector for the l-th term of the k-th product
    # will have c_ij in the position corresponding to A_j.
    
    # Let's map the 2x2 block indices to 4x4 matrix indices.
    # Block 00: indices 0,1,4,5
    # Block 01: indices 2,3,6,7
    # Block 10: indices 8,9,12,13
    # Block 11: indices 10,11,14,15
    
    # Let's create a function to expand a 2x2 block index and entry index to a 4x4 matrix index.
    def block_to_idx(block, entry):
        # block is 0,1,2,3 (00,01,10,11)
        # entry is 0,1,2,3 (00,01,10,11 within the block)
        # 4x4 index:
        # Block 00: 0,1,4,5
        # Block 01: 2,3,6,7
        # Block 10: 8,9,12,13
        # Block 11: 10,11,14,15
        
        # Let's use a lookup table.
        # Block 0: [0,1,4,5]
        # Block 1: [2,3,6,7]
        # Block 2: [8,9,12,13]
        # Block 3: [10,11,14,15]
        
        table = [
            [0,1,4,5],
            [2,3,6,7],
            [8,9,12,13],
            [10,11,14,15]
        ]
        return table[block][entry]
    
    # Now, let's build U_4x4, V_4x4, W_4x4.
    # There are 7*7 = 49 columns.
    # Column index c = k*7 + l, where k is the Strassen term (0..6) and l is the inner Strassen term (0..6).
    
    # Define the linear combinations for X and Y.
    # X_k is a 2x2 matrix. X_k[i] is a linear combination of A entries.
    # We represent X_k[i] as a dict {block: {entry: coeff}}
    
    # Term 1: X1 = A00 + A11
    X1 = {
        0: {0:1, 1:1, 2:1, 3:1}, # A00
        3: {0:1, 1:1, 2:1, 3:1}  # A11
    }
    Y1 = {
        0: {0:1, 1:1, 2:1, 3:1}, # B00
        3: {0:1, 1:1, 2:1, 3:1}  # B11
    }
    
    # Term 2: X2 = A10 + A11, Y2 = B00
    X2 = {
        2: {0:1, 1:1, 2:1, 3:1}, # A10
        3: {0:1, 1:1, 2:1, 3:1}  # A11
    }
    Y2 = {
        0: {0:1, 1:1, 2:1, 3:1}  # B00
    }
    
    # Term 3: X3 = A00, Y3 = B10 - B11
    X3 = {
        0: {0:1, 1:1, 2:1, 3:1}
    }
    Y3 = {
        2: {0:1, 1:1, 2:1, 3:1}, # B10
        3: {0:-1, 1:-1, 2:-1, 3:-1} # B11
    }
    
    # Term 4: X4 = A11, Y4 = B01 - B00
    X4 = {
        3: {0:1, 1:1, 2:1, 3:1}
    }
    Y4 = {
        1: {0:1, 1:1, 2:1, 3:1}, # B01
        0: {0:-1, 1:-1, 2:-1, 3:-1} # B00
    }
    
    # Term 5: X5 = A00 + A01, Y5 = B11
    X5 = {
        0: {0:1, 1:1, 2:1, 3:1}, # A00
        1: {0:1, 1:1, 2:1, 3:1}  # A01
    }
    Y5 = {
        3: {0:1, 1:1, 2:1, 3:1}  # B11
    }
    
    # Term 6: X6 = A10 - A00, Y6 = B00 + B10
    X6 = {
        2: {0:1, 1:1, 2:1, 3:1}, # A10
        0: {0:-1, 1:-1, 2:-1, 3:-1} # A00
    }
    Y6 = {
        0: {0:1, 1:1, 2:1, 3:1}, # B00
        2: {0:1, 1:1, 2:1, 3:1}  # B10
    }
    
    # Term 7: X7 = A01 - A11, Y7 = B01 + B11
    X7 = {
        1: {0:1, 1:1, 2:1, 3:1}, # A01
        3: {0:-1, 1:-1, 2:-1, 3:-1} # A11
    }
    Y7 = {
        1: {0:1, 1:1, 2:1, 3:1}, # B01
        3: {0:1, 1:1, 2:1, 3:1}  # B11
    }
    
    X_terms = [X1, X2, X3, X4, X5, X6, X7]
    Y_terms = [Y1, Y2, Y3, Y4, Y5, Y6, Y7]
    
    # Now, for each term k and inner term l:
    for k in range(7):
        for l in range(7):
            c = k * 7 + l
            
            # Get the coefficients for U and V from strassen_2x2
            # U_2[i, l] is the coefficient for the i-th entry of X in the l-th term.
            # So for each i in 0..3:
            for i in range(4):
                coeff_u = U_2[i][l]
                if coeff_u != 0:
                    # X_k[i] is a linear combination of A entries.
                    # X_k[i] = sum_{block, entry} X_terms[k][block][entry] * A[block, entry]
                    # So we add coeff_u * X_terms[k][block][entry] to U_4x4[idx, c]
                    for block in X_terms[k]:
                        for entry in X_terms[k][block]:
                            idx = block_to_idx(block, entry)
                            U_4x4[idx][c] += coeff_u * X_terms[k][block][entry]
                
                coeff_v = V_2[i][l]
                if coeff_v != 0:
                    for block in Y_terms[k]:
                        for entry in Y_terms[k][block]:
                            idx = block_to_idx(block, entry)
                            V_4x4[idx][c] += coeff_v * Y_terms[k][block][entry]
            
            # W is similar, but the output C is a 2x2 matrix.
            # The output of the k-th term is added to specific blocks of C.
            # Let's define the linear combination for C_k.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So, for term k, the output C_k is added to the C blocks with certain coefficients.
            # Let's define C_terms[k] as a dict {block: coeff}
            
            C_terms = [
                {0:1, 3:1},       # M1 -> C00, C11
                {2:1, 3:-1},      # M2 -> C10, C11
                {1:1, 3:1},       # M3 -> C01, C11
                {0:1, 2:1},       # M4 -> C00, C10
                {1:1, 0:-1},      # M5 -> C01, C00
                {3:1},            # M6 -> C11
                {0:1}             # M7 -> C00
            ]
            
            for i in range(4):
                coeff_w = W_2[i][l]
                if coeff_w != 0:
                    # C_k[i] is the i-th entry of the 2x2 output matrix for term k.
                    # This entry is added to the corresponding entry in the C blocks.
                    # The C blocks are 00, 01, 10, 11.
                    # The i-th entry of C_k corresponds to a specific entry in the C block.
                    # But C_k is a 2x2 matrix.
                    # The i-th entry of C_k is added to the i-th entry of each C block with the coefficient from C_terms[k].
                    # Wait, C_k is a 2x2 matrix. Its entries are C_k[0,0], C_k[0,1], C_k[1,0], C_k[1,1].
                    # These correspond to the entries of the C blocks.
                    # So, for each block in C_terms[k]:
                    for block in C_terms[k]:
                        coeff_block = C_terms[k][block]
                        idx = block_to_idx(block, i)
                        W_4x4[idx][c] += coeff_w * coeff_block
    
    # Part 2: 4x4 * 4x3 schoolbook
    U_43 = [[0]*48 for _ in range(16)]
    V_43 = [[0]*48 for _ in range(12)]
    W_43 = [[0]*48 for _ in range(12)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_43[i*4+j][col] = 1
                V_43[j*3+k][col] = 1
                # Output C is 4x3. Index is k*4+i.
                # But this C is the second part of the overall C (4x7).
                # The overall C has columns 0..6.
                # The first part (4x4) gives columns 0..3.
                # The second part (4x3) gives columns 4..6.
                # So the column index in the overall C is k+4.
                # The output index is (k+4)*4 + i.
                W_43[k*4+i][col] = 1
                col += 1
    
    # Combine
    rank = 49 + 48
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    
    for i in range(16):
        for c in range(49):
            U[i][c] = U_4x4[i][c]
        for c in range(48):
            U[i][49+c] = U_43[i][c]
    
    for i in range(16):
        for c in range(49):
            V[i][c] = V_4x4[i][c]
    for i in range(12):
        for c in range(48):
            V[16+i][49+c] = V_43[i][c]
    
    for i in range(16):
        for c in range(49):
            W[i][c] = W_4x4[i][c]
    for i in range(12):
        for c in range(48):
            W[16+i][49+c] = W_43[i][c]
    
    return {"U": U, "V": V, "W": W}

