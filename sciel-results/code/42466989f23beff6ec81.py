"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a low-rank tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Strategy: Build the 4x4x4 part using recursive Strassen (rank 49), then
    handle the 4x4x3 part. For the 4x4x3 part, we can use the fact that it's
    equivalent to computing 4x4 * 4x3. We can split this into:
    - Treat 4x3 as [4x2, 4x1]
    - 4x4 * 4x2: can be done with Strassen 4x4x4 restricted to 2 cols, or directly
    - Actually, let's use a different approach:
    
    The tensor for (n,m,p) is T[i,j,k]. We need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    
    Key insight: We can construct the decomposition by treating the 7 columns of B
    as 4 columns (handled by Strassen 4x4x4, rank 49) + 3 columns (handled separately).
    
    For the 4x4x3 part, we need a decomposition. The schoolbook rank is 4*4*3=48.
    But we can do better. Note that 4x4x3 is the same as computing A*B where B is 4x3.
    This is equivalent to 3 separate 4x4 * 4x1 multiplications, but they share the A factors.
    
    Actually, a better approach: Use the known result that the rank of 4x4x4 is 49 (Strassen).
    For 4x4x7, we can view it as:
    C = A * B where B has 7 columns.
    Split B into B1 (4x4) and B2 (4x3).
    C1 = A * B1 (4x4 result) -> rank 49 via Strassen
    C2 = A * B2 (4x3 result) -> need rank for this
    
    For 4x4x3: We can use the fact that this is a sub-tensor of 4x4x4.
    The 4x4x4 tensor has rank 49. The 4x4x3 tensor is obtained by restricting
    k to {0,1,2} (3 values instead of 4).
    
    One approach: Use the Strassen decomposition for 4x4x4 and simply drop the
    contributions where k=3. But this doesn't directly give a valid decomposition
    for 4x4x3 because the W matrix would have wrong dimensions.
    
    Alternative: For 4x4x3, we can use a block decomposition:
    A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    B = [[B11, B12, B13], [B21, B22, B23]] (2x3 blocks of 2x2)
    C = [[C11, C12, C13], [C21, C22, C23]] (2x3 blocks of 2x2)
    
    Each Cij = A1i * B1j + A2i * B2j (sum of two 2x2 * 2x2 products)
    Using Strassen for each 2x2*2x2: rank 7 per product.
    So each Cij needs rank 7+7=14? No, we can share the Strassen factors.
    
    Actually, for each pair (i,j), Cij = A1i*B1j + A2i*B2j.
    This is a 2x2 matrix product with 2 terms. The rank is at most 2*7=14,
    but might be less.
    
    Total for 4x4x3: 4*3 = 12 output blocks, each of size 2x2.
    If each needs rank 14, total is 12*14 = 168. That's worse than schoolbook 48.
    
    Better approach for 4x4x3: Just use schoolbook. Rank 48.
    Total rank: 49 + 48 = 97.
    
    But wait, we can do better. Let's think about this differently.
    
    Actually, the simplest approach that's guaranteed to work:
    1. Use Strassen for 4x4x4 (rank 49) for the first 4 columns
    2. Use schoolbook for 4x4x3 (rank 48) for the last 3 columns
    Total rank: 49 + 48 = 97.
    
    But we can potentially do better for the 4x4x3 part. Let's try:
    For 4x4x3, we can use the fact that it's a "thin" multiplication.
    One known result: rank(4x4x3) <= 4x4x4 rank = 49? No, that's not right.
    
    Let's just implement the 49+48=97 approach and see if we can improve.
    
    Actually, let me reconsider. The task says the output factor follows W[k*n+i,r].
    So W has shape (p*n, rank) = (7*4, rank) = (28, rank).
    U has shape (n*m, rank) = (16, rank).
    V has shape (m*p, rank) = (4*7, rank) = (28, rank).
    
    Let me implement the Strassen 4x4x4 decomposition properly.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140
    
    # Step 1: Build Strassen 2x2x2 decomposition
    def strassen_2x2_2x2():
        n2, m2, p2 = 2, 2, 2
        rank = 7
        U = [[0]*rank for _ in range(n2*m2)]
        V = [[0]*rank for _ in range(m2*p2)]
        W = [[0]*rank for _ in range(p2*n2)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4 = A11(B01-B00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    # Step 2: Build 4x4x4 decomposition using Strassen on 2x2 blocks
    # A is 4x4, split into 2x2 blocks: A11, A12, A21, A22
    # B is 4x4, split into 2x2 blocks: B11, B12, B21, B22
    # C is 4x4, split into 2x2 blocks: C11, C12, C21, C22
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For each Cij, we need to compute sum of two 2x2*2x2 products.
    # Using Strassen, each 2x2*2x2 has rank 7.
    # For the sum, we can use 14 rank terms (7 for each product).
    # Total for 4x4x4: 4 * 14 = 56? But Strassen says 49.
    # 
    # The 49 comes from applying Strassen at the block level:
    # Treat the 2x2 block matrix as a "super-matrix" and apply Strassen.
    # This gives 7 "super-multiplications" of 2x2 matrices.
    # Each super-multiplication is a 2x2*2x2 product, which has rank 7.
    # Total: 7 * 7 = 49.
    
    # Let's implement this properly.
    # Strassen on the block level:
    # S1 = (A11+A22)(B11+B22)
    # S2 = (A21+A22)B11
    # S3 = A11(B21-B22)
    # S4 = A22(B12-B11)
    # S5 = (A11+A12)B22
    # S6 = (A21-A11)(B11+B12)
    # S7 = (A12-A22)(B21+B22)
    # 
    # C11 = S1 + S4 - S5 + S7
    # C12 = S3 + S5
    # C21 = S2 + S4
    # C22 = S1 - S2 + S3 + S6
    
    # Each Sk is a 2x2*2x2 product, computed via Strassen (rank 7).
    # So total rank = 7 * 7 = 49.
    
    def build_4x4x4_strassen():
        """Build 4x4x4 decomposition with rank 49 using recursive Strassen."""
        n4, m4, p4 = 4, 4, 4
        rank = 49
        
        U = [[0]*rank for _ in range(n4*m4)]  # 16 rows
        V = [[0]*rank for _ in range(m4*p4)]  # 16 rows
        W = [[0]*rank for _ in range(p4*n4)]  # 16 rows
        
        # Block indices for 4x4 matrix split into 2x2 blocks:
        # A block (i,j) corresponds to rows i*2..i*2+1, cols j*2..j*2+1
        # A11: rows 0-1, cols 0-1 -> A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # A12: rows 0-1, cols 2-3 -> A indices: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
        # A21: rows 2-3, cols 0-1 -> A indices: (2,0)=8, (2,1)=9, (3,0)=12, (3,1)=13
        # A22: rows 2-3, cols 2-3 -> A indices: (2,2)=10, (2,3)=11, (3,2)=14, (3,3)=15
        
        # B block (i,j) corresponds to rows i*2..i*2+1, cols j*2..j*2+1
        # B11: rows 0-1, cols 0-1 -> B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B12: rows 0-1, cols 2-3 -> B indices: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
        # B21: rows 2-3, cols 0-1 -> B indices: (2,0)=8, (2,1)=9, (3,0)=12, (3,1)=13
        # B22: rows 2-3, cols 2-3 -> B indices: (2,2)=10, (2,3)=11, (3,2)=14, (3,3)=15
        
        # C block (i,j) corresponds to rows i*2..i*2+1, cols j*2..j*2+1
        # C11: rows 0-1, cols 0-1 -> C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # C12: rows 0-1, cols 2-3 -> C indices: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
        # C21: rows 2-3, cols 0-1 -> C indices: (2,0)=8, (2,1)=9, (3,0)=12, (3,1)=13
        # C22: rows 2-3, cols 2-3 -> C indices: (2,2)=10, (2,3)=11, (3,2)=14, (3,3)=15
        
        # U index: i*m4+j where i is row of A, j is col of A
        # V index: j*p4+k where j is row of B, k is col of B
        # W index: k*n4+i where k is row of C, i is col of C
        
        # For each Strassen term S_s (s=1..7), we have a 2x2*2x2 product.
        # The 2x2*2x2 product uses Strassen with 7 sub-terms.
        # So for each S_s, we have 7 rank terms.
        # Total: 7*7 = 49 rank terms.
        
        # Define the Strassen 2x2x2 decomposition
        U2, V2, W2 = strassen_2x2_2x2()
        
        # For each of the 7 Strassen terms at the block level,
        # we need to map the 2x2 block products to the 4x4 matrices.
        
        # Block level Strassen:
        # S1 = (A11+A22)(B11+B22)
        # S2 = (A21+A22)B11
        # S3 = A11(B21-B22)
        # S4 = A22(B12-B11)
        # S5 = (A11+A12)B22
        # S6 = (A21-A11)(B11+B12)
        # S7 = (A12-A22)(B21+B22)
        
        # For each S_s, the left factor is a sum/difference of 2x2 blocks of A,
        # and the right factor is a sum/difference of 2x2 blocks of B.
        # The product is a 2x2 matrix, which we compute via Strassen 2x2x2.
        
        # Let's define the block combinations:
        # For S_s, left = sum_{(i,j) in L_s} coeff * A_{i,j}
        #               right = sum_{(i,j) in R_s} coeff * B_{i,j}
        
        # Block indices: 11=(0,0), 12=(0,1), 21=(1,0), 22=(1,1)
        # A block (bi,bj) has global indices:
        #   row offset: bi*2, col offset: bj*2
        #   U indices for A_{bi,bj}[r,c] = U[(bi*2+r)*4 + (bj*2+c)]
        
        # B block (bi,bj) has global indices:
        #   row offset: bi*2, col offset: bj*2
        #   V indices for B_{bi,bj}[r,c] = V[(bi*2+r)*4 + (bj*2+c)]
        
        # C block (ci,cj) has global indices:
        #   row offset: ci*2, col offset: cj*2
        #   W indices for C_{ci,cj}[r,c] = W[(ci*2+r)*4 + (cj*2+c)]
        
        # Strassen terms at block level:
        # S1: L = {11:1, 22:1}, R = {11:1, 22:1}
        # S2: L = {21:1, 22:1}, R = {11:1}
        # S3: L = {11:1}, R = {21:1, 22:-1}
        # S4: L = {22:1}, R = {12:1, 11:-1}
        # S5: L = {11:1, 12:1}, R = {22:1}
        # S6: L = {21:1, 11:-1}, R = {11:1, 12:1}
        # S7: L = {12:1, 22:-1}, R = {21:1, 22:1}
        
        strassen_terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # S1
            ({(1,0):1, (1,1):1}, {(0,0):1}),             # S2
            ({(0,0):1}, {(1,0):1, (1,1):-1}),            # S3
            ({(1,1):1}, {(0,1):1, (0,0):-1}),            # S4
            ({(0,0):1, (0,1):1}, {(1,1):1}),             # S5
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),   # S6
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),   # S7
        ]
        
        # Output combinations:
        # C11 = S1 + S4 - S5 + S7
        # C12 = S3 + S5
        # C21 = S2 + S4
        # C22 = S1 - S2 + S3 + S6
        
        output_combos = {
            (0,0): [(0,1), (3,1), (4,-1), (6,1)],   # C11 = S1+S4-S5+S7
            (0,1): [(2,1), (4,1)],                   # C12 = S3+S5
            (1,0): [(1,1), (3,1)],                   # C21 = S2+S4
            (1,1): [(0,1), (1,-1), (2,1), (5,1)],   # C22 = S1-S2+S3+S6
        }
        
        # Now build the U, V, W matrices.
        # For each Strassen term s (0-6) and each sub-term t (0-6), we have a rank term.
        # Rank index = s*7 + t
        
        for s in range(7):
            L_blocks, R_blocks = strassen_terms[s]
            for t in range(7):
                rank_idx = s*7 + t
                u_coeff = U2[t]  # This is a list of 4 values for the 2x2 A block
                v_coeff = V2[t]  # This is a list of 4 values for the 2x2 B block
                w_coeff = W2[t]  # This is a list of 4 values for the 2x2 C block
                
                # For each A block in L_blocks, add to U
                for (bi, bj), coeff in L_blocks.items():
                    for r in range(2):
                        for c in range(2):
                            u_idx = (bi*2+r)*4 + (bj*2+c)
                            u_sub = u_coeff[r*2+c]  # Wait, u_coeff is indexed by 2x2 position
                            # U2 has 4 rows (2x2 flattened), so u_coeff[r*2+c] is the value
                            U[u_idx][rank_idx] += coeff * u_coeff[r*2+c]
                
                # For each B block in R_blocks, add to V
                for (bi, bj), coeff in R_blocks.items():
                    for r in range(2):
                        for c in range(2):
                            v_idx = (bi*2+r)*4 + (bj*2+c)
                            V[v_idx][rank_idx] += coeff * v_coeff[r*2+c]
                
                # For each C block in output_combos, add to W
                # The output of S_s goes to specific C blocks
                # We need to figure out which C blocks get contributions from S_s
                # Actually, the W matrix for the 2x2x2 Strassen gives the C block values.
                # The C block from S_s is a 2x2 matrix, and it contributes to multiple C blocks.
                
                # Wait, I need to reconsider. The Strassen term S_s is a 2x2 matrix product.
                # The result is a 2x2 matrix. This 2x2 matrix is added to specific C blocks.
                # But which C blocks? It depends on the Strassen combination.
                
                # Actually, in the block-level Strassen, each S_s is a 2x2 matrix that
                # contributes to the final C blocks according to the output_combos.
                # But the 2x2 result of S_s is a single 2x2 matrix, not spread across
                # multiple C blocks.
                
                # I think I'm overcomplicating this. Let me reconsider.
                
                # In the block-level Strassen:
                # S1 = (A11+A22)(B11+B22) is a 2x2 matrix.
                # This 2x2 matrix is used in C11 and C22.
                # C11 = S1 + S4 - S5 + S7 (each is a 2x2 matrix)
                # C22 = S1 - S2 + S3 + S6 (each is a 2x2 matrix)
                
                # So S1 contributes to both C11 and C22.
                # The 2x2 result of S1 is added to C11 and C22.
                
                # For the W matrix, we need to add the 2x2 result of S_s to the
                # appropriate C blocks.
                
                # Let me redefine: for each S_s, the 2x2 result is added to specific C blocks
                # with specific coefficients.
                
                # S1 -> C11 (coeff 1), C22 (coeff 1)
                # S2 -> C21 (coeff 1), C22 (coeff -1)
                # S3 -> C12 (coeff 1), C22 (coeff 1)
                # S4 -> C11 (coeff 1), C21 (coeff 1)
                # S5 -> C11 (coeff -1), C12 (coeff 1)
                # S6 -> C22 (coeff 1)
                # S7 -> C11 (coeff 1)
                
                # Let me define this properly.
                break  # Break out to redefine
                
        # Let me redefine the approach more carefully.
        
        # For each Strassen term s, the 2x2 result is a matrix M_s.
        # M_s is added to C blocks as follows:
        # M1 -> C11 (+1), C22 (+1)
        # M2 -> C21 (+1), C22 (-1)
        # M3 -> C12 (+1), C22 (+1)
        # M4 -> C11 (+1), C21 (+1)
        # M5 -> C11 (-1), C12 (+1)
        # M6 -> C22 (+1)
        # M7 -> C11 (+1)
        
        strassen_to_c = {
            0: {((0,0)): 1, ((1,1)): 1},    # S1 -> C11, C22
            1: {((1,0)): 1, ((1,1)): -1},   # S2 -> C21, C22
            2: {((0,1)): 1, ((1,1)): 1},    # S3 -> C12, C22
            3: {((0,0)): 1, ((1,0)): 1},    # S4 -> C11, C21
            4: {((0,0)): -1, ((0,1)): 1},   # S5 -> C11, C12
            5: {((1,1)): 1},                # S6 -> C22
            6: {((0,0)): 1},                # S7 -> C11
        }
        
        # Now rebuild
        U = [[0]*rank for _ in range(n4*m4)]
        V = [[0]*rank for _ in range(m4*p4)]
        W = [[0]*rank for _ in range(p4*n4)]
        
        for s in range(7):
            L_blocks, R_blocks = strassen_terms[s]
            for t in range(7):
                rank_idx = s*7 + t
                u_coeff = U2[t]
                v_coeff = V2[t]
                w_coeff = W2[t]
                
                # U: for each A block in L_blocks
                for (bi, bj), coeff in L_blocks.items():
                    for r in range(2):
                        for c in range(2):
                            u_idx = (bi*2+r)*4 + (bj*2+c)
                            U[u_idx][rank_idx] += coeff * u_coeff[r*2+c]
                
                # V: for each B block in R_blocks
                for (bi, bj), coeff in R_blocks.items():
                    for r in range(2):
                        for c in range(2):
                            v_idx = (bi*2+r)*4 + (bj*2+c)
                            V[v_idx][rank_idx] += coeff * v_coeff[r*2+c]
                
                # W: for each C block that S_s contributes to
                for (ci, cj), coeff in strassen_to_c[s].items():
                    for r in range(2):
                        for c in range(2):
                            w_idx = (ci*2+r)*4 + (cj*2+c)
                            W[w_idx][rank_idx] += coeff * w_coeff[r*2+c]
        
        return U, V, W
    
    # Build the 4x4x4 decomposition
    U4, V4, W4 = build_4x4x4_strassen()
    
    # Verify the 4x4x4 decomposition
    def validate_4x4x4(U, V, W):
        n4, m4, p4 = 4, 4, 4
        rank = len(U[0])
        for i in range(n4):
            for j in range(m4):
                for k in range(p4):
                    total = sum(U[i*m4+j][c] * V[j*p4+k][c] * W[k*n4+i][c] for c in range(rank))
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    valid4 = validate_4x4x4(U4, V4, W4)
    
    # Step 3: Build the 4x4x3 decomposition (schoolbook, rank 48)
    # For 4x4x3, U has shape (16, 48), V has shape (12, 48), W has shape (12, 48)
    # Wait, for 4x4x3: n=4, m=4, p=3
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (12, rank)
    # W: (p*n, rank) = (12, rank)
    
    # But we're combining with the 4x4x4 part, so the full problem is 4x4x7.
    # U: (16, total_rank)
    # V: (28, total_rank)
    # W: (28, total_rank)
    
    # For the 4x4x3 part (columns 4-6 of B, i.e., k=3,4,5 in 0-indexed... wait, p=7)
    # Actually, for 4x4x7, B is 4x7. The first 4 columns are handled by the 4x4x4 part.
    # The last 3 columns (cols 4,5,6) are handled by the 4x4x3 part.
    
    # For the 4x4x3 part:
    # U3: (16, 48) - same A factors
    # V3: (12, 48) - B factors for cols 4-6 (but V is indexed by j*p+k, so for p=7)
    # W3: (12, 48) - C factors for cols 4-6
    
    # Wait, I need to be careful with the indexing.
    # In the full 4x4x7 problem:
    # U[i*4+j, c] for i=0..3, j=0..3
    # V[j*7+k, c] for j=0..3, k=0..6
    # W[k*4+i, c] for k=0..6, i=0..3
    
    # For the 4x4x4 part (k=0..3):
    # U4[i*4+j, c] for i=0..3, j=0..3, c=0..48
    # V4[j*4+k, c] for j=0..3, k=0..3, c=0..48
    # W4[k*4+i, c] for k=0..3, i=0..3, c=0..48
    
    # But in the full problem, V is indexed by j*7+k, not j*4+k.
    # So V4[j*4+k, c] needs to be mapped to V[j*7+k, c] for k=0..3.
    
    # Similarly, W4[k*4+i, c] needs to be mapped to W[k*4+i, c] for k=0..3.
    # This is fine since k=0..3 in both cases.
    
    # For the 4x4x3 part (k=4..6):
    # We need to compute C[:, 4:7] = A * B[:, 4:7]
    # This is a 4x4 * 4x3 multiplication.
    # Schoolbook: rank 4*4*3 = 48.
    
    # U3[i*4+j, c] for i=0..3, j=0..3, c=0..47
    # V3[j*3+k', c] for j=0..3, k'=0..2, c=0..47 (local indices for the 4x3 part)
    # W3[k'*4+i, c] for k'=0..2, i=0..3, c=0..47 (local indices)
    
    # In the full problem:
    # V[j*7+(4+k'), c] = V3[j*3+k', c]
    # W[(4+k')*4+i, c] = W3[k'*4+i, c]
    
    # Build the 4x4x3 schoolbook decomposition
    rank3 = 4*4*3  # 48
    U3 = [[0]*rank3 for _ in range(16)]
    V3 = [[0]*rank3 for _ in range(12)]  # 4*3
    W3 = [[0]*rank3 for _ in range(12)]  # 3*4
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U3[i*4+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*4+i][col] = 1
                col += 1
    
    # Step 4: Combine the two parts
    total_rank = 49 + 48  # 97
    U_full = [[0]*total_rank for _ in range(16)]
    V_full = [[0]*total_rank for _ in range(28)]  # 4*7
    W_full = [[0]*total_rank for _ in range(28)]  # 7*4
    
    # Copy the 4x4x4 part (columns 0-48)
    for i in range(16):
        for c in range(49):
            U_full[i][c] = U4[i][c]
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V_full[j*7+k][c] = V4[j*4+k][c]
    
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W_full[k*4+i][c] = W4[k*4+i][c]
    
    # Copy the 4x4x3 part (columns 49-96)
    for i in range(16):
        for c in range(48):
            U_full[i][49+c] = U3[i][c]
    
    for j in range(4):
        for k in range(3):
            for c in range(48):
                V_full[j*7+(4+k)][49+c] = V3[j*3+k][c]
    
    for k in range(3):
        for i in range(4):
            for c in range(48):
                W_full[(4+k)*4+i][49+c] = W3[k*4+i][c]
    
    # Step 5: Validate the full decomposition
    def validate_full(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    valid_full = validate_full(U_full, V_full, W_full, n, m, p)
    
    if valid_full:
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: schoolbook for the whole thing
    rank_school = n * m * p  # 112
    U_school = [[0]*rank_school for _ in range(n*m)]
    V_school = [[0]*rank_school for _ in range(m*p)]
    W_school = [[0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
