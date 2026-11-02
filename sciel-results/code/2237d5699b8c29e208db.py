import numpy as np

def construct():
    n, m, p = 4, 4, 7
    # Rank 49 for 4x4x4 + Rank 48 for 4x4x3
    rank1 = 49
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 using Rank 49 (Strassen x Strassen) ---
    # Indices: A is 4x4, B is 4x4, C is 4x4
    # Blocks: 00, 01, 10, 11 (each 2x2)
    # Global indices for A:
    # A[0,0] A[0,1] A[0,2] A[0,3]
    # A[1,0] A[1,1] A[1,2] A[1,3]
    # A[2,0] A[2,1] A[2,2] A[2,3]
    # A[3,0] A[3,1] A[3,2] A[3,3]
    # Block 00: (0,0),(0,1),(1,0),(1,1)
    # Block 01: (0,2),(0,3),(1,2),(1,3)
    # Block 10: (2,0),(2,1),(3,0),(3,1)
    # Block 11: (2,2),(2,3),(3,2),(3,3)
    
    # Helper to get global index for A
    def a_idx(r, c):
        return r * m + c
        
    def b_idx(r, c):
        return r * p + c # Here p=4 for this part
        
    def c_idx(r, c):
        return r * n + c # Here n=4 for this part
        
    # Define the 7 outer Strassen steps
    # Each step s defines U_s and V_s as 2x2 matrices of linear combos of blocks
    # U_s is 2x2, each element is a linear combo of A's 4 blocks (each block is 2x2 matrix)
    # So U_s[i][j] is a 4x4 matrix of coeffs for A? No, it's a linear map from A (16 elems) to a scalar.
    # Let's represent U_s as a 2x2 array of vectors of length 16 (coeffs for A).
    # Same for V_s (vectors of length 16 for B).
    
    # Block offsets in A:
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # Let's define the 7 patterns for 2x2 Strassen
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map "A11" etc to the actual blocks.
    # A11 -> Block 00
    # A12 -> Block 01
    # A21 -> Block 10
    # A22 -> Block 11
    
    # For each outer step s, we have coefficients for the 4 blocks of A and 4 blocks of B.
    # Let's store them as 4x4 matrices? No, the "scalars" of the outer Strassen are 2x2 matrices.
    # So U_s is a 2x2 matrix where each entry is a linear combo of the 4 blocks of A.
    # Let U_s[i][j] be a vector of 16 coeffs.
    
    # Initialize U_s, V_s as 2x2 arrays of 16-vectors
    U_s = [[ [0]*16 for _ in range(2) ] for _ in range(2) ]
    V_s = [[ [0]*16 for _ in range(2) ] for _ in range(2) ]
    
    # Define block coefficient vectors
    # Block 00: indices 0,1,4,5
    # Block 01: indices 2,3,6,7
    # Block 10: indices 8,9,12,13
    # Block 11: indices 10,11,14,15
    
    def block_vec(block_id):
        v = [0]*16
        if block_id == 0: # 00
            v[0]=1; v[1]=1; v[4]=1; v[5]=1
        elif block_id == 1: # 01
            v[2]=1; v[3]=1; v[6]=1; v[7]=1
        elif block_id == 2: # 10
            v[8]=1; v[9]=1; v[12]=1; v[13]=1
        elif block_id == 3: # 11
            v[10]=1; v[11]=1; v[14]=1; v[15]=1
        return v
    
    # Outer Strassen Steps
    # s=0: M1 = (A00+A11)(B00+B11) -> U = A00+A11, V = B00+B11
    # U is a 2x2 matrix? No, in Strassen for 2x2, the inputs are 2x2 matrices.
    # Here the "scalars" are 2x2 blocks.
    # So U_s is a 2x2 matrix of blocks?
    # No, Strassen computes C = AB where A, B are 2x2.
    # Here A is 4x4, viewed as 2x2 blocks of 2x2.
    # So A_block is a 2x2 matrix of 2x2 matrices.
    # Strassen computes 7 products of 2x2 matrices.
    # So U_s is a 2x2 matrix (composed of linear combos of A's blocks).
    # Wait, U_s in the formula M1 = (A11+A22)(B11+B22) is just a single 2x2 matrix (the sum of blocks).
    # But A11 is a 2x2 block. So A11+A22 is a 2x2 matrix.
    # So U_s is a 2x2 matrix.
    # Its elements are linear combos of the elements of A's blocks.
    
    # Let's define U_s as a 2x2 array of 16-vectors (coeffs for A).
    # U_s[0][0] corresponds to the top-left element of the resulting 2x2 matrix U_s.
    # This element is a linear combo of A's elements.
    
    # Step 0: M1 = (A00+A11)(B00+B11)
    # U_0 = A00 + A11.
    # U_0[0][0] = A00[0][0] + A11[0][0] -> A[0][0] + A[2][2]
    # U_0[0][1] = A00[0][1] + A11[0][1] -> A[0][1] + A[2][3]
    # U_0[1][0] = A00[1][0] + A11[1][0] -> A[1][0] + A[3][2]
    # U_0[1][1] = A00[1][1] + A11[1][1] -> A[1][1] + A[3][3]
    
    # We can generate these programmatically.
    
    # Let's define the 7 steps explicitly.
    # Each step is a list of (coeff_A, coeff_B) for the 4 blocks.
    # But we need to know which block element goes to which position in the 2x2 matrix U_s.
    # Actually, U_s is just a 2x2 matrix.
    # U_s[i][j] = sum_{k in {0..3}} c_k * A_block_k[i][j]
    # So for each i,j in 0..1, we have a vector of 16 coeffs.
    
    # Let's create a function to add a term to U, V, W
    col = 0
    
    # Inner Strassen (for 2x2x2)
    # The 7 patterns for 2x2 matrices X, Y:
    # P1 = (X00+X11)(Y00+Y11)
    # P2 = (X10+X11)Y00
    # P3 = X00(Y01-Y11)
    # P4 = X11(Y10-Y00)
    # P5 = (X00+X01)Y11
    # P6 = (X10-X00)(Y00+Y01)
    # P7 = (X01-X11)(Y10+Y11)
    
    # Result Z:
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # For each outer step s, we have U_s (2x2 matrix of A-combos) and V_s (2x2 matrix of B-combos).
    # We compute 7 inner products P_{s,t}.
    # Each P_{s,t} is a scalar (linear combo of A * linear combo of B).
    # We add this scalar to the rank decomposition.
    # The contribution to C is determined by how M_s (which is Z) maps to C blocks.
    
    # Mapping of M_s to C blocks:
    # s=0 (M1): C00 += Z00, C01 += Z01, C10 += Z10, C11 += Z11
    # s=1 (M2): C10 += Z00, C11 += Z01  (Wait, M2 contributes to C21 and C22? No.)
    # Let's check Strassen formula again.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So:
    # M1 (s=0) contributes to C00 (coeff 1), C01 (coeff 1? No, C12 is C01 in 0-indexed? 
    # C is 4x4. Blocks:
    # C00 (top-left), C01 (top-right)
    # C10 (bottom-left), C11 (bottom-right)
    # C11 (block) is C[2:4, 0:2].
    # C12 in 1-indexed block notation is C[0:2, 2:4].
    # So C12 is C01 in 0-indexed block notation.
    
    # M1 (s=0) -> C00 (1), C01 (0? No, M1 is in C11 and C22 formulas?
    # C11 = M1 + ...
    # C22 = M1 - ...
    # So M1 contributes to C00 and C11.
    # Wait, standard Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So:
    # M1 (s=0) -> C00 (1), C11 (1)
    # M2 (s=1) -> C10 (1), C11 (-1)
    # M3 (s=2) -> C01 (1), C11 (1)
    # M4 (s=3) -> C00 (1), C10 (1)
    # M5 (s=4) -> C00 (-1), C01 (1)
    # M6 (s=5) -> C11 (1)
    # M7 (s=6) -> C00 (1)
    
    # Note: C11 in formula is bottom-right block. C01 is top-right.
    
    # For each s, we have a 2x2 matrix M_s = Z.
    # Z00 contributes to C_block[s_block1][0][0]
    # Z01 contributes to C_block[s_block1][0][1]
    # Z10 contributes to C_block[s_block1][1][0]
    # Z11 contributes to C_block[s_block1][1][1]
    
    # We need to map s to the C blocks and coefficients.
    # s=0 (M1): C00 (1), C11 (1)
    # s=1 (M2): C10 (1), C11 (-1)
    # s=2 (M3): C01 (1), C11 (1)
    # s=3 (M4): C00 (1), C10 (1)
    # s=4 (M5): C00 (-1), C01 (1)
    # s=5 (M6): C11 (1)
    # s=6 (M7): C00 (1)
    
    # For each s, M_s is a 2x2 matrix.
    # The elements of M_s are Z00, Z01, Z10, Z11.
    # These are added to the corresponding positions in the C blocks.
    
    # So for s=0:
    # Z00 adds to C[0][0], C[0][1], C[1][0], C[1][1] of Block C00?
    # No, Z00 is the top-left element of the 2x2 matrix M_s.
    # M_s is a 2x2 matrix.
    # If M_s is added to C00, then Z00 goes to C[0][0], Z01 to C[0][1], Z10 to C[1][0], Z11 to C[1][1].
    # If M_s is added to C11, then Z00 goes to C[2][2], Z01 to C[2][3], Z10 to C[3][2], Z11 to C[3][3].
    
    # This logic holds.
    
    # Now, Z00, Z01, etc are linear combos of P_{s,t}.
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # So for each s, and each t (1..7), we have a scalar P_{s,t}.
    # This scalar contributes to Z00, Z01, Z10, Z11 with specific coefficients.
    # And each Z element contributes to specific C blocks with specific coefficients.
    
    # So total coeff for P_{s,t} on C[i][j] is:
    # sum over Z_elements (coeff of P_{s,t} in Z_elem) * (coeff of Z_elem in C_block) * (coeff of C_block in C[i][j])
    
    # We can precompute this.
    
    # Let's define the outer Strassen steps U_s, V_s.
    # We'll use a list of 7 tuples (U_coeffs, V_coeffs, C_map)
    # U_coeffs: 2x2 array of 16-vectors
    # V_coeffs: 2x2 array of 16-vectors
    # C_map: dict mapping (block_id, local_i, local_j) to coeff
    
    # This is getting too complex to write in the head. I will write the code carefully.
    
    # Simplification:
    # Just iterate s=0..6.
    # Construct U_s, V_s.
    # For t=0..6 (inner Strassen steps):
    #   Construct the linear combo for A (16-vector) and B (16-vector).
    #   Add to U, V.
    #   Construct the linear combo for C (16-vector) based on s and t.
    #   Add to W.
    #   col++.
    
    # Let's do it.
