import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # The strategy: Construct a decomposition using Strassen's algorithm
    # recursively applied to the block structure.
    #
    # Matrix A is (4, 5), B is (5, 5).
    # We can view A as four 2x2 blocks horizontally? No, 4 rows, 5 cols.
    # A = [A00 A01 A02] where A00 is 2x2, A01 is 2x2, A02 is 2x1?
    # This is messy. Let's use the 5x5x5 Strassen decomposition and project.
    #
    # The 5x5x5 matrix multiplication tensor has a known rank decomposition.
    # Standard Strassen is 7 for 2x2x2.
    # For 5x5x5, we can use a block decomposition.
    #
    # However, the prompt suggests: "constructing a rank-85 decomposition for the full 5x5x5... 
    # then explicitly verifying that the 4x5x5 multiplication tensor is a sub-tensor"
    #
    # Actually, a simpler structural insight:
    # 4x5x5 can be seen as:
    # A (4x5) * B (5x5) = C (4x5)
    #
    # Let's try to build the decomposition directly using Strassen on the 4x4x4 sub-problem
    # and handle the remaining 1 column/row with schoolbook or extended Strassen.
    #
    # Better: Use the known rank-7 decomposition for 2x2x2.
    # A (4x5) can be split into 2 blocks of size 2x5.
    # B (5x5) is one block.
    # This doesn't fit Strassen directly.
    
    # Let's stick to the "Sub-tensor of 5x5x5" idea but implement a concrete 5x5x5 Strassen-like scheme.
    # The 5x5x5 tensor can be decomposed using 7x7x7? No.
    
    # Let's use the most robust structural approach:
    # Decompose the 4x5x5 tensor into a sum of rank-1 terms.
    # We know that the 4x4x4 tensor has rank 49 (using Strassen recursively: 7*7=49).
    # The 4x5x5 tensor is slightly larger.
    
    # Let's try to construct a rank-77 or lower decomposition.
    # Schoolbook is 100.
    
    # Strategy:
    # 1. Define the 4x4x4 Strassen decomposition (rank 49).
    # 2. Handle the 5th column of A and 5th row/col of B.
    
    # Let's implement a generic Strassen decomposition for 2^k x 2^k.
    # 4x4x4 -> Rank 49.
    # 4x5x5 is not a power of 2.
    
    # Let's try a different structural decomposition:
    # A (4x5) * B (5x5)
    # Split A into A_left (4x2) and A_right (4x3)? No, Strassen needs square.
    
    # Let's use the "Padding to 8x8x8" approach?
    # Rank of 8x8x8 Strassen is 7^3 = 343. Too high.
    
    # Let's go back to the specific geometry.
    # C = A * B
    # A is 4x5. B is 5x5.
    
    # We can write A as [A1 A2] where A1 is 4x2 and A2 is 4x3.
    # This is not symmetric.
    
    # Let's try to find a rank-77 decomposition.
    # 4x5x5 = 4x4x4 + 4x1x5 + 4x4x1 + 4x1x1?
    # Rank(4x4x4) = 49.
    # Rank(4x1x5) = 20 (Schoolbook).
    # Rank(4x4x1) = 16 (Schoolbook).
    # Rank(4x1x1) = 4 (Schoolbook).
    # Total = 49 + 20 + 16 + 4 = 89.
    
    # Can we do better on the cross terms?
    # 4x1x5 is essentially 4 vectors of length 5. Rank is 20?
    # Actually, 4x1x5 tensor T_{i,j,k} = A_{i,j} B_{j,k}.
    # This is just outer product of A (4x5) and B (5x5) restricted?
    # No, it's just matrix multiplication where one dimension is 1.
    # The rank of m x 1 x p matrix multiplication is m*p.
    # So 4x1x5 has rank 20. 4x4x1 has rank 16. 4x1x1 has rank 4.
    
    # So 49 + 20 + 16 + 4 = 89.
    
    # Can we optimize the 4x4x4 part? Yes, 49 is optimal for 4x4x4.
    # Can we optimize the cross terms?
    # The cross terms involve the 5th column of A and 5th row/col of B.
    
    # Let's try to combine them.
    # A = [A4 | a] where A4 is 4x4 and a is 4x1.
    # B = [B4, b; b_row, b_scalar] where B4 is 4x4, b is 4x1, b_row is 1x4, b_scalar is 1x1.
    
    # C = A * B = [A4, a] * [B4, b; b_row, b_scalar]
    # C = A4*B4 + A4*b + a*b_row + a*b_scalar
    
    # Term 1: A4*B4 (4x4x4). Rank 49.
    # Term 2: A4*b (4x4x1). Rank 16.
    # Term 3: a*b_row (4x1x4). Rank 16.
    # Term 4: a*b_scalar (4x1x1). Rank 4.
    
    # Total Rank = 49 + 16 + 16 + 4 = 85.
    
    # This matches the "rank-85" hint in the directive!
    # We just need to implement this decomposition.
    
    # 1. Generate the 4x4x4 Strassen decomposition (Rank 49).
    # 2. Generate the 4x4x1 Schoolbook decomposition (Rank 16).
    # 3. Generate the 4x1x4 Schoolbook decomposition (Rank 16).
    # 4. Generate the 4x1x1 Schoolbook decomposition (Rank 4).
    
    # We need to map these to the global indices.
    
    # Global Indices:
    # A: (4, 5) -> u = i*5 + j, i in 0..3, j in 0..4
    # B: (5, 5) -> v = j*5 + k, j in 0..4, k in 0..4
    # C: (4, 5) -> w = k*4 + i, k in 0..4, i in 0..3 (Trace convention: W[k*n+i,r])
    
    # Let's define the local indices for the sub-tensors.
    
    # Sub-tensor 1: A4 (4x4) * B4 (4x4) -> C4 (4x4)
    # A4 indices: i in 0..3, j in 0..3. Global u = i*5 + j.
    # B4 indices: j in 0..3, k in 0..3. Global v = j*5 + k.
    # C4 indices: k in 0..3, i in 0..3. Global w = k*4 + i.
    
    # Sub-tensor 2: A4 (4x4) * b (4x1) -> C_col (4x1)
    # A4 indices: i in 0..3, j in 0..3. Global u = i*5 + j.
    # b indices: j in 0..3, k=4. Global v = j*5 + 4.
    # C_col indices: k=4, i in 0..3. Global w = 4*4 + i = 16 + i.
    
    # Sub-tensor 3: a (4x1) * B_row (1x4) -> C_row (4x4)
    # a indices: i in 0..3, j=4. Global u = i*5 + 4.
    # B_row indices: j=4, k in 0..3. Global v = 4*5 + k = 20 + k.
    # C_row indices: k in 0..3, i in 0..3. Global w = k*4 + i.
    
    # Sub-tensor 4: a (4x1) * b_scalar (1x1) -> C_scalar (4x1)
    # a indices: i in 0..3, j=4. Global u = i*5 + 4.
    # b_scalar indices: j=4, k=4. Global v = 4*5 + 4 = 24.
    # C_scalar indices: k=4, i in 0..3. Global w = 4*4 + i = 16 + i.
    
    # We need to construct the U, V, W matrices for each part and concatenate them.
    
    # Part 1: 4x4x4 Strassen
    # We need a rank-49 decomposition for 4x4x4.
    # This is 7x7x7 Strassen applied to 2x2x2 blocks?
    # 4x4x4 = 2x2 blocks of 2x2x2.
    # Each 2x2x2 block multiplication has rank 7.
    # There are 4x4=16 such interactions? No.
    # Strassen for 4x4x4 uses 7 multiplications of 2x2x2 matrices.
    # Wait, the rank of 4x4x4 is 49.
    # This comes from 7 * 7 = 49 rank-1 terms.
    # The 7 Strassen products are 2x2x2 matrices.
    # Each 2x2x2 product is decomposed into 7 rank-1 terms.
    # So total 7 * 7 = 49.
    
    # Let's hardcode the 4x4x4 Strassen decomposition or generate it.
    # Generating it is safer to avoid hardcoding errors.
    
    def strassen_2x2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        # A (2x2), B (2x2), C (2x2)
        # Indices: u=i*2+j, v=j*2+k, w=k*2+i
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A01-A00)(B00+B10)
        # M7 = (A10-A11)(B01+B11)
        
        # We need to express these as rank-1 terms in the tensor T_{u,v,w}
        # T_{u,v,w} = sum_l U_{u,l} V_{v,l} W_{w,l}
        
        # It is easier to define the decomposition in terms of the 7 intermediate products.
        # But we need the rank-1 terms of the 2x2x2 tensor.
        
        # The 2x2x2 tensor has rank 7.
        # We can find the 7 rank-1 vectors.
        # Or we can use the known decomposition.
        
        # Let's use a known rank-7 decomposition for 2x2x2.
        # Reference: Strassen's algorithm.
        
        # The 7 terms correspond to the 7 multiplications.
        # However, the "rank-1 terms" of the tensor are not exactly the Strassen products,
        # but the Strassen products are themselves sums of rank-1 terms?
        # No, the Strassen algorithm computes the product using 7 scalar multiplications.
        # The tensor T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}
        # Strassen shows T = sum_{l=1}^7 u_l \otimes v_l \otimes w_l
        
        # Let's extract the vectors u_l, v_l, w_l for 2x2x2.
        
        # u_l, v_l, w_l are vectors of length 4.
        
        # L=1: M1 = (A00+A11)(B00+B11)
        # Contribution to C:
        # C00 += M1 - M4 - M6 + M7
        # C01 += M3 + M5
        # C10 += M2 + M4
        # C11 += M1 - M3 + M5 - M7
        
        # This is for the 2x2 matrix result.
        # The tensor indices are u (A), v (B), w (C).
        
        # Let's define the 7 vectors.
        # u_l is the coefficient of A in the l-th product.
        # v_l is the coefficient of B in the l-th product.
        # w_l is the coefficient of C in the l-th product.
        
        # Actually, it's easier to just define the 7 rank-1 tensors directly.
        
        # Term 1: (A00+A11)(B00+B11)
        # u1 = [1, 0, 0, 1] (A00, A01, A10, A11)
        # v1 = [1, 0, 0, 1] (B00, B01, B10, B11)
        # w1 = [1, 0, 0, -1] (C00, C01, C10, C11) ? No, w is for the output tensor.
        # The output tensor T_{u,v,w} = 1 if u,v,w form a valid path.
        # The decomposition is T = sum U_l \otimes V_l \otimes W_l.
        
        # Let's use a precomputed set of 7 vectors for 2x2x2.
        # From literature:
        # u1 = [1, 0, 0, 1], v1 = [1, 0, 0, 1], w1 = [1, 0, 0, -1]
        # u2 = [0, 0, 1, 1], v2 = [1, 0, 0, 0], w2 = [0, 0, 1, 0]
        # u3 = [1, 0, 0, 0], v3 = [0, 1, 0, -1], w3 = [0, 1, 0, 0]
        # u4 = [0, 0, 0, 1], v4 = [0, 0, 1, -1], w4 = [0, 0, 1, 0]
        # u5 = [1, 0, 1, 0], v5 = [0, 0, 0, 1], w5 = [0, 1, 0, 0]
        # u6 = [0, 1, -1, 0], v6 = [1, 0, 1, 0], w6 = [0, 1, 0, 0]
        # u7 = [0, 0, 1, -1], v7 = [0, 1, 0, 1], w7 = [1, 0, 0, -1]
        
        # Let's verify one:
        # Sum u_l[i] v_l[j] w_l[k]
        # For i=0, j=0, k=0 (A00*B00 -> C00):
        # l=1: 1*1*1 = 1
        # l=2: 0*1*0 = 0
        # l=3: 1*0*0 = 0
        # l=4: 0*0*0 = 0
        # l=5: 1*0*0 = 0
        # l=6: 0*1*0 = 0
        # l=7: 0*0*1 = 0
        # Sum = 1. Correct.
        
        # For i=0, j=1, k=0 (A00*B01 -> C01):
        # l=1: 1*0*0 = 0
        # l=2: 0*0*0 = 0
        # l=3: 1*1*1 = 1
        # l=4: 0*0*0 = 0
        # l=5: 1*0*1 = 0
        # l=6: 0*1*1 = 0
        # l=7: 0*1*0 = 0
        # Sum = 1. Correct.
        
        # For i=1, j=0, k=0 (A10*B00 -> C10):
        # l=1: 0*1*0 = 0
        # l=2: 1*1*1 = 1
        # l=3: 0*0*0 = 0
        # l=4: 0*0*0 = 0
        # l=5: 1*0*0 = 0
        # l=6: -1*1*0 = 0
        # l=7: 1*0*1 = 0
        # Sum = 1. Correct.
        
        # This looks promising.
        
        u22 = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, -1, 0],
            [0, 0, 1, -1]
        ])
        v22 = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ])
        w22 = np.array([
            [1, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, -1]
        ])
        
        return u22, v22, w22

    u22, v22, w22 = strassen_2x2x2()
    
    # Now build 4x4x4 from 2x2x2.
    # 4x4x4 has 16 blocks of 2x2x2?
    # No, the 4x4x4 tensor is decomposed into 7 "Strassen products" which are 2x2x2 tensors.
    # Wait, the rank of 4x4x4 is 49.
    # This means we need 49 rank-1 terms.
    # The 7 Strassen steps each produce a 2x2x2 tensor.
    # Each 2x2x2 tensor is decomposed into 7 rank-1 terms.
    # So 7 * 7 = 49 terms.
    
    # We need to map the 2x2x2 indices to 4x4x4 indices.
    # The 7 Strassen products are formed from the 2x2 blocks of A, B.
    # Let A = [A00 A01; A10 A11] where each is 2x2.
    # Let B = [B00 B01; B10 B11] where each is 2x2.
    
    # The 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    
    # Each M_l is a 2x2 matrix.
    # M_l = X_l * Y_l where X_l, Y_l are linear combinations of the 2x2 blocks.
    # The tensor for M_l is T_l = X_l \otimes Y_l \otimes Z_l?
    # No, M_l is an intermediate result.
    # The final C is a linear combination of M_l.
    
    # The tensor T_4x4 = sum_l (U_l \otimes V_l \otimes W_l)
    # where U_l, V_l are 4x4 matrices (flattened to 16-vectors) and W_l is a 4x4 matrix (flattened to 16-vector).
    
    # Actually, it's easier to construct the 49 rank-1 terms by expanding the Strassen formula.
    
    # Let's define the 4x4x4 decomposition directly.
    
    # We will generate the 49 terms.
    
    # Term index t from 0 to 48.
    # t = 7 * l + s, where l is Strassen step (0-6), s is 2x2x2 rank-1 term (0-6).
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    # Define the block coefficients for Strassen steps.
    # A blocks: A00 (0-3), A01 (4-7), A10 (8-11), A11 (12-15)
    # B blocks: B00 (0-3), B01 (4-7), B10 (8-11), B11 (12-15)
    # C blocks: C00 (0-3), C01 (4-7), C10 (8-11), C11 (12-15)
    
    # For each Strassen step l, we have:
    # M_l = (sum_a c_a A_a) * (sum_b d_b B_b)
    # The tensor contribution is sum_a,b c_a d_b (A_a \otimes B_b \otimes C_contrib)
    # But C_contrib is not a single block, it's a combination.
    
    # Let's use the 2x2x2 decomposition vectors.
    # For a given Strassen step l, the "effective" A matrix is X_l = sum_a c_{l,a} A_a.
    # The "effective" B matrix is Y_l = sum_b d_{l,b} B_b.
    # The "effective" C matrix is Z_l = sum_c e_{l,c} C_c.
    
    # The tensor for this step is T_l = X_l \otimes Y_l \otimes Z_l?
    # No, X_l * Y_l = Z_l is a matrix equation.
    # The tensor T_l is the multiplication tensor for 2x2x2 applied to X_l, Y_l, Z_l.
    
    # We can express T_l = sum_{s=0}^6 (u_{l,s} \otimes v_{l,s} \otimes w_{l,s})
    # where u_{l,s}, v_{l,s}, w_{l,s} are vectors of length 4 (for the 2x2 blocks).
    
    # Then we map these block vectors to the 4x4 vectors.
    
    # Let's define the coefficients c_{l,a}, d_{l,b}, e_{l,c}.
    
    # l=0: M1 = (A00+A11)(B00+B11)
    # c: A00=1, A11=1. d: B00=1, B11=1.
    # e: C00=1, C01=0, C10=0, C11=-1? No, the C coefficients come from the final combination.
    # Wait, the Strassen algorithm computes M_l, then combines them to get C.
    # The tensor decomposition T = sum_l T_l is valid if T_l is the tensor for the product X_l Y_l.
    # But X_l Y_l is not C, it's an intermediate.
    
    # The correct decomposition is:
    # T = sum_{l=1}^7 (X_l \otimes Y_l \otimes Z_l) is WRONG.
    # The correct form is T = sum_{l=1}^7 (U_l \otimes V_l \otimes W_l)
    # where U_l, V_l are the input tensors for the l-th Strassen product,
    # and W_l is the output tensor?
    
    # Actually, the rank-7 decomposition of 2x2x2 is:
    # T_{2x2} = sum_{l=1}^7 u_l \otimes v_l \otimes w_l
    # For 4x4x4, we use the block structure.
    # T_{4x4} = sum_{l=1}^7 (U_l \otimes V_l \otimes W_l)
    # where U_l is a 4x4 matrix formed by the block coefficients of A in M_l,
    # V_l is a 4x4 matrix formed by the block coefficients of B in M_l,
    # and W_l is a 4x4 matrix formed by the block coefficients of C in M_l.
    
    # But M_l is a 2x2 matrix.
    # So U_l is a 4x4 matrix where each 2x2 block is a scalar multiple of the identity?
    # No, U_l is a 4x4 matrix that represents the linear combination of A blocks.
    
    # Let's construct U_l, V_l, W_l for each l.
    
    # For l=0 (M1):
    # X_0 = A00 + A11.
    # Y_0 = B00 + B11.
    # Z_0 = C00 - C11? No, the combination is C = ... + M1 ...
    # The coefficient of M1 in C is:
    # C00: +1
    # C01: 0
    # C10: 0
    # C11: -1 (from M1 - M3 + M5 - M7? No, C11 = M1 - M3 + M5 - M7)
    # So Z_0 has blocks: C00 coeff 1, C11 coeff -1.
    
    # So U_0 is a 4x4 matrix with A00 block = I, A11 block = I, others 0.
    # V_0 is a 4x4 matrix with B00 block = I, B11 block = I, others 0.
    # W_0 is a 4x4 matrix with C00 block = I, C11 block = -I, others 0.
    
    # Then the tensor term is U_0 \otimes V_0 \otimes W_0?
    # No, this would be a rank-1 term in the 4x4x4 tensor space?
    # Yes, if U_0, V_0, W_0 are vectors.
    # But they are matrices.
    
    # The rank-1 term is vec(U_0) \otimes vec(V_0) \otimes vec(W_0).
    # This is one rank-1 term.
    # But we need 7 such terms for each Strassen step?
    # No, the Strassen step l corresponds to ONE rank-7 decomposition of the 2x2x2 tensor.
    # So for each l, we add 7 rank-1 terms.
    
    # Total 7 * 7 = 49 terms.
    
    # Let's implement this.
    
    # Coefficients for A blocks (0:A00, 1:A01, 2:A10, 3:A11)
    # Coefficients for B blocks (0:B00, 1:B01, 2:B10, 3:B11)
    # Coefficients for C blocks (0:C00, 1:C01, 2:C10, 3:C11)
    
    # l=0: M1
    # A: [1, 0, 0, 1]
    # B: [1, 0, 0, 1]
    # C: [1, 0, 0, -1]
    
    # l=1: M2
    # A: [0, 0, 1, 1]
    # B: [1, 0, 0, 0]
    # C: [0, 0, 1, 0]
    
    # l=2: M3
    # A: [1, 0, 0, 0]
    # B: [0, 1, 0, -1]
    # C: [0, 1, 0, 0]
    
    # l=3: M4
    # A: [0, 0, 0, 1]
    # B: [0, 0, 1, -1]
    # C: [0, 0, 1, 0]
    
    # l=4: M5
    # A: [1, 0, 1, 0]
    # B: [0, 0, 0, 1]
    # C: [0, 1, 0, 0]
    
    # l=5: M6
    # A: [0, 1, -1, 0]
    # B: [1, 0, 1, 0]
    # C: [0, 1, 0, 0]
    
    # l=6: M7
    # A: [0, 0, 1, -1]
    # B: [0, 1, 0, 1]
    # C: [1, 0, 0, -1]
    
    coeff_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, -1, 0],
        [0, 0, 1, -1]
    ])
    coeff_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    coeff_C = np.array([
        [1, 0, 0, -1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, -1]
    ])
    
    # Now generate the 49 terms.
    
    for l in range(7):
        # For each of the 7 rank-1 terms in the 2x2x2 decomposition
        for s in range(7):
            t = l * 7 + s
            
            # The 2x2x2 vectors u22[:, s], v22[:, s], w22[:, s]
            # These are coefficients for the 2x2 blocks.
            
            # We need to construct U_44[:, t], V_44[:, t], W_44[:, t]
            
            # U_44 is 16x1.
            # It is formed by taking the 2x2 block structure.
            # The 2x2x2 vector u22[:, s] has 4 entries, corresponding to the 4 blocks of the 2x2 matrix X_l.
            # Wait, u22[:, s] is a vector of length 4, corresponding to the 4 elements of the 2x2 matrix.
            # But X_l is a 2x2 matrix formed from 2x2 blocks of A.
            # So the "element" (i,j) of X_l is a 2x2 matrix (block of A).
            
            # This is getting complicated.
            # Let's use a simpler mapping.
            
            # The 4x4x4 tensor T_{i,j,k} where i,j,k are indices in 0..3.
            # We can view i as (i_block, i_local) where i_block in 0..1, i_local in 0..1.
            
            # The Strassen decomposition works on the block level.
            # T = sum_{l,s} U_{l,s} \otimes V_{l,s} \otimes W_{l,s}
            
            # U_{l,s} is a 16-vector.
            # It is non-zero only at the blocks specified by coeff_A[l] and u22[s].
            
            # Let's construct U_{l,s} explicitly.
            
            U_vec = np.zeros(16)
            V_vec = np.zeros(16)
            W_vec = np.zeros(16)
            
            # For each block (ba, bb, bc) in 0..3
            # ba is block index in A (0:A00, 1:A01, 2:A10, 3:A11)
            # The local index within the block is (la, lb) in 0..1
            # Global index in A: u = (ba//2)*2 + la * 2 + lb? No.
            # A is 4x4.
            # Block ba: row block ba//2, col block ba%2.
            # Local row la, col lb.
            # Global row i = (ba//2)*2 + la
            # Global col j = (ba%2)*2 + lb
            # Global index u = i*4 + j
            
            # The coefficient for this block in X_l is coeff_A[l, ba] * u22[la*2+lb, s]?
            # No, u22 is the vector for the 2x2 matrix elements.
            # The 2x2 matrix X_l has elements (X_l)_{pq} = sum_{ba} coeff_A[l, ba] * (A_{ba})_{pq}
            # where (A_{ba})_{pq} is the element (p,q) of the block A_{ba}.
            
            # So the coefficient of A_{ba, p, q} in X_l is coeff_A[l, ba].
            # And the coefficient of (X_l)_{p,q} in the rank-1 term s is u22[p*2+q, s].
            
            # So the total coefficient for A_{ba, p, q} is coeff_A[l, ba] * u22[p*2+q, s].
            
            for ba in range(4):
                if coeff_A[l, ba] == 0:
                    continue
                for p in range(2):
                    for q in range(2):
                        coeff = coeff_A[l, ba] * u22[p*2+q, s]
                        if coeff == 0:
                            continue
                        # Global index in A
                        i_block = ba // 2
                        j_block = ba % 2
                        i = i_block * 2 + p
                        j = j_block * 2 + q
                        u_idx = i * 4 + j
                        U_vec[u_idx] += coeff
            
            for bb in range(4):
                if coeff_B[l, bb] == 0:
                    continue
                for p in range(2):
                    for q in range(2):
                        coeff = coeff_B[l, bb] * v22[p*2+q, s]
                        if coeff == 0:
                            continue
                        i_block = bb // 2
                        j_block = bb % 2
                        i = i_block * 2 + p
                        j = j_block * 2 + q
                        v_idx = i * 4 + j
                        V_vec[v_idx] += coeff
            
            for bc in range(4):
                if coeff_C[l, bc] == 0:
                    continue
                for p in range(2):
                    for q in range(2):
                        coeff = coeff_C[l, bc] * w22[p*2+q, s]
                        if coeff == 0:
                            continue
                        i_block = bc // 2
                        j_block = bc % 2
                        i = i_block * 2 + p
                        j = j_block * 2 + q
                        w_idx = i * 4 + j
                        W_vec[w_idx] += coeff
            
            U_44[:, t] = U_vec
            V_44[:, t] = V_vec
            W_44[:, t] = W_vec
    
    # Now we have the 4x4x4 decomposition (Rank 49).
    # We need to map this to the 4x5x5 problem.
    
    # Part 1: A4 (4x4) * B4 (4x4) -> C4 (4x4)
    # A4 is A[:, 0:4]. Global u = i*5 + j.
    # B4 is B[0:4, 0:4]. Global v = j*5 + k.
    # C4 is C[:, 0:4]. Global w = k*4 + i.
    
    U_part1 = np.zeros((20, 49))
    V_part1 = np.zeros((25, 49))
    W_part1 = np.zeros((20, 49))
    
    for t in range(49):
        for i in range(4):
            for j in range(4):
                u_local = i * 4 + j
                u_global = i * 5 + j
                U_part1[u_global, t] = U_44[u_local, t]
        
        for i in range(4):
            for j in range(4):
                v_local = i * 4 + j
                v_global = j * 5 + i # Wait, B is 5x5. B[j,k].
                # In 4x4x4, v index is j*4+k.
                # Here j is row of B, k is col of B.
                # v_local = j*4 + k.
                # v_global = j*5 + k.
                pass
        
        # Correct mapping for V:
        # v_local = j*4 + k
        # v_global = j*5 + k
        for j in range(4):
            for k in range(4):
                v_local = j * 4 + k
                v_global = j * 5 + k
                V_part1[v_global, t] = V_44[v_local, t]
        
        # Correct mapping for W:
        # w_local = k*4 + i
        # w_global = k*4 + i (since n=4, p=5, w = k*n + i)
        for k in range(4):
            for i in range(4):
                w_local = k * 4 + i
                w_global = k * 4 + i
                W_part1[w_global, t] = W_44[w_local, t]
    
    # Part 2: A4 (4x4) * b (4x1) -> C_col (4x1)
    # Rank 16.
    # Schoolbook: sum_{j=0}^3 A_{i,j} b_j -> C_{i,4}
    # u = i*5 + j
    # v = j*5 + 4
    # w = 4*4 + i = 16 + i
    
    U_part2 = np.zeros((20, 16))
    V_part2 = np.zeros((25, 16))
    W_part2 = np.zeros((20, 16))
    
    idx = 0
    for i in range(4):
        for j in range(4):
            u_global = i * 5 + j
            v_global = j * 5 + 4
            w_global = 16 + i
            U_part2[u_global, idx] = 1
            V_part2[v_global, idx] = 1
            W_part2[w_global, idx] = 1
            idx += 1
    
    # Part 3: a (4x1) * B_row (1x4) -> C_row (4x4)
    # Rank 16.
    # Schoolbook: sum_{k=0}^3 a_i B_{4,k} -> C_{i,k}
    # u = i*5 + 4
    # v = 4*5 + k = 20 + k
    # w = k*4 + i
    
    U_part3 = np.zeros((20, 16))
    V_part3 = np.zeros((25, 16))
    W_part3 = np.zeros((20, 16))
    
    idx = 0
    for i in range(4):
        for k in range(4):
            u_global = i * 5 + 4
            v_global = 20 + k
            w_global = k * 4 + i
            U_part3[u_global, idx] = 1
            V_part3[v_global, idx] = 1
            W_part3[w_global, idx] = 1
            idx += 1
    
    # Part 4: a (4x1) * b_scalar (1x1) -> C_scalar (4x1)
    # Rank 4.
    # u = i*5 + 4
    # v = 24
    # w = 16 + i
    
    U_part4 = np.zeros((20, 4))
    V_part4 = np.zeros((25, 4))
    W_part4 = np.zeros((20, 4))
    
    idx = 0
    for i in range(4):
        u_global = i * 5 + 4
        v_global = 24
        w_global = 16 + i
        U_part4[u_global, idx] = 1
        V_part4[v_global, idx] = 1
        W_part4[w_global, idx] = 1
        idx += 1
    
    # Concatenate
    U = np.hstack([U_part1, U_part2, U_part3, U_part4])
    V = np.hstack([V_part1, V_part2, V_part3, V_part4])
    W = np.hstack([W_part1, W_part2, W_part3, W_part4])
    
    # Verify
    # Check if U, V, W are integers
    if not np.allclose(U, np.round(U)):
        U = np.round(U).astype(int)
    if not np.allclose(V, np.round(V)):
        V = np.round(V).astype(int)
    if not np.allclose(W, np.round(W)):
        W = np.round(W).astype(int)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

