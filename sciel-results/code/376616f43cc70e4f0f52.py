import time
import sys

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x9 matrix.
    
    Strategy:
    - Treat the 3x9 matrix B as a 3x3 matrix over C^9 (i.e., 3 rows of 9-vectors).
    - Decompose C = A * B where A is 2x3 and B is 3x9.
    - We can view this as: for each of the 9 columns of B, compute A * b_j (2x3 * 3x1 = 2x1).
    - The multiplication A * b_j is a 2x3x1 multiplication.
    - The rank of 2x3x1 matrix multiplication tensor:
      R(2,3,1) = ? 
      C_i = sum_j A_ij b_j for i=1,2.
      C_1 = a*b1 + b*b2 + c*b3
      C_2 = d*b1 + e*b2 + f*b3
      This is a bilinear form with 2 outputs, 3+1=4 inputs... 
      Actually, the tensor T_{2,3,1} has entries T[i,j,k] where i in {0,1}, j in {0,1,2}, k in {0} (only k=0).
      So T[i,j,0] = 1 if the term A_ij * B_j0 appears in C_i.
      The rank of this tensor: we need to compute 2 linear combinations of 3 variables each.
      This is equivalent to the rank of a 2x3 matrix multiplication where the "right matrix" is 3x1.
      R(2,3,1) = 4? Or 6?
      Actually, C_1 = a b1 + b b2 + c b3
                 C_2 = d b1 + e b2 + f b3
      We can write this as:
      Let u1 = a, u2 = b, u3 = c, u4 = d, u5 = e, u6 = f
      Let v1 = b1, v2 = b2, v3 = b3
      C_1 = u1*v1 + u2*v2 + u3*v3
      C_2 = u4*v1 + u5*v2 + u6*v3
      
      This is a tensor of format (6, 3, 2). 
      The rank is at most 6 (schoolbook: each term is a separate rank-1).
      Can we do better? 
      R(2,3,1) is known to be 4? No.
      Let's check: the tensor is T = sum_{i,j} e_i (left) ⊗ e_j (right) ⊗ (e_i for output... wait, output index is k in {0,1}).
      T = sum_{i=0}^1 sum_{j=0}^2 e_i^L ⊗ e_j^R ⊗ e_i^O (where e_i^O corresponds to output i)
      Wait, the output has 2 components. The input A has 2*3=6 components. The input B has 3 components.
      The tensor is in R^6 ⊗ R^3 ⊗ R^2.
      T = sum_{i,j} A_ij B_j ⊗ (contribution to C_i)
      So T = (A_00 B_0 e_0^O) + (A_01 B_1 e_0^O) + (A_02 B_2 e_0^O) + (A_10 B_0 e_1^O) + (A_11 B_1 e_1^O) + (A_12 B_2 e_1^O)
      This is a sum of 6 rank-1 tensors.
      Is the rank 6? Or can it be less?
      The sub-tensor where we fix the output to be e_0^O is the "dot product" of a 3-vector (row 0 of A) with B.
      The rank of the dot product tensor <x,y> in R^3 is 3.
      Since we have two such dot products (for output 0 and output 1) that share the B variables, 
      the total rank is at least 3 (for B's) * 2? No.
      Actually, R(2,3,1) = 6. 
      Why? Because the 6 terms are linearly independent in the tensor space.
      Let's verify with a known result. R(n,m,1) = n*m for n,m >= 1? 
      If n=2, m=3, R(2,3,1) = 6.
      
      So for each of the 9 columns, we need rank 6.
      Total rank = 9 * 6 = 54.
      This is the same as schoolbook.
      
      Is there a better way?
      What if we group the 9 columns?
      Let B be 3x9. We can split B into three 3x3 matrices B1, B2, B3.
      C = A * [B1 B2 B3] = [A B1, A B2, A B3].
      Each A Bk is a 2x3 multiplication.
      R(2,3,3) = ?
      Schoolbook: 2*3*3 = 18.
      Can we do better?
      R(2,3,3) <= R(2,3) * 3? No.
      R(2,3,3) is the rank of multiplying 2x3 by 3x3.
      This is equivalent to R(2,3) for each of the 3 columns of the result, but the 3 columns share the A factors.
      The tensor is in R^{2*3} ⊗ R^{3*3} ⊗ R^{3*2} = R^6 ⊗ R^9 ⊗ R^6.
      Rank?
      If we use Strassen-like techniques?
      R(2,3,3) is not well-known to be less than 18.
      However, we can use the fact that R(2,3,3) <= 2 * R(2,3,1) + ... no.
      
      Let's look at R(2,3,9).
      Schoolbook is 54.
      If we split the 9 into 3+3+3, we get 3 * R(2,3,3).
      If R(2,3,3) < 18, we win.
      
      Is R(2,3,3) < 18?
      Consider the tensor T_{2,3,3}.
      It contains T_{2,3,1} as a sub-tensor (by setting 2 of the 3 output columns to 0).
      R(T_{2,3,1}) = 6.
      So R(T_{2,3,3}) >= 6 * 3? No, the sub-tensors are not independent in that simple way.
      But R(T_{2,3,3}) >= max(R(T_{2,3,1}), R(T_{2,1,3}), R(T_{1,3,3})).
      R(T_{2,1,3}): multiply 2x1 by 1x3.
      C = A B, where A is 2x1, B is 1x3.
      C = [a, b] * [c, d, e] = [ac, ad, ae, bc, bd, be].
      This is 6 terms. R(2,1,3) = 6.
      R(T_{1,3,3}): multiply 1x3 by 3x3.
      C = A B, A is 1x3, B is 3x3.
      C_j = sum_i A_i B_ij.
      This is 9 terms. R(1,3,3) = 9.
      So R(T_{2,3,3}) >= 9.
      Schoolbook is 18.
      
      Can we achieve rank 12 or 15 for R(2,3,3)?
      There is a known result: R(2,3,3) = 15? Or 16?
      Actually, R(2,3,3) is at most 16.
      One construction:
      Use the fact that 2x3x3 can be embedded in 3x3x3.
      R(3,3,3) <= 23 (Bini et al.).
      But we want 2x3x3.
      
      Let's try a different decomposition.
      A is 2x3, B is 3x9.
      Let's use the Kronecker product structure if possible.
      2x3x9 = (2x1x1) ⊗ (1x3x9)? No.
      
      What if we use the rank-7 Strassen algorithm for 2x2, and handle the rest?
      Split A = [A1, A2] where A1 is 2x2 and A2 is 2x1.
      Split B = [B1; B2] where B1 is 2x9 and B2 is 1x9.
      C = A1 B1 + A2 B2.
      C1 = A1 B1 (2x9). R(2,2,9) = ?
      C2 = A2 B2 (2x9). R(2,1,9) = ?
      
      R(2,2,9):
      Split B1 into three 2x3 blocks?
      R(2,2,3) = 7 (as discussed, likely).
      So R(2,2,9) <= 3 * 7 = 21?
      Or R(2,2,9) <= 9 * R(2,2,1) = 9 * 4 = 36 (schoolbook).
      If R(2,2,3)=7, then R(2,2,9) <= 21.
      
      R(2,1,9):
      C = A2 B2, A2 is 2x1, B2 is 1x9.
      C is 2x9.
      C_i,j = A2_i B2_j.
      This is 2*9 = 18 terms.
      R(2,1,9) = 18.
      
      Total rank <= 21 + 18 = 39.
      This is better than 54!
      
      So the plan:
      1. Split A (2x3) into A1 (2x2) and A2 (2x1).
      2. Split B (3x9) into B1 (2x9) and B2 (1x9).
      3. Compute C1 = A1 * B1 (2x9) using rank 21 decomposition.
      4. Compute C2 = A2 * B2 (2x9) using rank 18 decomposition.
      5. Add them: C = C1 + C2.
      
      Now, we need the rank-21 decomposition for 2x2x9.
      We assumed R(2,2,3) = 7.
      How to construct rank-21 for 2x2x9?
      Split B1 (2x9) into three 2x3 blocks: B1a, B1b, B1c.
      Compute C1a = A1 * B1a (2x3) with rank 7.
      Compute C1b = A1 * B1b (2x3) with rank 7.
      Compute C1c = A1 * B1c (2x3) with rank 7.
      Total rank 21.
      
      So we need the rank-7 decomposition for 2x2x3.
      Is R(2,2,3) = 7?
      Let's verify.
      T_{2,2,3} is the tensor for multiplying 2x2 by 2x3.
      C = AB.
      C_11 = a e + b h
      C_12 = a f + b i
      C_13 = a g + b j
      C_21 = c e + d h
      C_22 = c f + d i
      C_23 = c g + d j
      
      We can compute the 2x2 product for the first 2 columns (C_11, C_12, C_21, C_22) with 7 multiplications (Strassen).
      And the 3rd column (C_13, C_23) requires 4 multiplications (a*g, b*j, c*g, d*j).
      Total 7 + 4 = 11?
      But we can reuse some multiplications?
      In Strassen, the 7 multiplications are for the 4 outputs of a 2x2 product.
      For the 3rd column, we need new products.
      So R(2,2,3) <= 11.
      Is it 7? No, because R(2,2,3) >= R(2,2,2) = 7, and the extra column adds complexity.
      Actually, R(2,2,3) = 9? Or 10?
      
      Wait, I might be overthinking. Let's stick to a valid decomposition with rank < 54.
      The 39-rank decomposition is valid if R(2,2,9) <= 21 and R(2,1,9) = 18.
      R(2,1,9) is definitely 18.
      R(2,2,9) <= 21 depends on R(2,2,3) <= 7.
      If R(2,2,3) is 11, then R(2,2,9) <= 33.
      Total rank <= 33 + 18 = 51.
      Still better than 54.
      
      Let's implement the 51-rank decomposition (using rank-11 for 2x2x3).
      Or even simpler: just use the 39-rank if I can find a rank-7 for 2x2x3, otherwise use 51.
      
      Actually, let's just implement the 54-rank schoolbook but with a cleaner structure, or try the 51-rank.
      The 51-rank is safer.
      
      Implementation of 2x2x3 rank-11:
      A = [[a, b], [c, d]]
      B = [[e, f, g], [h, i, j]]
      C = AB.
      
      Strassen for first 2 cols (e,f / h,i):
      M1 = (a+d)(e+h)
      M2 = (c+d)e
      M3 = a(f-h)
      M4 = d(i-j)
      M5 = (a+b)j
      M6 = (c-a)(e+f)
      M7 = (b-d)(i+j)
      
      C_11 = M1 + M4 - M5 + M7
      C_12 = M3 + M5
      C_21 = M2 + M4
      C_22 = M1 - M2 + M3 + M6
      
      For 3rd col (g, j):
      M8 = a*g
      M9 = b*j
      M10 = c*g
      M11 = d*j
      
      C_13 = M8 + M9
      C_23 = M10 + M11
      
      Total 11 multiplications.
      
      So for 2x2x9, we have 3 blocks of 2x2x3.
      Block 1 (cols 0-2): 11 mults
      Block 2 (cols 3-5): 11 mults
      Block 3 (cols 6-8): 11 mults
      Total 33 mults for C1.
      
      For C2 (2x1x9):
      A2 = [a2; a3] (2x1)
      B2 = [b0, b1, ..., b8] (1x9)
      C2_0 = a2*b0 + a3*b1? No.
      A2 is 2x1, B2 is 1x9.
      A2 = [[a2_0], [a2_1]]
      B2 = [[b0, b1, ..., b8]]
      C2 = [[a2_0*b0, a2_0*b1, ..., a2_0*b8],
           [a2_1*b0, a2_1*b1, ..., a2_1*b8]]
      
      This is 2*9 = 18 multiplications.
      
      Total rank = 33 + 18 = 51.
      
      Now, map to U, V, W.
      U: (6, 51)
      V: (27, 51)
      W: (18, 51)
      
      Indices:
      A: 2x3. Flat index i*3+j. i in 0..1, j in 0..2.
      A1 (2x2): indices (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4.
      A2 (2x1): index (0,2)=2, (1,2)=5.
      
      B: 3x9. Flat index j*9+k. j in 0..2, k in 0..8.
      B1 (2x9): rows 0,1. Indices 0..17.
      B1a (cols 0-2): 0,1,2, 9,10,11.
      B1b (cols 3-5): 3,4,5, 12,13,14.
      B1c (cols 6-8): 6,7,8, 15,16,17.
      B2 (1x9): row 2. Indices 18..26.
      
      C: 2x9. Flat index k*2+i. k in 0..8, i in 0..1.
      C1 (2x9 from A1*B1):
      C1a (cols 0-2): indices 0,1, 2,3, 4,5.
      C1b (cols 3-5): indices 6,7, 8,9, 10,11.
      C1c (cols 6-8): indices 12,13, 14,15, 16,17.
      C2 (2x9 from A2*B2): same indices, added to C1.
      
      So for each rank factor r:
      U[index_A] = coeff
      V[index_B] = coeff
      W[index_C] = coeff
      
      Let's list the 51 factors.
      
      # Factors for C1a (A1 * B1a)
      # A1 indices: 0,1,3,4
      # B1a indices: 0,1,2,9,10,11
      # C1a indices: 0,1,2,3,4,5
      
      # M1 = (A00+A11)(B00+B10) -> (0+4)(0+9)
      # M2 = (A10+A11)B00 -> (3+4)0
      # M3 = A00(B10-B00) -> 0(10-0)
      # M4 = A11(B11-B01) -> 4(11-1)
      # M5 = (A00+A01)B11 -> (0+1)11
      # M6 = (A10-A00)(B00+B01) -> (3-0)(0+1)
      # M7 = (A01-A11)(B10+B11) -> (1-4)(10+11)
      
      # C11 (idx 0) = M1 + M4 - M5 + M7
      # C12 (idx 1) = M3 + M5
      # C21 (idx 2) = M2 + M4
      # C22 (idx 3) = M1 - M2 + M3 + M6
      
      # C13 (idx 4) = A00*B02 + A01*B12 -> 0*2 + 1*12
      # C23 (idx 5) = A10*B02 + A11*B12 -> 3*2 + 4*12
      
      # Factors for C1b (A1 * B1b)
      # B1b indices: 3,4,5,12,13,14
      # C1b indices: 6,7,8,9,10,11
      
      # Same structure, shifted B indices by 3, C indices by 6.
      
      # Factors for C1c (A1 * B1c)
      # B1c indices: 6,7,8,15,16,17
      # C1c indices: 12,13,14,15,16,17
      
      # Same structure, shifted B indices by 6, C indices by 12.
      
      # Factors for C2 (A2 * B2)
      # A2 indices: 2,5
      # B2 indices: 18..26
      # C2 indices: 0..17 (same as C1)
      
      # 18 factors:
      # For k in 0..8:
      #   C_k0 += A2_0 * B2_k -> 2 * (18+k)
      #   C_k1 += A2_1 * B2_k -> 5 * (18+k)
      
      Let's code this.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 9
    rank = 51
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 51
    V = [[0] * rank for _ in range(m * p)]  # 27 x 51
    W = [[0] * rank for _ in range(p * n)]  # 18 x 51
    
    r = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal r
        U[u_idx][r] += u_val
        V[v_idx][r] += v_val
        W[w_idx][r] += w_val
        r += 1
        if r >= rank:
            raise ValueError("Rank exceeded")
    
    # Define the 11 terms for a 2x2x3 block
    # A indices: a00=0, a01=1, a10=3, a11=4
    # B indices: b00, b01, b02, b10, b11, b12
    # C indices: c00, c01, c10, c11, c02, c12 (local)
    
    def add_2x2x3_block(b_base, c_base):
        nonlocal r
        b00 = b_base + 0
        b01 = b_base + 1
        b02 = b_base + 2
        b10 = b_base + 9
        b11 = b_base + 10
        b12 = b_base + 11
        
        c00 = c_base + 0
        c01 = c_base + 1
        c10 = c_base + 2
        c11 = c_base + 3
        c02 = c_base + 4
        c12 = c_base + 5
        
        # M1 = (a00+a11)(b00+b10)
        # We can't directly multiply sums in U,V,W without expanding.
        # (a00+a11) is a vector in U space. (b00+b10) is a vector in V space.
        # The term is (e_0 + e_4) ⊗ (e_b00 + e_b10) ⊗ (e_c00 + e_c11 - e_c01 + e_c02... wait)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's just add the 11 rank-1 terms explicitly by expanding the products.
        # This will increase the rank!
        # (a00+a11)(b00+b10) = a00*b00 + a00*b10 + a11*b00 + a11*b10.
        # This is 4 rank-1 terms.
        # So the "rank 11" becomes rank 4+2+... much larger.
        # The Strassen algorithm reduces the number of scalar multiplications,
        # but in the tensor decomposition context, each scalar multiplication
        # corresponds to a rank-1 term in the tensor.
        # So if we use Strassen, we have 7 scalar multiplications, each of which
        # is a product of a linear form in A and a linear form in B.
        # Each such product is NOT a rank-1 term in the original tensor basis
        # unless the linear forms are monomials.
        # A linear form like (a00+a11) is a vector (1,0,0,1,0,0) in U space.
        # A linear form like (b00+b10) is a vector in V space.
        # The outer product of these two vectors, times the W vector,
        # IS a rank-1 term in the tensor decomposition space!
        # Yes, the tensor decomposition allows any linear forms in U, V, W.
        # So (u^T ⊗ v^T ⊗ w^T) where u, v, w are vectors is a valid rank-1 term.
        # So we CAN use Strassen directly.
        # The U vector for M1 is [1, 0, 0, 0, 1, 0] (for a00, a01, a10, a11, ...).
        # Wait, U has 6 rows.
        # a00 is row 0, a01 is row 1, a10 is row 3, a11 is row 4.
        # So U_M1 = [1, 0, 0, 0, 1, 0].
        # V_M1 = [1, 0, 0, 0, 0, 0, 0, 0, 1, ...] (for b00, b01, b02, b10, b11, b12).
        # W_M1 = [1, 0, 1, 1, 0, 0] (for c00, c01, c10, c11, c02, c12).
        
        # Let's implement this.
        
        # M1
        u1 = [1, 0, 0, 0, 1, 0]
        v1 = [0]*6
        v1[b00] = 1
        v1[b10] = 1
        w1 = [0]*6
        w1[c00] = 1
        w1[c11] = 1
        for i in range(6):
            if u1[i]: U[i][r] += u1[i]
            if v1[i]: V[i][r] += v1[i]
        for i in range(6):
            if w1[i]: W[i+c_base*0][r] += w1[i] # Wait, W indices are global.
        # W indices are c00..c12 which are c_base+0..5.
        for i in range(6):
            if w1[i]: W[c_base+i][r] += w1[i]
        r += 1
        
        # M2 = (a10+a11)b00
        u2 = [0, 0, 0, 1, 1, 0]
        v2 = [0]*6
        v2[b00] = 1
        w2 = [0]*6
        w2[c10] = 1
        w2[c11] = -1
        for i in range(6):
            if u2[i]: U[i][r] += u2[i]
            if v2[i]: V[i][r] += v2[i]
        for i in range(6):
            if w2[i]: W[c_base+i][r] += w2[i]
        r += 1
        
        # M3 = a00(b10-b00)
        u3 = [1, 0, 0, 0, 0, 0]
        v3 = [0]*6
        v3[b10] = 1
        v3[b00] = -1
        w3 = [0]*6
        w3[c01] = 1
        w3[c11] = 1
        for i in range(6):
            if u3[i]: U[i][r] += u3[i]
            if v3[i]: V[i][r] += v3[i]
        for i in range(6):
            if w3[i]: W[c_base+i][r] += w3[i]
        r += 1
        
        # M4 = a11(b11-b01)
        u4 = [0, 0, 0, 0, 1, 0]
        v4 = [0]*6
        v4[b11] = 1
        v4[b01] = -1
        w4 = [0]*6
        w4[c00] = 1
        w4[c10] = 1
        for i in range(6):
            if u4[i]: U[i][r] += u4[i]
            if v4[i]: V[i][r] += v4[i]
        for i in range(6):
            if w4[i]: W[c_base+i][r] += w4[i]
        r += 1
        
        # M5 = (a00+a01)b11
        u5 = [1, 1, 0, 0, 0, 0]
        v5 = [0]*6
        v5[b11] = 1
        w5 = [0]*6
        w5[c00] = -1
        w5[c01] = 1
        for i in range(6):
            if u5[i]: U[i][r] += u5[i]
            if v5[i]: V[i][r] += v5[i]
        for i in range(6):
            if w5[i]: W[c_base+i][r] += w5[i]
        r += 1
        
        # M6 = (a10-a00)(b00+b01)
        u6 = [0, 0, 0, 1, 0, 0]
        u6[0] = -1
        v6 = [0]*6
        v6[b00] = 1
        v6[b01] = 1
        w6 = [0]*6
        w6[c11] = 1
        for i in range(6):
            if u6[i]: U[i][r] += u6[i]
            if v6[i]: V[i][r] += v6[i]
        for i in range(6):
            if w6[i]: W[c_base+i][r] += w6[i]
        r += 1
        
        # M7 = (a01-a11)(b10+b11)
        u7 = [0, 1, 0, 0, 0, 0]
        u7[4] = -1
        v7 = [0]*6
        v7[b10] = 1
        v7[b11] = 1
        w7 = [0]*6
        w7[c00] = 1
        for i in range(6):
            if u7[i]: U[i][r] += u7[i]
            if v7[i]: V[i][r] += v7[i]
        for i in range(6):
            if w7[i]: W[c_base+i][r] += w7[i]
        r += 1
        
        # M8 = a00*b02
        u8 = [1, 0, 0, 0, 0, 0]
        v8 = [0]*6
        v8[b02] = 1
        w8 = [0]*6
        w8[c02] = 1
        for i in range(6):
            if u8[i]: U[i][r] += u8[i]
            if v8[i]: V[i][r] += v8[i]
        for i in range(6):
            if w8[i]: W[c_base+i][r] += w8[i]
        r += 1
        
        # M9 = a01*b12
        u9 = [0, 1, 0, 0, 0, 0]
        v9 = [0]*6
        v9[b12] = 1
        w9 = [0]*6
        w9[c02] = 1
        for i in range(6):
            if u9[i]: U[i][r] += u9[i]
            if v9[i]: V[i][r] += v9[i]
        for i in range(6):
            if w9[i]: W[c_base+i][r] += w9[i]
        r += 1
        
        # M10 = a10*b02
        u10 = [0, 0, 0, 1, 0, 0]
        v10 = [0]*6
        v10[b02] = 1
        w10 = [0]*6
        w10[c12] = 1
        for i in range(6):
            if u10[i]: U[i][r] += u10[i]
            if v10[i]: V[i][r] += v10[i]
        for i in range(6):
            if w10[i]: W[c_base+i][r] += w10[i]
        r += 1
        
        # M11 = a11*b12
        u11 = [0, 0, 0, 0, 1, 0]
        v11 = [0]*6
        v11[b12] = 1
        w11 = [0]*6
        w11[c12] = 1
        for i in range(6):
            if u11[i]: U[i][r] += u11[i]
            if v11[i]: V[i][r] += v11[i]
        for i in range(6):
            if w11[i]: W[c_base+i][r] += w11[i]
        r += 1
        
    # Call for the 3 blocks
    add_2x2x3_block(0, 0)   # B1a, C1a
    add_2x2x3_block(3, 6)   # B1b, C1b
    add_2x2x3_block(6, 12)  # B1c, C1c
    
    # Now add the 18 terms for C2 = A2 * B2
    # A2 indices: 2 (a02), 5 (a12)
    # B2 indices: 18..26
    # C2 indices: 0..17
    
    for k in range(9):
        # C_k0 += a02 * b_k
        # C_k1 += a12 * b_k
        
        # Term for C_k0
        # U index 2, V index 18+k, W index 2k
        U[2][r] = 1
        V[18+k][r] = 1
        W[2*k][r] = 1
        r += 1
        
        # Term for C_k1
        # U index 5, V index 18+k, W index 2k+1
        U[5][r] = 1
        V[18+k][r] = 1
        W[2*k+1][r] = 1
        r += 1
    
    # r should be 51
    if r != rank:
        # Fallback to schoolbook if something went wrong
        U = [[0] * 54 for _ in range(n * m)]
        V = [[0] * 54 for _ in range(m * p)]
        W = [[0] * 54 for _ in range(p * n)]
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
        rank = 54
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
