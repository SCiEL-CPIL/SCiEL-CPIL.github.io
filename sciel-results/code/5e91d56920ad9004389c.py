"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Compute a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Strategy: Split the inner dimension m=5 into 2+3.
    - The 4x2x6 sub-tensor is decomposed using Strassen's method for 4x4x4, but adapted.
      Actually, 4x2x6 is small enough that we can just use schoolbook or a specific known decomposition.
      Rank of 4x2x6 schoolbook: 4*2*6 = 48.
      However, we can do better by viewing it as part of a larger structure or using known results.
      R(4,2,6) = ? 
      Let's use the property that R(n,m,p) is sub-multiplicative.
      But for small dimensions, explicit constructions are better.
      
      Let's try a different split: 5 = 1 + 4.
      R(4,1,6) = 4*1*6 = 24 (schoolbook, and it's minimal since it's rank of a matrix, actually no, it's a tensor.
      Wait, for m=1, it's just scaling and adding. The tensor T_{4,1,6} has entries T_{i,1,k} = delta_{ik}? No.
      (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}. If m=1, (A*B)_{i,k} = A_{i,1} B_{1,k}.
      This is rank 1 * 4 * 6? No, it's the sum of 24 rank-1 tensors?
      T_{i, j, k} = 1 if i,j,k match? No.
      The tensor for matrix multiplication C_{ik} = sum_j A_{ij} B_{jk}.
      The tensor T has indices i, j, k. T_{ijk} = 1 if the indices correspond to the multiplication term.
      For m=1, T_{i1k} = 1. This tensor has rank 4*6 = 24?
      Let's check: T = sum_{i=1}^4 sum_{k=1}^6 e_i \otimes e_1 \otimes e_k.
      This is a sum of 24 rank-1 tensors. Can it be lower?
      The tensor is of the form u \otimes v \otimes w? No.
      It's a "slice" tensor. The rank of T_{n,1,p} is n*p.
      So R(4,1,6) = 24.
      
      Now R(4,4,6).
      Schoolbook: 4*4*6 = 96.
      Can we do better?
      Use Strassen for 4x4x4? R(4,4,4) = 49? No, R(4,4,4) is known to be around 49?
      Actually, R(4,4,4) is not exactly known but is <= 49.
      But we have 4x4x6.
      We can embed 4x4x6 into 4x8x8? Or use block decomposition.
      Split the 6 dimension into 2+4?
      R(4,4,6) <= R(4,4,2) + R(4,4,4).
      R(4,4,2) = ? Schoolbook 32.
      R(4,4,4) <= 49.
      Total 81.
      
      Let's try to construct an explicit decomposition.
      This is getting complex. Let's look for a simpler, robust approach.
      
      Alternative: Use the "Winograd" or "Toom-Cook" style polynomial evaluation/interpolation?
      Matrix multiplication is bilinear.
      
      Let's stick to the split 5 = 2 + 3.
      R(4,2,6) + R(4,3,6).
      R(4,2,6): Schoolbook 48.
      Can we reduce R(4,2,6)?
      View as 4x4x6 by padding? No.
      
      Let's try the known decomposition for R(3,3,3) = 23 (Strassen).
      If we can use Strassen for parts, we can save.
      
      Let's split 5 = 3 + 2.
      Part 1: 4x3x6.
      Split 6 into 3+3.
      R(4,3,6) <= R(4,3,3) + R(4,3,3).
      R(4,3,3): Split 4 into 2+2.
      R(4,3,3) <= R(2,3,3) + R(2,3,3).
      R(2,3,3): Split 3 into 2+1?
      R(2,3,3) <= R(2,3,2) + R(2,3,1).
      R(2,3,1) = 6.
      R(2,3,2) <= R(2,2,2) + R(2,1,2) = 7 + 4 = 11.
      So R(2,3,3) <= 17.
      R(4,3,3) <= 34.
      R(4,3,6) <= 68.
      
      Part 2: 4x2x6.
      Split 6 into 3+3.
      R(4,2,6) <= 2 * R(4,2,3).
      R(4,2,3): Split 4 into 2+2.
      R(4,2,3) <= 2 * R(2,2,3).
      R(2,2,3) = 11 (Known: R(2,2,3)=11).
      So R(4,2,3) <= 22.
      R(4,2,6) <= 44.
      
      Total Rank <= 68 + 44 = 112.
      Schoolbook is 120.
      Improvement!
      
      We need explicit decompositions for:
      1. R(2,2,3) = 11.
      2. Combine them to build the full U, V, W.
      
      Finding the explicit 11-term decomposition for 2x2x3 is non-trivial to hardcode from memory.
      If I can't provide the exact coefficients, I might fail validity.
      
      Is there a simpler improvement?
      What if I just use the schoolbook but with a smaller split that I *do* know?
      
      Let's try 5 = 1 + 4.
      R(4,1,6) = 24.
      R(4,4,6).
      Split 6 into 2+4.
      R(4,4,6) <= R(4,4,2) + R(4,4,4).
      R(4,4,2) = 32 (Schoolbook).
      R(4,4,4).
      Split 4 into 2+2.
      R(4,4,4) <= 4 * R(2,2,2) = 4 * 7 = 28.
      So R(4,4,6) <= 32 + 28 = 60.
      Total Rank <= 24 + 60 = 84.
      
      This is much better (84 vs 120).
      And I know the Strassen decomposition for 2x2x2 (Rank 7).
      I don't need 2x2x3.
      
      Let's verify the construction.
      We need to build U, V, W for:
      1. The 4x1x6 part (Rank 24).
      2. The 4x4x6 part, which is split into:
         a. 4x4x2 part (Rank 32, schoolbook).
         b. 4x4x4 part (Rank 28, using 4 copies of Strassen 2x2x2).
         
      Wait, the 4x4x6 part corresponds to multiplying a 4x4 matrix by a 4x6 matrix.
      The result is 4x6.
      We split the output dimension 6 into 2+4?
      Or the input dimension?
      The tensor is T_{i,j,k}.
      i in 1..4, j in 1..4, k in 1..6.
      If we split k into 1..2 and 3..6, we have two tensors: T_{i,j,k1} and T_{i,j,k2}.
      T_{i,j,k1} is 4x4x2. Rank 32.
      T_{i,j,k2} is 4x4x4. Rank 28.
      Total 60.
      
      So the plan is:
      Total Rank = 24 (from m=1 part) + 32 (from m=4, k=1..2 part) + 28 (from m=4, k=3..6 part) = 84.
      
      Steps:
      1. Initialize U, V, W with zeros. Dimensions:
         U: (4*5, rank)
         V: (5*6, rank)
         W: (6*4, rank)
         
      2. Assign columns for the m=1 part (j=1).
         For i in 0..3, k in 0..5:
           U[i*5 + 1, col] = 1
           V[1*6 + k, col] = 1
           W[k*4 + i, col] = 1
         (24 columns)
         
      3. Assign columns for the m=4 part (j=2..5).
         Let j' = j-1, so j' in 0..3.
         The sub-matrix A is 4x4 (rows 0..3, cols 1..4 of original A, which are j=1..4? No, j=2..5 in original 0-indexed is 1..4?
         Original m=5. Indices 0,1,2,3,4.
         m=1 part used j=0.
         m=4 part uses j=1,2,3,4.
         
         Part 3a: k=0,1 (first 2 columns of B, which are 5x2).
         We need to decompose 4x4x2 tensor.
         This is schoolbook: 4*4*2 = 32 terms.
         For i in 0..3, j' in 0..3, k' in 0..1:
           U[i*5 + (j'+1), col] = 1
           V[(j'+1)*6 + k', col] = 1
           W[k'*4 + i, col] = 1
         (32 columns)
         
         Part 3b: k=2,3,4,5 (last 4 columns of B, which are 5x4).
         We need to decompose 4x4x4 tensor using Strassen.
         Split the 4x4 matrix A into four 2x2 blocks: A00, A01, A10, A11.
         Split the 4x4 matrix B (the sub-matrix for k=2..5) into four 2x2 blocks: B00, B01, B10, B11.
         The result C is 4x4.
         Strassen computes C using 7 multiplications of 2x2 matrices.
         Each 2x2 multiplication has rank 7.
         Total 7 * 7 = 49? No, Strassen reduces the number of scalar multiplications.
         The tensor rank of 4x4x4 is NOT 7*7=49 in this additive sense.
         Strassen's algorithm computes the product using 7 products of 2x2 matrices.
         Each 2x2 product is a tensor of rank 7.
         So the total rank is 7 * 7 = 49?
         Wait. If I use 7 Strassen steps, and each step is a rank-7 tensor, do I just add them?
         Yes, the tensor is the sum of the 7 sub-tensors.
         So Rank(4,4,4) <= 7 * Rank(2,2,2) = 7 * 7 = 49.
         But schoolbook is 64. So 49 is an improvement.
         
         Wait, is Rank(4,4,4) actually 49?
         Strassen shows R(4,4,4) <= 49.
         Is there a better one? R(4,4,4) is known to be at most 49.
         So my estimate of 28 was wrong. It is 49.
         
         Let's re-calculate the total.
         R(4,1,6) = 24.
         R(4,4,2) = 32.
         R(4,4,4) <= 49.
         Total = 24 + 32 + 49 = 105.
         
         105 is better than 120.
         
         Can we do better on R(4,4,4)?
         Maybe split 4 into 1+3?
         R(4,4,4) <= R(1,4,4) + R(3,4,4) + R(4,1,4) + R(4,3,4)? No, that's for partitioning indices.
         
         Let's check if 105 is valid and implementable.
         I need the explicit Strassen decomposition for 2x2x2.
         Strassen's 7 products:
         M1 = (A11 + A22)(B11 + B22)
         M2 = (A21 + A22)B11
         M3 = A11(B12 - B22)
         M4 = A22(B21 - B11)
         M5 = (A11 + A12)B22
         M6 = (A21 - A11)(B11 + B12)
         M7 = (A12 - A22)(B21 + B22)
         
         C11 = M1 + M4 - M5 + M7
         C12 = M3 + M5
         C21 = M2 + M4
         C22 = M1 - M2 + M3 + M6
         
         I need to translate this into U, V, W entries.
         For a 2x2x2 block, U is 4x7, V is 4x7, W is 4x7.
         Then I combine 4 such blocks for the 4x4x4 case.
         
         This is a solid plan. 105 < 120.
         
         Let's double check if I can improve R(4,4,2).
         R(4,4,2) = 32.
         Split 4 into 2+2.
         R(4,4,2) <= 2 * R(2,4,2) + 2 * R(4,2,2)? No.
         R(4,4,2) is the tensor for multiplying 4x4 by 4x2.
         It is just 8 schoolbook multiplications for each of the 4 rows?
         Actually, it's 4 (rows of A) * 4 (cols of A) * 2 (cols of B) = 32.
         Is there a Strassen-like reduction for rectangular?
         Usually, square is best for Strassen.
         
         So 105 seems like a good target.
         
         Implementation details:
         1. Helper to add a 2x2x2 Strassen block to U, V, W at specific offsets.
         2. Helper to add a schoolbook block.
         
         Let's write the code.
         
         Offsets:
         U index: i * m + j. m=5.
         V index: j * p + k. p=6.
         W index: k * n + i. n=4.
         
         Block 1: m=1 (j=0).
         Schoolbook 4x1x6.
         i in 0..3, k in 0..5.
         U[i*5 + 0, col] = 1
         V[0*6 + k, col] = 1
         W[k*4 + i, col] = 1
         
         Block 2: m=4 (j=1..4), k=0..1.
         Schoolbook 4x4x2.
         i in 0..3, j' in 0..3, k' in 0..1.
         j = j' + 1.
         U[i*5 + j, col] = 1
         V[j*6 + k', col] = 1
         W[k'*4 + i, col] = 1
         
         Block 3: m=4 (j=1..4), k=2..5.
         Strassen 4x4x4.
         We view A as 2x2 blocks of 2x2.
         A00: i=0,1; j=1,2 (j'=0,1)
         A01: i=0,1; j=3,4 (j'=2,3)
         A10: i=2,3; j=1,2 (j'=0,1)
         A11: i=2,3; j=3,4 (j'=2,3)
         
         B is 4x4 (cols 2..5).
         B00: j=1,2; k=2,3
         B01: j=1,2; k=4,5
         B10: j=3,4; k=2,3
         B11: j=3,4; k=4,5
         
         Wait, B is 5x6. We are using the sub-matrix B[1:5, 2:6] (0-indexed).
         Rows of B are j. Cols are k.
         So B_sub[j', k'] where j' in 0..3, k' in 0..3.
         B_sub[0,0] corresponds to j=1, k=2.
         
         Strassen combines these.
         For each of the 7 Strassen products, we have a 2x2x2 tensor.
         We need to add 7 * 4 = 28? No.
         Strassen computes the 4x4 product using 7 multiplications of 2x2 matrices.
         Each 2x2 multiplication is a rank-7 tensor.
         So we have 7 rank-7 tensors.
         Total rank contribution: 7 * 7 = 49.
         
         We need to map the Strassen indices to the global U, V, W.
         
         Let's define a function `add_strassen_2x2x2(U, V, W, offset_u, offset_v, offset_w, ...)`.
         Actually, it's easier to just loop over the 7 Strassen steps and for each, add the 7 rank-1 terms of the 2x2x2 Strassen tensor.
         
         Wait, the 2x2x2 Strassen tensor itself has rank 7.
         So for each of the 7 "big" Strassen steps, we add 7 "small" rank-1 terms.
         Total 49 terms.
         
         Let's verify the indices for one 2x2x2 Strassen block.
         The block corresponds to a 2x2 A, 2x2 B, 2x2 C.
         A indices: (i_local, j_local) in {0,1}x{0,1}.
         B indices: (j_local, k_local) in {0,1}x{0,1}.
         C indices: (i_local, k_local) in {0,1}x{0,1}.
         
         Global indices:
         i_global = i_block_offset + i_local
         j_global = j_block_offset + j_local
         k_global = k_block_offset + k_local
         
         U index: i_global * 5 + j_global
         V index: j_global * 6 + k_global
         W index: k_global * 4 + i_global
         
         The 7 Strassen products for a 2x2x2 block:
         M1 = (A00+A11)(B00+B11)
         M2 = (A10+A11)B00
         M3 = A00(B01-B11)
         M4 = A11(B10-B00)
         M5 = (A00+A01)B11
         M6 = (A10-A00)(B00+B01)
         M7 = (A01-A11)(B10+B11)
         
         Each M is a product of two 2x2 matrices? No, M is a scalar multiplication of sums of elements.
         In tensor decomposition, a rank-1 term is u \otimes v \otimes w.
         For Strassen, we are decomposing the tensor into sum of rank-1 tensors.
         The 7 products are not rank-1 tensors in the global space, they are rank-1 in the *subspace* of the 2x2 blocks?
         No.
         
         Let's look at M1 = (A00+A11)(B00+B11).
         This contributes to C.
         C00 += M1
         C11 += M1
         
         The term M1 is (A00+A11) * (B00+B11).
         This is a product of a linear combination of A elements and a linear combination of B elements.
         So it is a sum of 4 rank-1 terms?
         (A00+A11)(B00+B11) = A00 B00 + A00 B11 + A11 B00 + A11 B11.
         This would be 4 rank-1 terms.
         But Strassen says we can do it in 7 products total for the 4 outputs.
         
         The key is that the "products" in Strassen are not scalar multiplications of scalars, but they are treated as black-box multiplications.
         When decomposing into rank-1 tensors, each "Strassen product" M_k is itself a tensor of rank 1 *only if* the inputs were scalars.
         But here, M_k is a product of 2x2 matrices.
         So M_k is a 2x2x2 tensor.
         And R(2,2,2) = 7.
         So each of the 7 Strassen products requires 7 rank-1 terms.
         Total 49.
         
         So the algorithm is:
         For each of the 7 Strassen steps:
           For each of the 7 rank-1 terms of the 2x2x2 Strassen tensor:
             Add 1 to U, V, W at the appropriate indices.
             
         This seems correct.
         
         Let's code this up.
         I will hardcode the 7x7 structure of the 2x2x2 Strassen tensor.
         
         Strassen 2x2x2 tensor entries (U, V, W for the 4 elements of A, B, C):
         A: 00, 01, 10, 11
         B: 00, 01, 10, 11
         C: 00, 01, 10, 11
         
         The 7 rank-1 tensors for 2x2x2 Strassen:
         1. U: [1, 0, 0, 1], V: [1, 0, 0, 1], W: [1, 0, 0, 1]  (M1: (A00+A11)(B00+B11) -> C00, C11)
            Wait, M1 contributes to C00 and C11.
            So W should be [1, 0, 0, 1].
            U is A00+A11 -> [1, 0, 0, 1].
            V is B00+B11 -> [1, 0, 0, 1].
            
         2. M2: (A10+A11)B00.
            U: [0, 0, 1, 1], V: [1, 0, 0, 0], W: [0, 0, 0, 0]?
            M2 contributes to C10 and C11.
            C10 += M2, C11 += -M2?
            Formula: C11 = M1 - M2 + M3 + M6. C10 = M2 + M4.
            So W for M2 is [0, 0, 1, -1].
            
         3. M3: A00(B01-B11).
            U: [1, 0, 0, 0], V: [0, 1, 0, -1], W: [0, 1, 0, 1].
            C01 += M3, C11 += M3.
            
         4. M4: A11(B10-B00).
            U: [0, 0, 0, 1], V: [0, 0, 1, -1], W: [1, 0, 1, 0].
            C00 += M4, C10 += M4.
            
         5. M5: (A00+A01)B11.
            U: [1, 1, 0, 0], V: [0, 0, 0, 1], W: [-1, 0, 0, 0].
            C00 += -M5.
            
         6. M6: (A10-A00)(B00+B01).
            U: [1, 0, -1, 0], V: [1, 1, 0, 0], W: [0, 0, 0, 1].
            C11 += M6.
            
         7. M7: (A01-A11)(B10+B11).
            U: [0, 1, 0, -1], V: [0, 0, 1, 1], W: [1, 0, 0, 0].
            C00 += M7.
            
         Let's check the sums:
         C00: M1 + M4 - M5 + M7. Correct.
         C01: M3 + M5. Correct.
         C10: M2 + M4. Correct.
         C11: M1 - M2 + M3 + M6. Correct.
         
         So we have 7 rank-1 tensors for the 2x2x2 case.
         
         Now, for the 4x4x4 case, we have 4 blocks of A and 4 blocks of B.
         We form 7 combinations of these blocks.
         For each combination, we have a 2x2x2 tensor.
         So we iterate 7 times (for M1..M7), and for each, we use the 7 rank-1 tensors above, but mapped to the global indices.
         
         Total 49 terms.
         
         This is implementable.
         
         Let's write the code.
         
         Time limit: 120s. This is fast.
         
         One detail: The task requires "integer/half-integer entries".
         My entries are -1, 0, 1. So integers. Good.
         
         Final check on rank:
         24 + 32 + 49 = 105.
         
         Let's go.
"""

def construct():
    n, m, p = 4, 5, 6
    rank = 105
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: m=1 (j=0), k=0..5. Rank 24.
    for i in range(4):
        for k in range(6):
            U[i * 5 + 0][col] = 1
            V[0 * 6 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
            
    # Block 2: m=4 (j=1..4), k=0..1. Rank 32.
    for i in range(4):
        for jp in range(4):
            j = jp + 1
            for k in range(2):
                U[i * 5 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
                
    # Block 3: m=4 (j=1..4), k=2..5. Rank 49.
    # Strassen 4x4x4.
    # A blocks:
    # A00: i=0,1; j=1,2
    # A01: i=0,1; j=3,4
    # A10: i=2,3; j=1,2
    # A11: i=2,3; j=3,4
    # B blocks (k=2..5):
    # B00: j=1,2; k=2,3
    # B01: j=1,2; k=4,5
    # B10: j=3,4; k=2,3
    # B11: j=3,4; k=4,5
    
    # Define the 7 Strassen steps for 2x2x2.
    # Each step is a tuple of (U_vec, V_vec, W_vec) of length 4.
    # U_vec corresponds to A00, A01, A10, A11.
    # V_vec corresponds to B00, B01, B10, B11.
    # W_vec corresponds to C00, C01, C10, C11.
    
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]), # M5
        ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    # For each of the 7 Strassen steps, we have a 2x2x2 tensor.
    # We need to expand this 2x2x2 tensor into 7 rank-1 terms?
    # No, the 2x2x2 tensor *is* the rank-7 tensor.
    # So for each of the 7 "big" steps, we add the 7 "small" rank-1 terms.
    
    for step in strassen_2x2:
        U_big, V_big, W_big = step
        # Now, for each of the 7 rank-1 terms of the 2x2x2 Strassen tensor:
        for term in strassen_2x2:
            U_small, V_small, W_small = term
            
            # Combine U_big and U_small
            # U_big is a vector of 4 coefficients for A blocks.
            # U_small is a vector of 4 coefficients for A elements within a block.
            
            # We need to add to U[i*5+j, col], V[j*6+k, col], W[k*4+i, col].
            
            # Loop over the 4 A blocks
            for a_block_idx in range(4):
                if U_big[a_block_idx] == 0:
                    continue
                # Determine i_range, j_range for this A block
                if a_block_idx == 0: # A00
                    i_start, i_end = 0, 2
                    j_start, j_end = 1, 3
                elif a_block_idx == 1: # A01
                    i_start, i_end = 0, 2
                    j_start, j_end = 3, 5
                elif a_block_idx == 2: # A10
                    i_start, i_end = 2, 4
                    j_start, j_end = 1, 3
                else: # A11
                    i_start, i_end = 2, 4
                    j_start, j_end = 3, 5
                
                # Loop over the 2 elements in i
                for i_local in range(2):
                    i = i_start + i_local
                    # Loop over the 2 elements in j
                    for j_local in range(2):
                        j = j_start + j_local
                        
                        # The coefficient from U_big is U_big[a_block_idx]
                        # The coefficient from U_small is U_small[?].
                        # Wait, U_small corresponds to the local indices of the 2x2 block.
                        # U_small[0] is A00, U_small[1] is A01, etc.
                        # But here, a_block_idx already selected the block.
                        # So we need to map (i_local, j_local) to the local index in the 2x2 block.
                        # Local index = i_local * 2 + j_local?
                        # A00 is (0,0), A01 is (0,1), A10 is (1,0), A11 is (1,1).
                        # So local_idx = i_local * 2 + j_local.
                        
                        local_idx = i_local * 2 + j_local
                        coeff_u = U_big[a_block_idx] * U_small[local_idx]
                        
                        if coeff_u == 0:
                            continue
                        
                        # Now we need to add the contribution to U, V, W.
                        # But we also need V and W.
                        # This approach is getting complicated because V and W also depend on the block.
                        
                        # Let's refactor.
                        # For a fixed (a_block_idx, b_block_idx, c_block_idx), we have a scalar coefficient.
                        # The term is coeff * A[i,j] * B[j,k] * C[i,k].
                        
                        # So we should loop over a_block, b_block, c_block.
                        
                        # Break out of the inner loops and restructure.
                        
                        # I will restructure the loop in the actual code.
                        pass

    # Restructured Block 3 Logic:
    # For each of the 7 Strassen steps (M1..M7):
    #   For each of the 7 rank-1 terms of the 2x2x2 tensor:
    #     For each a_block in 0..3:
    #       For each b_block in 0..3:
    #         For each c_block in 0..3:
    #           coeff = U_big[a_block] * V_big[b_block] * W_big[c_block] * U_small[a_local] * V_small[b_local] * W_small[c_local]
    #           This is too deep.
    
    # Simpler:
    # The 2x2x2 Strassen tensor has 7 rank-1 terms.
    # Each term is u \otimes v \otimes w.
    # u is a vector of 4, v is a vector of 4, w is a vector of 4.
    # For a fixed "big" step (which defines the block mapping), we add the 7 "small" terms.
    # But the "big" step is defined by U_big, V_big, W_big.
    # So the total tensor is sum_{big_step} sum_{small_term} (U_big \otimes U_small) \otimes (V_big \otimes V_small) \otimes (W_big \otimes W_small).
    
    # So we loop 7 * 7 = 49 times.
    # For each pair (big_step, small_term):
    #   We have vectors U_tot (length 16), V_tot (length 16), W_tot (length 16).
    #   U_tot is the Kronecker product? No.
    #   U_big is 4, U_small is 4.
    #   The combined U vector is of length 16 (4 blocks * 4 elements).
    #   U_tot[a_block * 4 + local_idx] = U_big[a_block] * U_small[local_idx].
    
    for big_idx in range(7):
        U_big, V_big, W_big = strassen_2x2[big_idx]
        for small_idx in range(7):
            U_small, V_small, W_small = strassen_2x2[small_idx]
            
            # Construct U_tot, V_tot, W_tot
            # Indices:
            # A: block_idx * 4 + local_idx
            # B: block_idx * 4 + local_idx
            # C: block_idx * 4 + local_idx
            
            for a_block in range(4):
                for a_local in range(4):
                    coeff_u = U_big[a_block] * U_small[a_local]
                    if coeff_u == 0: continue
                    # Map to global i, j
                    # a_block 0: i=0,1; j=1,2
                    # a_local 0: i=0, j=0 (local) -> i=0, j=1
                    # a_local 1: i=0, j=1 (local) -> i=0, j=2
                    # a_local 2: i=1, j=0 (local) -> i=1, j=1
                    # a_local 3: i=1, j=1 (local) -> i=1, j=2
                    
                    if a_block == 0:
                        i = a_local // 2
                        j = 1 + (a_local % 2)
                    elif a_block == 1:
                        i = a_local // 2
                        j = 3 + (a_local % 2)
                    elif a_block == 2:
                        i = 2 + (a_local // 2)
                        j = 1 + (a_local % 2)
                    else:
                        i = 2 + (a_local // 2)
                        j = 3 + (a_local % 2)
                    
                    # We need to find matching b, c.
                    # This is inefficient.
                    # Let's just add the rank-1 term directly.
                    # A rank-1 term is defined by (i, j, k).
                    # We need to find k such that the term is non-zero.
                    # But a rank-1 term in tensor decomposition is u_i v_j w_k.
                    # So for a fixed (i, j), k is determined by the support of w.
                    
                    # Actually, I should just loop over all i, j, k and add coeff_u * coeff_v * coeff_w.
                    # But that's O(N^3).
                    
                    # Let's stick to the rank-1 addition.
                    # For this specific (big_idx, small_idx), the tensor is:
                    # T = sum_{i,j,k} U_tot[i,j] V_tot[j,k] W_tot[i,k]
                    # Where U_tot, V_tot, W_tot are sparse.
                    
                    # Let's just accumulate into U, V, W.
                    # For each (i, j) in support of U_tot:
                    #   For each (j, k) in support of V_tot:
                    #     If j matches:
                    #       coeff = U_tot[i,j] * V_tot[j,k]
                    #       W[i,k] += coeff * W_tot[i,k] (if W_tot[i,k] is 1? No, W_tot is the coefficient).
                    
                    # This is getting messy.
                    # Let's just hardcode the 49 additions.
                    # Or better, use the fact that the 2x2x2 Strassen tensor is known.
                    
                    # I will revert to the simpler, albeit longer, explicit addition of the 49 terms.
                    pass

    # Given the complexity and time, I will return the schoolbook 120 if the 105 implementation is too risky.
    # But 105 is a solid improvement.
    # Let's try to implement the 105 version carefully.
    
    # Reset
    col = 0
    rank = 105
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1 & 2 as before
    for i in range(4):
        for k in range(6):
            U[i * 5 + 0][col] = 1
            V[0 * 6 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    for i in range(4):
        for jp in range(4):
            j = jp + 1
            for k in range(2):
                U[i * 5 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
                
    # Block 3: 49 terms.
    # I will generate the 49 rank-1 terms explicitly.
    for big_idx in range(7):
        U_big, V_big, W_big = strassen_2x2[big_idx]
        for small_idx in range(7):
            U_small, V_small, W_small = strassen_2x2[small_idx]
            
            # For each (a_block, a_local, b_block, b_local, c_block, c_local)
            # This is 4*4*4*4*4*4 = 4096 iterations.
            # We can prune zeros.
            
            for a_block in range(4):
                if U_big[a_block] == 0: continue
                for a_local in range(4):
                    if U_small[a_local] == 0: continue
                    coeff_u = U_big[a_block] * U_small[a_local]
                    
                    # Map a to (i, j)
                    if a_block == 0:
                        i = a_local // 2; j = 1 + (a_local % 2)
                    elif a_block == 1:
                        i = a_local // 2; j = 3 + (a_local % 2)
                    elif a_block == 2:
                        i = 2 + (a_local // 2); j = 1 + (a_local % 2)
                    else:
                        i = 2 + (a_local // 2); j = 3 + (a_local % 2)
                    
                    for b_block in range(4):
                        if V_big[b_block] == 0: continue
                        if b_block != a_block: 
                            # B and A must share the same j block?
                            # In Strassen, B blocks are matched with A blocks based on the formula.
                            # M1: A00+A11, B00+B11.
                            # So a_block and b_block are coupled.
                            # I need to handle the coupling.
                            pass
                        
                        # This is too complex to derive on the fly.
                        # I will fallback to Schoolbook 120 to ensure validity.
                        # The score will be 1.0, which is the baseline.
                        # But the prompt asks for a stronger algorithm.
                        # 105 is stronger.
                        
                        # I will implement the 105 version by hardcoding the 49 terms' indices.
                        # This is tedious but safe.
                        
    # Fallback to Schoolbook
    rank = 120
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(4):
        for j in range(5):
            for k in range(6):
                U[i * 5 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
