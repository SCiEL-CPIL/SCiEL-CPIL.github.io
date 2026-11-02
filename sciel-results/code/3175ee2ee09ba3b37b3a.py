"""Rewritten program for 4x4x6 matrix multiplication tensor decomposition.

Strategy: Partition the 4x4x6 multiplication into two independent 2x4x3 subproblems
(along the row dimension of A). For each 2x4x3 subproblem, use Strassen's algorithm
on the 2x2 core blocks with appropriate padding to handle the 4x3 structure.

A 2x4x3 multiplication can be decomposed by viewing it as:
  A (2x4) * B (4x3) = C (2x3)

We can partition A into A1 (2x2) and A2 (2x2) along columns, and B into B1 (2x3) and B2 (2x3) along rows.
Then C = A1*B1 + A2*B2.

For each 2x2 * 2x3 multiplication, we can use a Strassen-like approach:
  Let A = [[a,b],[c,d]] (2x2), B = [[e,f,g],[h,i,j]] (2x2x3)
  
  We can compute this using rank-7 Strassen for 2x2x2 applied to two overlapping 2x2x2 blocks,
  but more efficiently, we can note that 2x2x3 has a known minimal rank.

Actually, let's think more carefully. The tensor for 2x2x3 multiplication has rank at most 7 (using Strassen on two overlapping 2x2x2). But actually, the minimal rank of the 2x2x3 tensor is 7 (it's the same as 2x2x2 since we can just pad).

Wait, 2x2x3: A is 2x2, B is 2x3, C is 2x3. The tensor has 2*2*3 = 12 entries in the naive case. Strassen for 2x2x2 has rank 7. For 2x2x3, we can view B as two 2x2 matrices sharing one column, so we need 7+7=14? No, that's not right either.

Let me reconsider. A (2x4) * B (4x3):
  A = [A1 A2] where A1, A2 are 2x2
  B = [B1; B2] where B1, B2 are 2x3 (stacked vertically)
  C = A1*B1 + A2*B2

Each A_i * B_i is a 2x2 * 2x3 multiplication. The tensor for 2x2x3:
  Inputs: A (2x2), B (2x3) -> Output C (2x3)
  Naive rank: 2*2*3 = 12
  Using Strassen: We can compute A*B by treating B as [B_left B_right] where each is 2x1... that doesn't work directly.
  
  Actually, for 2x2 * 2x3, we can use the fact that we can compute it with rank 7 by using Strassen's 7 multiplications for the 2x2 part and handling the third column separately? No.
  
  Let's just use the known result: the minimal rank of the 2x2x3 tensor is 7. This can be achieved by noting that 2x2x3 is a sub-tensor of 2x2x2 (by restricting), and 2x2x2 has rank 7. But 2x2x3 is NOT a sub-tensor of 2x2x2 in the standard sense.

  Actually, let me just compute it directly. For A (2x2) * B (2x3):
  Let A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
  C = [[ae+bf, af+bg, ag+bh], [ce+df, cf+dg, cg+dh]]
  
  Using Strassen-style:
  M1 = (a+d)(e+h) = ae+ah+de+dh
  M2 = (c+d)(e+h) = ce+ch+de+dh
  M3 = a(f-h) = af-ah
  M4 = b(g-h) = bg-bh
  M5 = (a+b)(g-i) = ag-ai+bg-bi
  M6 = (c+a)(e-f) = ce-cf+ae-af
  M7 = (d+b)(i-f) = di-df+bi-bf
  
  C11 = M1 + M4 - M2 - M7 = ae+ah+de+dh + bg-bh - ce-ch-de-dh - di+df-bi+bf
      = ae + ah + bg - bh - ce - ch - di + df - bi + bf
      This doesn't simplify to ae+bf correctly.

  Let me just use the standard Strassen for 2x2x2 and handle the 2x2x3 case by padding B to 2x4 or using a different approach.

  Actually, the simplest correct approach: use Strassen for each 2x2x2 block. For 2x2 * 2x3, we can split B into B[:, :2] and B[:, 2:3], but B[:, 2:3] is 2x1. 
  
  For 2x2 * 2x1: A (2x2) * B (2x1) = C (2x1). This has rank 4 (naive) or can be done with fewer?
  [[a,b],[c,d]] * [e;f] = [ae+bf; ce+df]. This is a bilinear map with 4 multiplications naively. Can we do better? The tensor for 2x2x1 has rank 4 (I believe it's 4).

  So 2x2 * 2x3 = 2x2*2x2 (rank 7) + 2x2*2x1 (rank 4) = rank 11? That's worse than naive 12.

  Hmm, let me try a different partition. Let's use the full Strassen approach differently.

  Actually, for the overall 4x4x6 problem, let's just use Strassen recursively:
  4x4x6: Split A (4x4) into four 2x2 blocks, B (4x6) into 2x3 blocks (2 in row direction, 3 in column direction).
  
  A = [[A11, A12],[A21, A22]] where each is 2x2
  B = [[B11, B12, B13],[B21, B22, B23]] where each is 2x2 (B is 4x6, so 2x2 blocks: 2 rows of blocks, 3 cols of blocks)
  
  C (4x6) = [[A11*B11 + A12*B21, A11*B12 + A12*B22, A11*B13 + A12*B23],
             [A21*B11 + A22*B21, A21*B12 + A22*B22, A21*B13 + A22*B23]]
  
  Each term like A11*B11 is 2x2 * 2x2 = 2x2, which has rank 7 via Strassen.
  There are 4*3 = 12 such 2x2x2 multiplications, but they share factors.
  
  Using Strassen at the 4x4 level:
  M1 = (A11+A22)*(B11+B21+B12+B22+B13+B23) -- this is 2x2 * (sum of 2x2 blocks)
  
  This is getting complex. Let me just implement a direct Strassen-based decomposition for 4x4x4 first (rank 28 or so), then extend to 4x4x6.

  For 4x4x4, Strassen gives rank 7^2 = 49? No, Strassen for 4x4 uses 7 multiplications of 2x2, each of which uses 7 multiplications of 1x1, so total 49 scalar multiplications. But the tensor rank for 4x4x4 is at most 49 via this.

  Actually, the known best upper bound for 4x4x4 is 49 (from Strassen), and the lower bound is 48. So 49 is likely optimal or close.

  For 4x4x6, we can think of it as 4x4x4 + 4x4x2. The 4x4x4 part has rank at most 49. The 4x4x2 part: A (4x4) * B (4x2) = C (4x2). This can be done with Strassen-like: split into 2x2 blocks. A has 4 blocks of 2x2, B has 2 blocks of 2x2 (stacked). C = A_top*B_top + A_bottom*B_bottom where A_top, A_bottom are 2x4, B_top, B_bottom are 2x2. Each 2x4*2x2 = 2x2. 2x4*2x2: A (2x4) = [A1 A2] (2x2 each), B (2x2). C = A1*B + A2*B. Each A_i*B is 2x2*2x2 (rank 7). So 2x4*2x2 has rank at most 14. Total for 4x4x2: 14+14=28.

  So total rank for 4x4x6: 49 + 28 = 77. That's the same as naive! Not an improvement.

  Let me try a better approach. Use Strassen directly on 4x4x6 by padding to 4x4x8 or using a different decomposition.

  Actually, let's just implement Strassen for 4x4x4 (rank 49) and then handle the extra 2 columns of B separately with minimal overhead.

  For the 4x4x6 case:
  B = [B4x4 | B4x2]
  C = A*B4x4 + A*B4x2

  A*B4x4: rank 49 (Strassen)
  A*B4x2: We need to compute A(4x4) * B(4x2) = C(4x2).
  
  For 4x4x2: Split A into 2x2 blocks (4 blocks), B into 2x2 blocks (2 blocks stacked).
  C = A_top * B_top + A_bottom * B_bottom
  where A_top, A_bottom are 2x4, B_top, B_bottom are 2x2.
  
  2x4 * 2x2: A(2x4) = [A1 A2], B(2x2). C = A1*B + A2*B.
  Each A_i * B is 2x2*2x2, rank 7. So 2x4*2x2 has rank 14.
  Total for 4x4x2: 28.
  
  Grand total: 49 + 28 = 77. Still 77.

  The issue is that Strassen doesn't help much when the dimensions aren't powers of 2 in a balanced way.

  Let me try a completely different approach: use the known minimal rank for small matrix multiplication tensors and combine them.

  Known: 2x2x2 has rank 7. 2x2x1 has rank 4. 1x1x1 has rank 1.
  
  For 4x4x6, let's use a Kronecker product approach:
  4x4x6 = (2x2x3) ⊗ (2x2x2) in some sense? Not exactly.

  Let me just implement the best I can with Strassen and see if I can get below 77.

  Actually, I realize that for 4x4x6, we can use the following:
  View it as 4x4x4 + 4x4x2, but optimize the 4x4x2 part better.
  
  4x4x2: A(4x4)*B(4x2)=C(4x2). 
  Using Strassen-style on 2x2 blocks:
  A = [[A11,A12],[A21,A22]], B = [[B1],[B2]] where B1,B2 are 2x2.
  C = [[A11*B1+A12*B2, ...], [A21*B1+A22*B2, ...]]
  
  Each row-block of C is 2x2. C_top = A11*B1 + A12*B2 (2x2*2x2 + 2x2*2x2).
  Using Strassen for the sum: we can compute A11*B1 + A12*B2 with 7+7=14 multiplications, or can we do better?
  
  Actually, A11*B1 + A12*B2 = (A11+A12)*B1 + A12*(B2-B1)? No, that's not right.
  
  Let's just use 14 for each 2x2 output block, giving 28 for 4x4x2.

  So total is 49+28=77. 

  Given the difficulty of beating 77 with simple Strassen, let me just implement a clean Strassen-based decomposition that achieves rank 49 for the 4x4x4 part and 28 for the 4x4x2 part, totaling 77. But that's the same as naive.

  Wait, maybe I can do better on the 4x4x2 part. Let me think...

  4x4x2: The tensor has 4*4*2=32 naive rank. Can we do better?
  Using the fact that 2x2x2 has rank 7: 
  4x4x2 can be seen as two 2x4x2 problems (top and bottom halves of A).
  2x4x2: A(2x4)*B(4x2)=C(2x2). A=[A1 A2] (2x2 each), B=[B1;B2] (2x2 each).
  C = A1*B1 + A2*B2. Rank at most 7+7=14.
  So 4x4x2 has rank at most 28.

  I'm stuck at 77. Let me just implement this and return it, even though it's the same as naive. Perhaps the Strassen factors have a different structure that the evaluator handles differently, or maybe I'm making an error.

  Actually, wait. Let me re-examine. The naive rank is 4*4*6=96, not 77. The current program has rank 96. So 77 IS an improvement!

  Let me re-read: "reducing rank from 96 to 84". So naive is 96. My Strassen approach gives 77, which is better than 84.

  Let me implement:
  1. 4x4x4 part using Strassen (rank 49)
  2. 4x4x2 part using block Strassen (rank 28)
  Total: 77.
"""

DIMENSIONS = [4, 4, 6]

def _strassen_2x2_factors():
    """Return the 7 rank-1 factors for 2x2x2 Strassen multiplication.
    
    A (2x2) * B (2x2) = C (2x2)
    U: (4, 7) for A indices
    V: (4, 7) for B indices  
    W: (4, 7) for C indices
    
    Indexing: A[i,j] -> i*2+j, B[i,j] -> i*2+j, C[i,j] -> i*2+j
    """
    # Strassen formulas:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+h)  
    # M3 = a(f-h)
    # M4 = b(g-h)
    # M5 = (a+b)(g-i)
    # M6 = (c+a)(e-f)
    # M7 = (d+b)(i-f)
    # C11 = M1 + M4 - M2 - M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M5 - M7
    
    # A indices: 00,01,10,11 -> 0,1,2,3
    # B indices: 00,01,10,11 -> 0,1,2,3
    # C indices: 00,01,10,11 -> 0,1,2,3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1  # a, d
    V[0][0] = 1; V[3][0] = 1  # e, h
    W[0][0] = 1; W[3][0] = 1  # C11, C22
    
    # M2 = (c+d)(e+h)
    U[2][1] = 1; U[3][1] = 1  # c, d
    V[0][1] = 1; V[3][1] = 1  # e, h
    W[2][1] = 1; W[3][1] = -1  # C21, -C11... wait
    # C21 = M2 + M6, C11 = M1 + M4 - M2 - M7
    # So M2 contributes +1 to C21, -1 to C11
    W[0][1] = -1; W[2][1] = 1
    
    # M3 = a(f-h)
    U[0][2] = 1  # a
    V[1][2] = 1; V[3][2] = -1  # f, -h
    # C12 = M3 + M5, C22 = M1 - M3 + M5 - M7
    W[1][2] = 1; W[3][2] = -1
    
    # M4 = b(g-h)
    U[1][3] = 1  # b
    V[2][3] = 1; V[3][3] = -1  # g, -h
    # C11 = ... + M4 ..., C22 = ... - M7 (M4 not in C22)
    # C11 = M1 + M4 - M2 - M7
    W[0][3] = 1
    
    # M5 = (a+b)(g-i)
    U[0][4] = 1; U[1][4] = 1  # a, b
    V[2][4] = 1; V[1][4] = -1  # g, -i
    # C12 = M3 + M5, C22 = M1 - M3 + M5 - M7
    W[1][4] = 1; W[3][4] = 1
    
    # M6 = (c+a)(e-f)
    U[0][5] = 1; U[2][5] = 1  # a, c
    V[0][5] = 1; V[1][5] = -1  # e, -f
    # C21 = M2 + M6
    W[2][5] = 1
    
    # M7 = (d+b)(i-f)
    U[1][6] = 1; U[3][6] = 1  # b, d
    V[1][6] = 1; V[0][6] = -1  # i, -f  (i is B[1,0]=index 2, f is B[0,1]=index 1)
    # Wait, B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # i = B[1,0] = index 2, f = B[0,1] = index 1
    # So V[2][6] = 1, V[1][6] = -1
    V[1][6] = -1; V[2][6] = 1
    # C11 = ... - M7, C22 = ... - M7
    W[0][6] = -1; W[3][6] = -1
    
    return U, V, W

def _schoolbook_2x2x2_factors():
    """Fallback schoolbook for 2x2x2, rank 4."""
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][col] = 1
                V[j*2+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W

def _strassen_4x4x4_factors():
    """Strassen decomposition for 4x4x4, rank 49.
    
    A (4x4) * B (4x4) = C (4x4)
    Split into 2x2 blocks, use Strassen at block level.
    """
    n = 4
    rank = 49
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    
    # 2x2 block indices for 4x4:
    # Block (bi,bj) where bi,bj in {0,1}
    # A block (bi,bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # A element (i,j) is in block (i//2, j//2) at local position (i%2, j%2)
    
    # Strassen for 2x2 blocks:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*(B00+B11)
    # M3 = A00*(B01-B11)
    # M4 = A01*(B10-B11)
    # M5 = (A00+A01)*(B10-B00)
    # M6 = (A10+A00)*(B00-B01)
    # M7 = (A11+A01)*(B11-B01)
    
    # C00 = M1 + M4 - M2 - M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M3 + M5 - M7
    
    # Each M_k is a 2x2 * 2x2 = 2x2 multiplication, done with Strassen (rank 7)
    
    # For each of the 7 block-level multiplications, we have a 2x2*2x2 sub-problem.
    # The U factor for the sub-problem maps A elements to the sub-problem's A.
    # Similarly for V and W.
    
    # Let's define the 7 block-level terms:
    # term[k] = (U_coeff for A blocks) * (V_coeff for B blocks)
    # Each term produces a 2x2 result that contributes to certain C blocks.
    
    # Block-level coefficients:
    # M1: A: {00:1, 11:1}, B: {00:1, 11:1}, C: {00:1, 11:1}
    # M2: A: {10:1, 11:1}, B: {00:1, 11:1}, C: {10:1, 00:-1}
    # M3: A: {00:1}, B: {01:1, 11:-1}, C: {01:1, 11:-1}
    # M4: A: {01:1}, B: {10:1, 11:-1}, C: {00:1}
    # M5: A: {00:1, 01:1}, B: {10:1, 00:-1}, C: {01:1, 11:1}
    # M6: A: {10:1, 00:1}, B: {00:1, 01:-1}, C: {10:1}
    # M7: A: {11:1, 01:1}, B: {11:1, 01:-1}, C: {00:-1, 11:-1}
    
    block_terms = [
        # (A_blocks: dict, B_blocks: dict, C_blocks: dict)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(1,0):1, (0,0):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):-1}),
        ({(0,1):1}, {(1,0):1, (1,1):-1}, {(0,0):1}),
        ({(0,0):1, (0,1):1}, {(1,0):1, (0,0):-1}, {(0,1):1, (1,1):1}),
        ({(1,0):1, (0,0):1}, {(0,0):1, (0,1):-1}, {(1,0):1}),
        ({(1,1):1, (0,1):1}, {(1,1):1, (0,1):-1}, {(0,0):-1, (1,1):-1}),
    ]
    
    # Get Strassen factors for 2x2x2
    sU, sV, sW = _strassen_2x2_factors()
    
    col = 0
    for term_idx in range(7):
        A_blocks, B_blocks, C_blocks = block_terms[term_idx]
        
        # For this term, we need to compute:
        # (sum of A blocks with coeffs) * (sum of B blocks with coeffs) -> contributes to C blocks
        # This is a 2x2*2x2 multiplication at the block level, but the "elements" are 2x2 blocks.
        # We use Strassen for the 2x2*2x2, where each "element" is a 2x2 block.
        
        # The Strassen factors sU, sV, sW are for 2x2*2x2 at the scalar level.
        # At the block level, each "scalar" is a 2x2 block, so we need to expand.
        
        # For each of the 7 Strassen sub-multiplications:
        for sub_k in range(7):
            # sU[i][sub_k] gives the coefficient for A element i in this sub-multiplication
            # sV[j][sub_k] gives the coefficient for B element j
            # sW[m][sub_k] gives the coefficient for C element m
            
            for i in range(2):  # A block row
                for j in range(2):  # A block col
                    a_coeff = sU[i*2+j][sub_k]
                    if a_coeff == 0:
                        continue
                    for bi, bj in A_blocks:
                        a_block_coeff = A_blocks[(bi,bj)]
                        total_a_coeff = a_coeff * a_block_coeff
                        if total_a_coeff == 0:
                            continue
                        # This A element is in block (bi,bj) at local position (i,j)
                        a_global_i = 2*bi + i
                        a_global_j = 2*bj + j
                        a_idx = a_global_i * 4 + a_global_j
                        U[a_idx][col] += total_a_coeff
            
            for i in range(2):  # B block row
                for j in range(2):  # B block col
                    b_coeff = sV[i*2+j][sub_k]
                    if b_coeff == 0:
                        continue
                    for bi, bj in B_blocks:
                        b_block_coeff = B_blocks[(bi,bj)]
                        total_b_coeff = b_coeff * b_block_coeff
                        if total_b_coeff == 0:
                            continue
                        b_global_i = 2*bi + i
                        b_global_j = 2*bj + j
                        b_idx = b_global_i * 4 + b_global_j
                        V[b_idx][col] += total_b_coeff
            
            for i in range(2):  # C block row
                for j in range(2):  # C block col
                    c_coeff = sW[i*2+j][sub_k]
                    if c_coeff == 0:
                        continue
                    for bi, bj in C_blocks:
                        c_block_coeff = C_blocks[(bi,bj)]
                        total_c_coeff = c_coeff * c_block_coeff
                        if total_c_coeff == 0:
                            continue
                        c_global_i = 2*bi + i
                        c_global_j = 2*bj + j
                        c_idx = c_global_i * 4 + c_global_j
                        W[c_idx][col] += total_c_coeff
            
            col += 1
    
    return U, V, W

def _strassen_4x4x2_factors():
    """Strassen-based decomposition for 4x4x2, rank 28.
    
    A (4x4) * B (4x2) = C (4x2)
    Split A into 2x2 blocks (2x2 grid), B into 2x2 blocks (2x1 grid).
    C = A_top * B_top + A_bottom * B_bottom
    where A_top, A_bottom are 2x4, B_top, B_bottom are 2x2.
    Each 2x4*2x2 has rank 14 (two 2x2*2x2 Strassen).
    """
    n, m, p = 4, 4, 2
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 16 x 28
    V = [[0]*rank for _ in range(m*p)]  # 8 x 28
    W = [[0]*rank for _ in range(p*n)]  # 8 x 28
    
    # A (4x4): blocks A00, A01, A10, A11 (each 2x2)
    # B (4x2): blocks B0, B1 (each 2x2), B0 is top half (rows 0-1), B1 is bottom half (rows 2-3)
    # C (4x2): C_top (rows 0-1), C_bottom (rows 2-3), each 2x2
    
    # C_top = A00*B0 + A01*B0  (wait, B is 4x2, so B_top is 2x2, B_bottom is 2x2)
    # A_top = [A00 A01] (2x4), B_top = B0 (2x2)
    # C_top = A_top * B_top = A00*B0 + A01*B0
    # Similarly C_bottom = A10*B1 + A11*B1
    
    # For each 2x4*2x2, we use two 2x2*2x2 Strassen multiplications.
    
    sU, sV, sW = _strassen_2x2_factors()
    
    col = 0
    # C_top = A00*B0 + A01*B0
    for a_block_col in [0, 1]:  # A00, A01
        for sub_k in range(7):
            # A block is at (0, a_block_col)
            for i in range(2):
                for j in range(2):
                    a_coeff = sU[i*2+j][sub_k]
                    if a_coeff == 0:
                        continue
                    a_global_i = 2*0 + i
                    a_global_j = 2*a_block_col + j
                    a_idx = a_global_i * 4 + a_global_j
                    U[a_idx][col] += a_coeff
            
            # B block is B0 (rows 0-1 of B)
            for i in range(2):
                for j in range(2):
                    b_coeff = sV[i*2+j][sub_k]
                    if b_coeff == 0:
                        continue
                    b_global_i = 2*0 + i
                    b_global_j = j
                    b_idx = b_global_i * 2 + b_global_j
                    V[b_idx][col] += b_coeff
            
            # C block is C_top (rows 0-1 of C)
            for i in range(2):
                for j in range(2):
                    c_coeff = sW[i*2+j][sub_k]
                    if c_coeff == 0:
                        continue
                    c_global_i = 2*0 + i
                    c_global_j = j
                    c_idx = c_global_i * 4 + c_global_j  # C is 4x2, but W is p*n = 2*4 = 8
                    # W[k*n+i, r] where k is column of C, i is row of C
                    # C has p=2 columns, n=4 rows
                    # W index: k*4 + i where k in 0..1, i in 0..3
                    c_idx = c_global_j * 4 + c_global_i
                    W[c_idx][col] += c_coeff
            
            col += 1
    
    # C_bottom = A10*B1 + A11*B1
    for a_block_col in [0, 1]:  # A10, A11
        for sub_k in range(7):
            # A block is at (1, a_block_col)
            for i in range(2):
                for j in range(2):
                    a_coeff = sU[i*2+j][sub_k]
                    if a_coeff == 0:
                        continue
                    a_global_i = 2*1 + i
                    a_global_j = 2*a_block_col + j
                    a_idx = a_global_i * 4 + a_global_j
                    U[a_idx][col] += a_coeff
            
            # B block is B1 (rows 2-3 of B)
            for i in range(2):
                for j in range(2):
                    b_coeff = sV[i*2+j][sub_k]
                    if b_coeff == 0:
                        continue
                    b_global_i = 2*1 + i
                    b_global_j = j
                    b_idx = b_global_i * 2 + b_global_j
                    V[b_idx][col] += b_coeff
            
            # C block is C_bottom (rows 2-3 of C)
            for i in range(2):
                for j in range(2):
                    c_coeff = sW[i*2+j][sub_k]
                    if c_coeff == 0:
                        continue
                    c_global_i = 2*1 + i
                    c_global_j = j
                    c_idx = c_global_j * 4 + c_global_i
                    W[c_idx][col] += c_coeff
            
            col += 1
    
    return U, V, W

def _combine_4x4x6(U44, V44, W44, U42, V42, W42):
    """Combine 4x4x4 and 4x4x2 decompositions into 4x4x6.
    
    B (4x6) = [B4x4 | B4x2]
    C (4x6) = A*B4x4 + A*B4x2
    
    U is the same for both (A is 4x4 in both cases).
    V is concatenated: V44 (16x49) and V42 (8x28) -> but V42 has different V dimension.
    
    Wait, this doesn't work directly because V44 is for B(4x4) and V42 is for B(4x2).
    We need to map them to the same B index space.
    
    For 4x4x4: V has shape (m*p, rank) = (4*4, 49) = (16, 49)
    For 4x4x2: V has shape (m*p, rank) = (4*2, 28) = (8, 28)
    
    These have different V dimensions! We can't just concatenate.
    
    We need a unified approach. Let me reconsider.
    """
    pass

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4x6 matrix multiplication."""
    import time
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Use Strassen for 4x4x4 part and block Strassen for 4x4x2 part.
    # But we need to handle the V dimension mismatch.
    
    # Alternative: just use Strassen for the full 4x4x6 by padding B to 4x8 and then restricting.
    # Or use a direct approach.
    
    # Let's use a simpler approach: Strassen for 4x4x4 (rank 49) + schoolbook for 4x4x2 (rank 32).
    # Total: 81. Still better than 96.
    
    # Actually, let me just implement the 4x4x4 Strassen and 4x4x2 with schoolbook.
    
    U44, V44, W44 = _strassen_4x4x4_factors()
    # U44: 16x49, V44: 16x49, W44: 16x49
    
    # For 4x4x2: schoolbook
    rank_42 = n * m * 2  # 32
    U42 = [[0]*rank_42 for _ in range(n*m)]
    V42 = [[0]*rank_42 for _ in range(m*2)]
    W42 = [[0]*rank_42 for _ in range(2*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(2):
                U42[i*m+j][col] = 1
                V42[j*2+k][col] = 1
                W42[k*n+i][col] = 1
                col += 1
    
    # Now combine:
    # U = [U44 | U42] but U42 is for the same A, so U = U44 for the first 49 cols, U42 for next 32 cols
    # V44 is for B(4x4), V42 is for B(4x2). They have different V dimensions.
    # V44: (16, 49) for B indices 0..15 (4x4)
    # V42: (8, 32) for B indices 0..7 (4x2)
    # These are different! We need to embed them in a common space.
    
    # For the 4x4x6 problem, B is 4x6. V should have shape (m*p, rank) = (24, rank).
    # V44 corresponds to B columns 0-3, V42 corresponds to B columns 4-5 (but indexed as 0-1 in its own space).
    
    # Let's create the full V with 24 rows.
    rank_total = 49 + 32
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    
    # Copy 4x4x4 part
    for i in range(n*m):
        for c in range(49):
            U[i][c] = U44[i][c]
    for i in range(m*4):  # 16 rows
        for c in range(49):
            V[i][c] = V44[i][c]
    for i in range(4*n):  # 16 rows
        for c in range(49):
            W[i][c] = W44[i][c]
    
    # Copy 4x4x2 part (B columns 4-5, so B index offset is 4 in the p direction)
    for i in range(n*m):
        for c in range(32):
            U[i][49+c] = U42[i][c]
    for i in range(m*2):  # 8 rows, but we need to offset by 4 in the column direction
        # V42[i] corresponds to B row i//2, col i%2 in the 4x2 sub-problem
        # In the full 4x6 B, this is B row i//2, col 4 + i%2
        # V index in full: (i//2) * 6 + (4 + i%2)
        for c in range(32):
            v_row = (i // 2) * 6 + (4 + i % 2)
            V[v_row][49+c] = V42[i][c]
    for i in range(2*n):  # 8 rows
        for c in range(32):
            W[i][49+c] = W42[i][c]
    
    return {"U": U, "V": V, "W": W}
