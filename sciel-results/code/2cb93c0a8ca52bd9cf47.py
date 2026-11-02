"""
Find rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
Strategy: Deterministic Block-Structured Decomposition.
1. Base: Schoolbook (Rank 60).
2. Optimization: Apply Strassen's algorithm to 2x2 sub-blocks.
   - Partition A (2x5) into 2x2 and 2x3 blocks? No, dimensions are 2, 5, 6.
   - Let's partition A (2x5) into A11 (2x2) and A12 (2x3).
   - Partition B (5x6) into B11 (2x6) and B21 (3x6).
   - This doesn't directly allow Strassen because Strassen needs square blocks.
   
   Alternative Block Partition:
   - A is 2x5. B is 5x6.
   - We can view the multiplication as:
     C = A * B
     A = [A_left (2x2) | A_right (2x3)]
     B = [B_top (2x6) ; B_bottom (3x6)]
     
     C = A_left * B_top + A_right * B_bottom
     
     If we could decompose A_left * B_top and A_right * B_bottom efficiently...
     A_left is 2x2, B_top is 2x6.
     A_right is 2x3, B_bottom is 3x6.
     
     Let's try to partition into 2x2 blocks for the 2x2 parts.
     A_left (2x2) * B_top_left (2x2) -> 2x2 block of C.
     This part can be done with Strassen (rank 7 instead of 8).
     
     But B_top is 2x6. We can split B_top into B_tl (2x2), B_tm (2x2), B_tr (2x2).
     Similarly B_bottom is 3x6. Split into B_bl (3x2), B_bm (3x2), B_br (3x2).
     
     This seems complicated to mix Strassen with rectangular blocks.
     
     Let's stick to the known improvement: 
     The tensor rank of 2x5x6 matrix multiplication.
     Schoolbook: 60.
     
     Is there a known lower bound or construction?
     For 2x2x2, rank is 7 (Strassen).
     For 2x3x3, rank is ?
     
     Let's try a simpler structural reduction.
     Notice that the middle dimension is 5.
     We can split the sum over j=0..4 into two parts: j=0..1 and j=2..4.
     C = A[:, 0:2] * B[0:2, :] + A[:, 2:5] * B[2:5, :]
     
     Term 1: (2x2) * (2x6). 
     Term 2: (2x3) * (3x6).
     
     If we can find low-rank decompositions for these sub-products, we can sum them.
     
     Sub-product 1: M1 (2x2) * N1 (2x6) -> P1 (2x6).
     Tensor rank of 2x2x6?
     It is equivalent to 6 instances of 2x2x2 multiplications? No.
     P1[i, k] = sum_{j=0}^1 M1[i, j] N1[j, k].
     This is a linear map. The tensor T[i, j, k] = 1.
     Rank of 2x2x6 matrix multiplication tensor.
     Since N1 is 2x6, we can view it as 6 columns of 2x1 vectors.
     Actually, 2x2x6 is isomorphic to 2x2x2 * 3? No.
     
     Let's use the property: rank(2x2x6) <= 6 * rank(2x2x2) ? No.
     rank(2x2x6) is the rank of the tensor that maps A (2x2) and B (2x6) to C (2x6).
     This is equivalent to 3 independent 2x2x2 multiplications if we split B into 3 2x2 blocks?
     B = [B1, B2, B3] where each Bi is 2x2.
     C = [A*B1, A*B2, A*B3].
     The tensor for this is the direct sum of three 2x2x2 tensors.
     So rank(2x2x6) = 3 * rank(2x2x2) = 3 * 7 = 21.
     Schoolbook for 2x2x6 is 2*2*6 = 24.
     So we save 3 ranks here.
     
     Sub-product 2: M2 (2x3) * N2 (3x6) -> P2 (2x6).
     Split N2 into three 3x2 blocks: N2 = [N21, N22, N23].
     P2 = [M2*N21, M2*N22, M2*N23].
     Each term is (2x3) * (3x2) -> 2x2.
     What is the rank of 2x3x2 matrix multiplication?
     Schoolbook: 2*3*2 = 12.
     Can we do better?
     2x3x2 is equivalent to 2x2x2 + 2x2x2?
     M2 = [M2a (2x1), M2b (2x1), M2c (2x1)].
     N21 = [N21a (1x2), N21b (1x2), N21c (1x2)].
     P21 = M2a*N21a + M2b*N21b + M2c*N21c.
     This is a sum of 3 rank-1 tensors?
     The tensor T[i, j, k] for 2x3x2.
     Indices: i in 0..1, j in 0..2, k in 0..1.
     This is a 2x3x2 tensor.
     It is known that rank(2x3x2) = 7? Or 8?
     Actually, 2x3x2 is a sub-tensor of 2x2x2? No.
     
     Let's check the rank of 2x3x2.
     It is the tensor for multiplying a 2x3 matrix by a 3x2 matrix.
     This is equivalent to the 2x2x2 tensor with one dimension expanded?
     No, it's a specific tensor.
     
     Let's try to construct a decomposition for 2x3x2.
     Schoolbook: 12 terms.
     
     If we use Strassen-like ideas?
     Strassen works for 2x2x2.
     2x3x2 can be embedded in 2x4x2?
     
     Let's look up or derive rank(2x3x2).
     The tensor is T_{i,j,k} = 1.
     We want to write 1 = sum_c U_i V_j W_k.
     
     Actually, there is a known result: rank(2xn x n x 2) is related to Strassen.
     For n=1, 2x1x2 is rank 2 (just 2 terms).
     For n=2, 2x2x2 is rank 7.
     For n=3, 2x3x2?
     
     Let's try to find a rank 7 decomposition for 2x3x2?
     If rank(2x3x2) = 7, then:
     Rank(2x2x6) = 21.
     Rank(2x3x6) = 3 * 7 = 21.
     Total Rank = 21 + 21 = 42.
     This would be a huge improvement over 60.
     
     Is rank(2x3x2) = 7?
     Let's test.
     We need to solve for U (2x7), V (3x7), W (2x7).
     2*3*2 = 12 equations. 2*7 + 3*7 + 2*7 = 42 unknowns.
     The system is underdetermined.
     
     Let's try to construct it explicitly.
     We can view 2x3x2 as 2x2x2 + 2x1x2?
     M = [M11 M12 M13]
         [M21 M22 M23]
     N = [N11 N12]
         [N21 N22]
         [N31 N32]
     
     P11 = M11 N11 + M12 N21 + M13 N31
     P12 = M11 N12 + M12 N22 + M13 N32
     P21 = M21 N11 + M22 N21 + M23 N31
     P22 = M21 N12 + M22 N22 + M23 N32
     
     This looks like we are doing 3 multiplications of 2x1 by 1x2, plus...
     Actually, this is just the definition.
     
     Let's use the fact that 2x3x2 is a "slice" of 2x3x3.
     
     Alternative strategy: Just implement the 2x2x6 and 2x3x6 decomposition using the best known ranks for the sub-blocks.
     If I can't guarantee rank(2x3x2) < 12, I will just use schoolbook for the 2x3x2 parts.
     Rank(2x2x6) = 21 (using Strassen on 2x2x2 blocks).
     Rank(2x3x6) using schoolbook for 2x3x2 blocks = 3 * 12 = 36.
     Total = 21 + 36 = 57.
     Still better than 60.
     
     Can we do better on 2x3x2?
     Let's try to find a rank 8 or 7 decomposition for 2x3x2.
     
     Let's assume rank(2x3x2) is 8 for now and try to find it?
     Actually, let's just implement the 57-rank decomposition first. It's a guaranteed improvement.
     
     Implementation details for 2x2x6 (Rank 21):
     Split B (2x6) into B1 (2x2), B2 (2x2), B3 (2x2).
     Use Strassen for A (2x2) * B1, A * B2, A * B3.
     Strassen terms for one 2x2x2:
     m1 = (A11+A22)(B11+B22)
     m2 = (A21+A22)B11
     m3 = A11(B12-B22)
     m4 = A22(B21-B11)
     m5 = (A11+A12)B22
     m6 = (A21-A11)(B11+B12)
     m7 = (A12-A22)(B21+B22)
     
     C11 = m1 + m4 - m5 + m7
     C12 = m3 + m5
     C21 = m2 + m4
     C22 = m1 - m2 + m3 + m6
     
     We need to map these to the tensor factors U, V, W.
     U corresponds to A indices. V to B indices. W to C indices.
     
     For the whole 2x5x6 problem:
     A is 2x5. B is 5x6.
     Split A into A_L (2x2) and A_R (2x3).
     Split B into B_T (2x6) and B_B (3x6).
     
     C = A_L B_T + A_R B_B.
     
     Part 1: A_L (2x2) * B_T (2x6).
     Split B_T into B_T1 (2x2), B_T2 (2x2), B_T3 (2x2).
     Use Strassen for each.
     This gives 3 * 7 = 21 terms.
     
     Part 2: A_R (2x3) * B_B (3x6).
     Split B_B into B_B1 (3x2), B_B2 (3x2), B_B3 (3x2).
     Use Schoolbook for each (2x3x2).
     Rank for one 2x3x2 is 12.
     Total 3 * 12 = 36 terms.
     
     Total Rank = 21 + 36 = 57.
     
     Let's verify the index mapping.
     A indices: i in {0,1}, j in {0..4}.
     B indices: j in {0..4}, k in {0..5}.
     C indices: i in {0,1}, k in {0..5}.
     W index: k*n + i = k*2 + i.
     
     Part 1: A_L uses j=0,1. B_T uses j=0,1.
     B_T1 uses k=0,1.
     B_T2 uses k=2,3.
     B_T3 uses k=4,5.
     
     For B_T1 (2x2):
     A_L is rows 0,1 cols 0,1.
     B_T1 is rows 0,1 cols 0,1.
     Result C1 is rows 0,1 cols 0,1.
     
     Strassen factors for this block:
     U (2x2 part of A) -> U[0:2, c] for the 4 A entries.
     V (2x2 part of B) -> V[0:2, c] for the 4 B entries.
     W (2x2 part of C) -> W[0:2, c] for the 4 C entries.
     
     We need to construct U, V, W matrices of size (10, 57), (30, 57), (12, 57).
     
     Let's define the Strassen coefficients.
     A entries: A00, A01, A10, A11. (Indices 0,1,2,3 in U? No, U is indexed by i*m+j.
     i=0, j=0 -> 0. i=0, j=1 -> 1. i=1, j=0 -> 5. i=1, j=1 -> 6.
     Wait, A is 2x5.
     A[0,0] -> U[0]
     A[0,1] -> U[1]
     A[1,0] -> U[5]
     A[1,1] -> U[6]
     
     B entries: B00, B01, B10, B11.
     B is 5x6.
     B[0,0] -> V[0]
     B[0,1] -> V[1]
     B[1,0] -> V[6]
     B[1,1] -> V[7]
     
     C entries: C00, C01, C10, C11.
     C is 2x6.
     W index = k*n + i.
     C[0,0] -> W[0*2+0] = W[0]
     C[0,1] -> W[1*2+0] = W[2]
     C[1,0] -> W[0*2+1] = W[1]
     C[1,1] -> W[1*2+1] = W[3]
     
     Let's build the 21 terms for Part 1.
     Block 1 (k=0,1):
     m1 = (A00+A11)(B00+B11)
     U: A00:+1, A11:+1. V: B00:+1, B11:+1. W: C00:+1, C11:+1.
     m2 = (A10+A11)B00
     U: A10:+1, A11:+1. V: B00:+1. W: C10:+1, C11:-1.
     m3 = A00(B01-B11)
     U: A00:+1. V: B01:+1, B11:-1. W: C01:+1, C11:+1.
     m4 = A11(B10-B00)
     U: A11:+1. V: B10:+1, B00:-1. W: C10:+1, C11:+1.
     m5 = (A00+A01)B11
     U: A00:+1, A01:+1. V: B11:+1. W: C00:-1, C01:+1.
     m6 = (A10-A00)(B00+B01)
     U: A10:+1, A00:-1. V: B00:+1, B01:+1. W: C11:+1.
     m7 = (A01-A11)(B10+B11)
     U: A01:+1, A11:-1. V: B10:+1, B11:+1. W: C00:+1, C10:-1.
     
     Wait, check Strassen formulas again.
     C11 = m1 + m4 - m5 + m7
     C12 = m3 + m5
     C21 = m2 + m4
     C22 = m1 - m2 + m3 + m6
     
     My W assignments:
     m1: C11:+1, C22:+1. (W[0]:+1, W[3]:+1)
     m2: C21:+1, C22:-1. (W[1]:+1, W[3]:-1)
     m3: C12:+1, C22:+1. (W[2]:+1, W[3]:+1)
     m4: C21:+1, C22:+1. (W[1]:+1, W[3]:+1)
     m5: C11:-1, C12:+1. (W[0]:-1, W[2]:+1)
     m6: C22:+1. (W[3]:+1)
     m7: C11:+1, C21:-1. (W[0]:+1, W[1]:-1)
     
     Let's verify C11 = m1 + m4 - m5 + m7.
     W[0] coeffs: m1(+1) + m4(0) - m5(-1) + m7(+1) = 1 + 1 + 1 = 3?
     Wait. m4 does not affect C11.
     m1: W[0]=1.
     m5: W[0]=-1.
     m7: W[0]=1.
     Sum: 1 - (-1) + 1 = 3? No.
     Formula: C11 = m1 + m4 - m5 + m7.
     Coeff of m1 is 1. Coeff of m5 is -1. Coeff of m7 is 1.
     So in the decomposition, we add the term m1, subtract m5, add m7.
     In the linear combination sum_c U V W, the W entry for C11 is the sum of W_c for that term.
     So for C11, the value is 1*m1 + 0*m2 + 0*m3 + 1*m4 + (-1)*m5 + 0*m6 + 1*m7.
     So W[C11] for term m1 is 1.
     W[C11] for term m5 is -1.
     W[C11] for term m7 is 1.
     Sum = 1 - 1 + 1 = 1. Correct.
     
     Okay, I will code this up.
     Part 2: Schoolbook for 2x3x2 blocks.
     A_R (2x3) uses A[0,2], A[0,3], A[0,4], A[1,2], A[1,3], A[1,4].
     U indices:
     A[0,2] -> 2
     A[0,3] -> 3
     A[0,4] -> 4
     A[1,2] -> 7
     A[1,3] -> 8
     A[1,4] -> 9
     
     B_B1 (3x2) uses B[2,0], B[2,1], B[3,0], B[3,1], B[4,0], B[4,1].
     V indices:
     B[2,0] -> 12
     B[2,1] -> 13
     B[3,0] -> 18
     B[3,1] -> 19
     B[4,0] -> 24
     B[4,1] -> 25
     
     C1 (2x2) for k=0,1.
     W indices: 0,2,1,3.
     
     Schoolbook terms for 2x3x2:
     6 terms for C[0,0], 6 for C[0,1], 6 for C[1,0], 6 for C[1,1].
     Total 24 terms? No, 2*3*2 = 12 terms.
     Each term is U[i,j] V[j,k] W[k,i].
     For C[0,0] (W[0]):
     Sum over j=0..2 of A[0,j] B[j,0].
     j=0: A[0,2] B[2,0]. U[2]=1, V[12]=1, W[0]=1.
     j=1: A[0,3] B[3,0]. U[3]=1, V[18]=1, W[0]=1.
     j=2: A[0,4] B[4,0]. U[4]=1, V[24]=1, W[0]=1.
     
     For C[0,1] (W[2]):
     j=0: A[0,2] B[2,1]. U[2]=1, V[13]=1, W[2]=1.
     j=1: A[0,3] B[3,1]. U[3]=1, V[19]=1, W[2]=1.
     j=2: A[0,4] B[4,1]. U[4]=1, V[25]=1, W[2]=1.
     
     For C[1,0] (W[1]):
     j=0: A[1,2] B[2,0]. U[7]=1, V[12]=1, W[1]=1.
     j=1: A[1,3] B[3,0]. U[8]=1, V[18]=1, W[1]=1.
     j=2: A[1,4] B[4,0]. U[9]=1, V[24]=1, W[1]=1.
     
     For C[1,1] (W[3]):
     j=0: A[1,2] B[2,1]. U[7]=1, V[13]=1, W[3]=1.
     j=1: A[1,3] B[3,1]. U[8]=1, V[19]=1, W[3]=1.
     j=2: A[1,4] B[4,1]. U[9]=1, V[25]=1, W[3]=1.
     
     So 12 terms for the first block.
     Repeat for B_B2 (k=2,3) and B_B3 (k=4,5).
     B_B2:
     B[2,2] -> V[14]
     B[2,3] -> V[15]
     B[3,2] -> V[20]
     B[3,3] -> V[21]
     B[4,2] -> V[26]
     B[4,3] -> V[27]
     C indices: W[4], W[6], W[5], W[7]. (k=2 -> 2*2=4, 4+0=4, 4+1=5. k=3 -> 6, 7).
     
     B_B3:
     B[2,4] -> V[16]
     B[2,5] -> V[17]
     B[3,4] -> V[22]
     B[3,5] -> V[23]
     B[4,4] -> V[28]
     B[4,5] -> V[29]
     C indices: W[8], W[10], W[9], W[11].
     
     Total terms: 21 (Strassen) + 12 + 12 + 12 = 57.
     
     I will construct the matrices U, V, W with 57 columns.
"""

import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Total rank
    rank = 57
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 10 x 57
    V = [[0] * rank for _ in range(m * p)]  # 30 x 57
    W = [[0] * rank for _ in range(p * n)]  # 12 x 57
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Part 1: Strassen for 2x2 x 2x6 (3 blocks of 2x2x2)
    # A_L: i=0,1; j=0,1
    # B_T: j=0,1; k=0..5
    
    # Block 1: k=0,1
    # m1 = (A00+A11)(B00+B11) -> C00, C11
    add_term(0, 1, 0, 1, 0, 1) # A00, B00, C00
    add_term(6, 1, 0, 1, 0, 1) # A11, B00, C00 -- Wait, U[6] is A[1,1]?
    # A indices:
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 5
    # i=1, j=1 -> 6
    # So A00=U[0], A01=U[1], A10=U[5], A11=U[6].
    # B indices:
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 6
    # j=1, k=1 -> 7
    # So B00=V[0], B01=V[1], B10=V[6], B11=V[7].
    # C indices:
    # k=0, i=0 -> 0
    # k=1, i=0 -> 2
    # k=0, i=1 -> 1
    # k=1, i=1 -> 3
    # So C00=W[0], C01=W[2], C10=W[1], C11=W[3].
    
    # Reset col for this block? No, col is global.
    # I need to be careful. The add_term function increments col.
    # I should define the 7 terms explicitly.
    
    # Let's restart the construction logic more clearly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Strassen Block 1 (k=0,1)
    # m1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[6][col] = 1
    V[0][col] = 1; V[7][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # m2 = (A10+A11)B00
    U[5][col] = 1; U[6][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # m3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[7][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # m4 = A11(B10-B00)
    U[6][col] = 1
    V[6][col] = 1; V[0][col] = -1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # m5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    
    # m6 = (A10-A00)(B00+B01)
    U[5][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # m7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[6][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[0][col] = 1; W[1][col] = -1
    col += 1
    
    # Strassen Block 2 (k=2,3)
    # B indices:
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=1, k=2 -> 8
    # j=1, k=3 -> 9
    # C indices:
    # k=2, i=0 -> 4
    # k=3, i=0 -> 6
    # k=2, i=1 -> 5
    # k=3, i=1 -> 7
    
    # m1
    U[0][col] = 1; U[6][col] = 1
    V[2][col] = 1; V[9][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # m2
    U[5][col] = 1; U[6][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1
    
    # m3
    U[0][col] = 1
    V[3][col] = 1; V[9][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # m4
    U[6][col] = 1
    V[8][col] = 1; V[2][col] = -1
    W[5][col] = 1; W[7][col] = 1
    col += 1
    
    # m5
    U[0][col] = 1; U[1][col] = 1
    V[9][col] = 1
    W[4][col] = -1; W[6][col] = 1
    col += 1
    
    # m6
    U[5][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # m7
    U[1][col] = 1; U[6][col] = -1
    V[8][col] = 1; V[9][col] = 1
    W[4][col] = 1; W[5][col] = -1
    col += 1
    
    # Strassen Block 3 (k=4,5)
    # B indices:
    # j=0, k=4 -> 4
    # j=0, k=5 -> 5
    # j=1, k=4 -> 10
    # j=1, k=5 -> 11
    # C indices:
    # k=4, i=0 -> 8
    # k=5, i=0 -> 10
    # k=4, i=1 -> 9
    # k=5, i=1 -> 11
    
    # m1
    U[0][col] = 1; U[6][col] = 1
    V[4][col] = 1; V[11][col] = 1
    W[8][col] = 1; W[11][col] = 1
    col += 1
    
    # m2
    U[5][col] = 1; U[6][col] = 1
    V[4][col] = 1
    W[9][col] = 1; W[11][col] = -1
    col += 1
    
    # m3
    U[0][col] = 1
    V[5][col] = 1; V[11][col] = -1
    W[10][col] = 1; W[11][col] = 1
    col += 1
    
    # m4
    U[6][col] = 1
    V[10][col] = 1; V[4][col] = -1
    W[9][col] = 1; W[11][col] = 1
    col += 1
    
    # m5
    U[0][col] = 1; U[1][col] = 1
    V[11][col] = 1
    W[8][col] = -1; W[10][col] = 1
    col += 1
    
    # m6
    U[5][col] = 1; U[0][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[11][col] = 1
    col += 1
    
    # m7
    U[1][col] = 1; U[6][col] = -1
    V[10][col] = 1; V[11][col] = 1
    W[8][col] = 1; W[9][col] = -1
    col += 1
    
    # Part 2: Schoolbook for 2x3 x 3x6 (3 blocks of 2x3x2)
    # A_R: i=0,1; j=2,3,4
    # U indices:
    # A[0,2]=2, A[0,3]=3, A[0,4]=4
    # A[1,2]=7, A[1,3]=8, A[1,4]=9
    
    # Block 1: k=0,1
    # B_B1: j=2,3,4; k=0,1
    # V indices:
    # B[2,0]=12, B[2,1]=13
    # B[3,0]=18, B[3,1]=19
    # B[4,0]=24, B[4,1]=25
    # C indices: W[0], W[2], W[1], W[3]
    
    # C[0,0] (W[0]):
    U[2][col] = 1; V[12][col] = 1; W[0][col] = 1; col += 1
    U[3][col] = 1; V[18][col] = 1; W[0][col] = 1; col += 1
    U[4][col] = 1; V[24][col] = 1; W[0][col] = 1; col += 1
    
    # C[0,1] (W[2]):
    U[2][col] = 1; V[13][col] = 1; W[2][col] = 1; col += 1
    U[3][col] = 1; V[19][col] = 1; W[2][col] = 1; col += 1
    U[4][col] = 1; V[25][col] = 1; W[2][col] = 1; col += 1
    
    # C[1,0] (W[1]):
    U[7][col] = 1; V[12][col] = 1; W[1][col] = 1; col += 1
    U[8][col] = 1; V[18][col] = 1; W[1][col] = 1; col += 1
    U[9][col] = 1; V[24][col] = 1; W[1][col] = 1; col += 1
    
    # C[1,1] (W[3]):
    U[7][col] = 1; V[13][col] = 1; W[3][col] = 1; col += 1
    U[8][col] = 1; V[19][col] = 1; W[3][col] = 1; col += 1
    U[9][col] = 1; V[25][col] = 1; W[3][col] = 1; col += 1
    
    # Block 2: k=2,3
    # V indices:
    # B[2,2]=14, B[2,3]=15
    # B[3,2]=20, B[3,3]=21
    # B[4,2]=26, B[4,3]=27
    # C indices: W[4], W[6], W[5], W[7]
    
    # C[0,2] (W[4]):
    U[2][col] = 1; V[14][col] = 1; W[4][col] = 1; col += 1
    U[3][col] = 1; V[20][col] = 1; W[4][col] = 1; col += 1
    U[4][col] = 1; V[26][col] = 1; W[4][col] = 1; col += 1
    
    # C[0,3] (W[6]):
    U[2][col] = 1; V[15][col] = 1; W[6][col] = 1; col += 1
    U[3][col] = 1; V[21][col] = 1; W[6][col] = 1; col += 1
    U[4][col] = 1; V[27][col] = 1; W[6][col] = 1; col += 1
    
    # C[1,2] (W[5]):
    U[7][col] = 1; V[14][col] = 1; W[5][col] = 1; col += 1
    U[8][col] = 1; V[20][col] = 1; W[5][col] = 1; col += 1
    U[9][col] = 1; V[26][col] = 1; W[5][col] = 1; col += 1
    
    # C[1,3] (W[7]):
    U[7][col] = 1; V[15][col] = 1; W[7][col] = 1; col += 1
    U[8][col] = 1; V[21][col] = 1; W[7][col] = 1; col += 1
    U[9][col] = 1; V[27][col] = 1; W[7][col] = 1; col += 1
    
    # Block 3: k=4,5
    # V indices:
    # B[2,4]=16, B[2,5]=17
    # B[3,4]=22, B[3,5]=23
    # B[4,4]=28, B[4,5]=29
    # C indices: W[8], W[10], W[9], W[11]
    
    # C[0,4] (W[8]):
    U[2][col] = 1; V[16][col] = 1; W[8][col] = 1; col += 1
    U[3][col] = 1; V[22][col] = 1; W[8][col] = 1; col += 1
    U[4][col] = 1; V[28][col] = 1; W[8][col] = 1; col += 1
    
    # C[0,5] (W[10]):
    U[2][col] = 1; V[17][col] = 1; W[10][col] = 1; col += 1
    U[3][col] = 1; V[23][col] = 1; W[10][col] = 1; col += 1
    U[4][col] = 1; V[29][col] = 1; W[10][col] = 1; col += 1
    
    # C[1,4] (W[9]):
    U[7][col] = 1; V[16][col] = 1; W[9][col] = 1; col += 1
    U[8][col] = 1; V[22][col] = 1; W[9][col] = 1; col += 1
    U[9][col] = 1; V[28][col] = 1; W[9][col] = 1; col += 1
    
    # C[1,5] (W[11]):
    U[7][col] = 1; V[17][col] = 1; W[11][col] = 1; col += 1
    U[8][col] = 1; V[23][col] = 1; W[11][col] = 1; col += 1
    U[9][col] = 1; V[29][col] = 1; W[11][col] = 1; col += 1
    
    if col != rank:
        # Fallback to schoolbook if construction failed (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                max_err = max(max_err, abs(s - 1))
    print(f"Rank: {rank}, Max Error: {max_err}")

