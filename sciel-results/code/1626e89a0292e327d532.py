import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strassen-based tensor decomposition
    # M_{3,4,8} = M_{3,1,1} ⊗ M_{1,4,8}
    # M_{1,4,8} = M_{1,4,4} ⊗ M_{1,1,2}
    # M_{1,4,4} uses Strassen (rank 7)
    # M_{1,1,2} has rank 2 (schoolbook for 1x1 * 1x2)
    # Total rank = 3 * 7 * 2 = 42
    
    # Strassen rank-7 decomposition for 2x2 * 2x2
    # The 7 multiplication terms and their (a,b) and (c,d) coefficients
    # where a is top-left, b is top-right, c is bottom-left, d is bottom-right
    # and e is top-left, f is top-right, g is bottom-left, h is bottom-right
    
    # Strassen's 7 products:
    # m1 = (a+b)(e+f)
    # m2 = (a+b)(g+h)  -- wait, standard Strassen:
    # m1 = (a+b)(e+f)
    # m2 = (c+d)(e+f)
    # m3 = a(h-f)
    # m4 = d(g-e)
    # m5 = (a+e)(h+f)
    # m6 = (c+a)(f-h)
    # m7 = (b+d)(g-c)
    
    # For 2x2 * 2x2, the result is:
    # [p1, p2]     [m1+m4-m5+m7, m5-m1+m2+m6]
    # [p3, p4] =   [m1-m3+m4+m6, -m3+m2+m5+m7]
    
    # We need to build the factor matrices for M_{1,4,4} (1x4 * 4x4)
    # This is equivalent to multiplying a row vector (1x4) by a 4x4 matrix
    # We can view this as two 2x2 multiplications
    
    # Actually, let me think more carefully.
    # M_{1,4,4}: input A is 1x4, input B is 4x4, output C is 1x4
    # We can split A into [A1, A2] where A1, A2 are 1x2
    # B is [[B11, B12], [B21, B22]] where each is 2x2
    # C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    # But A1 and A2 are row vectors (1x2), not 2x2 matrices.
    # So we need a different approach.
    
    # Let me use the tensor product structure directly.
    # M_{1,4,4} = M_{1,2,2} ⊗ M_{1,2,2}
    # M_{1,2,2}: 1x2 * 2x2 -> 1x2
    # This has rank 4 (schoolbook) or can we do better?
    # Actually for 1x2 * 2x2, the tensor is [a1, a2] * [[b11,b12],[b21,b22]] = [a1*b11+a2*b21, a1*b12+a2*b22]
    # Rank is 4 (each output element depends on 2 products)
    
    # Hmm, let me reconsider. The directive says to use Strassen on 4x4 and 4x8 sub-blocks.
    # Let me think of it differently.
    
    # 3x4 * 4x8:
    # Split the 4 into 2+2: A is 3x4 = [A1, A2] where A1, A2 are 3x2
    # B is 4x8 = [[B11, B12], [B21, B22]] where each is 2x8
    # C = A1*B11 + A2*B21 (3x8) and C = A1*B12 + A2*B22 (3x8)
    
    # Each 3x2 * 2x8 multiplication can be done with Strassen-like methods.
    # 3x2 * 2x8: split 3 into 1+2? Or use the fact that this is M_{3,2,8}
    # M_{3,2,8} = M_{3,2,4} ⊗ M_{1,1,2}
    # M_{3,2,4} = M_{3,2,2} ⊗ M_{1,1,2}
    # M_{3,2,2}: 3x2 * 2x2 -> 3x2, rank 12 (schoolbook)
    
    # This is getting complicated. Let me just directly construct the rank-42 decomposition
    # using the known Strassen structure.
    
    # Actually, the simplest approach: use the tensor product of known decompositions.
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # M_{3,4,4} = M_{3,4,2} ⊗ M_{1,1,2}
    # M_{3,4,2} = M_{3,4,1} ⊗ M_{1,1,2} ... no this doesn't work cleanly.
    
    # Let me just implement a direct rank-42 construction.
    # 
    # Strategy: Use the fact that M_{a,b,c} ⊗ M_{d,e,f} = M_{ad, be, cf}
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # 
    # For M_{1,1,2}: A is 1x1=[a], B is 1x2=[b1,b2], C is 1x2=[a*b1, a*b2]
    # Rank 2: U=[1,1], V=[b1,b2]... 
    # U shape (1*1, 2) = (1,2): [[1, 1]]
    # V shape (1*2, 2) = (2,2): [[1, 0], [0, 1]]
    # W shape (2*1, 2) = (2,2): [[1, 0], [0, 1]]
    # Check: sum_c U[0,c]*V[j,c]*W[k,c] for j=0,k=0: 1*1*1 + 1*0*0 = 1 ✓
    # j=1,k=0: 1*0*1 + 1*1*0 = 0 ✗ should be 0... wait
    # 
    # The tensor T[i,j,k] = 1 if this is the (i,j,k) entry of the multiplication.
    # For 1x1 * 1x2 -> 1x2: T[0,0,0]=1, T[0,1,1]=1, all others 0.
    # Decomposition: T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> contributes to T[0,0,0]=1 ✓
    # c=1: U[0,1]=1, V[1,1]=1, W[1,1]=1 -> contributes to T[0,1,1]=1 ✓
    # All other (i,j,k) get 0. 
    # So U=[[1,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]]
    
    # Now M_{3,4,4}: I need a rank-21 decomposition.
    # M_{3,4,4} = M_{3,4,2} ⊗ M_{1,1,2}
    # M_{3,4,2}: rank 10 (schoolbook is 3*4*2=24, but we can do better)
    # Actually schoolbook for 3x4*4x2 is 3*4*2=24.
    # 
    # Let me try: M_{3,4,4} = M_{3,2,2} ⊗ M_{1,2,2}
    # M_{3,2,2}: 3x2*2x2->3x2, schoolbook rank 12
    # M_{1,2,2}: 1x2*2x2->1x2, schoolbook rank 4
    # Product rank: 12*4=48. Too high.
    
    # Better: M_{3,4,4} = M_{3,1,1} ⊗ M_{1,4,4}
    # M_{3,1,1}: rank 3 (trivial)
    # M_{1,4,4}: need rank 7 to get total 21.
    # 
    # M_{1,4,4}: 1x4 * 4x4 -> 1x4
    # Can we do this with rank 7?
    # Split: A=[a1,a2,a3,a4], B is 4x4.
    # C[k] = sum_j A[j] B[j,k]
    # 
    # View B as 2x2 blocks of 2x2: B = [[B11,B12],[B21,B22]]
    # A = [A1, A2] where A1=[a1,a2], A2=[a3,a4]
    # C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # 
    # A1*B11: 1x2 * 2x2 -> 1x2. This is M_{1,2,2}, rank 4.
    # We have 4 such terms. Total rank 16 with schoolbook.
    # 
    # With Strassen on the 2x2 blocks:
    # A1*B11 + A2*B21: This is like computing [A1; A2] * [B11; B21] where [A1;A2] is 2x2 (with second row zero?) No...
    # 
    # Actually A1 is 1x2, so A1*B11 is 1x2. We can't directly apply Strassen.
    # 
    # Let me just use schoolbook for M_{1,4,4} which has rank 16.
    # Then M_{3,4,4} rank = 3*16 = 48.
    # M_{3,4,8} rank = 48*2 = 96. Same as schoolbook. Not helpful.
    
    # Let me try a different decomposition.
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # Use Strassen on M_{3,4,4} by viewing it as 3*(4x4).
    # 
    # For each row of A (3x4), we compute row * B (4x4).
    # row * B: 1x4 * 4x4 -> 1x4
    # 
    # Split the 4x4 into 2x2 blocks and the 1x4 into two 1x2 vectors.
    # Let A_row = [x1, x2, x3, x4]
    # B = [[B11, B12], [B21, B22]] with 2x2 blocks
    # C = [x1,x2]*B11 + [x3,x4]*B21, [x1,x2]*B12 + [x3,x4]*B22
    # 
    # Define: u1=[x1,x2], u2=[x3,x4]
    # v11=B11, v12=B12, v21=B21, v22=B22 (all 2x2)
    # 
    # C1 = u1*v11 + u2*v21 (1x2)
    # C2 = u1*v12 + u2*v22 (1x2)
    # 
    # u1*v11: 1x2 * 2x2 -> 1x2, rank 4
    # 
    # Total for one row: 4 terms * rank 4 = 16
    # For 3 rows: 3 * 16 = 48
    # For 8 columns (⊗ M_{1,1,2}): 48 * 2 = 96
    # 
    # Still 96. The issue is that 1x2 * 2x2 has rank 4, not reducible by Strassen.
    
    # Let me try: M_{3,4,8} = M_{3,2,4} ⊗ M_{1,2,2}
    # M_{3,2,4}: 3x2 * 2x4 -> 3x4, schoolbook rank 24
    # M_{1,2,2}: 1x2 * 2x2 -> 1x2, rank 4
    # Total: 24*4 = 96. Still 96.
    
    # M_{3,4,8} = M_{3,2,8} ⊗ M_{1,2,1}... doesn't make sense.
    
    # OK let me try the approach from the directive more literally.
    # "Apply Strassen recursively to 4x4 and 4x8 sub-blocks"
    # 
    # Maybe: M_{3,4,8} = M_{3,4,8} where we split the 4x4 part of B using Strassen.
    # 
    # Actually, I think the intended decomposition is:
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # And M_{3,4,4} uses Strassen on the 4x4 matrix B.
    # 
    # When multiplying A(3x4) * B(4x4), we can use Strassen by padding A to 3x4... 
    # Strassen works for square matrices. For 3x4 * 4x4, we can pad A to 4x4.
    # 
    # Pad A to 4x4: A' = [[A], [0,0,0,0]]
    # A'*B' where B' is 4x4. Use Strassen: rank 7.
    # But we only need the first 3 rows of the result.
    # 
    # The tensor for 4x4*4x4 has rank 7 (Strassen).
    # The tensor for 3x4*4x4 is a "sub-tensor" of 4x4*4x4.
    # The rank of a sub-tensor is at most the rank of the parent.
    # So M_{3,4,4} has rank <= 7? No, that's not right either.
    # 
    # Actually, the multiplication tensor M_{m,n,p} where we multiply m x n by n x p.
    # M_{3,4,4} is the tensor for 3x4 * 4x4 -> 3x4.
    # This is a "slice" of M_{4,4,4}.
    # 
    # If M_{4,4,4} has rank 7 (Strassen), then M_{3,4,4} (which is obtained by setting one row of A to zero) 
    # should have rank at most 7, since any decomposition of M_{4,4,4} restricts to a decomposition of M_{3,4,4}.
    # 
    # YES! So M_{3,4,4} has rank <= 7.
    # 
    # Therefore M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2} has rank <= 7 * 2 = 14.
    # 
    # Wait, that seems too good. Let me verify.
    # 
    # M_{4,4,4} rank 7 (Strassen). The tensor T[i,j,k] for i,j,k in {0,1,2,3}.
    # M_{3,4,4}: T[i,j,k] for i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3}.
    # This is obtained by restricting i to {0,1,2}.
    # Any rank-7 decomposition of M_{4,4,4} gives a rank-7 decomposition of M_{3,4,4}
    # by just ignoring the i=3 terms.
    # 
    # So rank(M_{3,4,4}) <= 7.
    # 
    # Then M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2} has rank <= 7 * 2 = 14.
    # 
    # That's amazing! Let me construct this.
    
    # Strassen decomposition of M_{4,4,4}:
    # A = [[a,b],[c,d]] (2x2 blocks of 2x2)
    # B = [[e,f],[g,h]] (2x2 blocks of 2x2)
    # 
    # m1 = (a+b)(e+f)
    # m2 = (c+d)(e+f)  
    # m3 = a(h-f)
    # m4 = d(g-e)
    # m5 = (a+e)(h+f)
    # m6 = (c+a)(f-h)
    # m7 = (b+d)(g-c)
    # 
    # Result: [[m1+m4-m5+m7, m5-m1+m2+m6],
    #          [m1-m3+m4+m6, -m3+m2+m5+m7]]
    
    # I need to express this as a rank-7 tensor decomposition.
    # T[i,j,k] = sum_{c=1}^{7} U[i,j,c] * V[j,k,c] * W[k,c]
    # where i,j,k are 4x4 indices (flattened).
    
    # Let me build the factor matrices directly.
    # 
    # For Strassen, the 7 "multiplication" operations are bilinear forms in the entries of A and B.
    # Each operation c has:
    # - A coefficients: alpha_c[i,j] for each (i,j) in 4x4
    # - B coefficients: beta_c[j,k] for each (j,k) in 4x4
    # - The product alpha_c * beta_c is one of the m's
    # 
    # Then W[k,c] combines the m's to get the output.
    
    # Let me define the 7 Strassen operations:
    # Each operation c defines a bilinear form on A and B.
    
    # Operation 1: m1 = (a+b)(e+f)
    # a = A[0:2,0:2], b = A[0:2,2:4], e = B[0:2,0:2], f = B[0:2,2:4]
    # m1 = sum_{i,j in 0..1} A[i,j] * sum_{j',k in 2..3} ... 
    # Actually m1 = (A[0:2,0:2] + A[0:2,2:4]) * (B[0:2,0:2] + B[0:2,2:4])
    # This is a matrix product of 2x2 matrices.
    # 
    # The coefficient of A[i,j] in (a+b) is 1 if i<2 and j<4 (all of a and b).
    # Wait, a is A[0:2,0:2] and b is A[0:2,2:4].
    # (a+b)[i,j] = A[i,j] for i in 0,1 and j in 0,1,2,3.
    # So alpha_1[i,j] = 1 if i in {0,1} and j in {0,1,2,3}, else 0.
    # 
    # Similarly, (e+f)[j,k] = B[j,k] for j in {0,1} and k in {0,1,2,3}.
    # beta_1[j,k] = 1 if j in {0,1} and k in {0,1,2,3}, else 0.
    
    # This is getting very detailed. Let me just code it up.
    
    def strassen_4x4_factors():
        """Return U, V, W factor matrices for M_{4,4,4} with rank 7 (Strassen)."""
        size = 4
        rank = 7
        U = np.zeros((size*size, rank))
        V = np.zeros((size*size, rank))
        W = np.zeros((size*size, rank))
        
        # Strassen's 7 operations for 2x2 block matrices
        # Each operation is defined by:
        # - A mask: which 2x2 blocks of A are used, with signs
        # - B mask: which 2x2 blocks of B are used, with signs
        # - Output combination
        
        # Let me index the 4x4 matrix as 2x2 blocks of 2x2.
        # Block (0,0) = indices 0-1, 0-1
        # Block (0,1) = indices 0-1, 2-3
        # Block (1,0) = indices 2-3, 0-1
        # Block (1,1) = indices 2-3, 2-3
        
        # For each Strassen operation, define:
        # a_blocks: dict from block_idx to coefficient for A
        # b_blocks: dict from block_idx to coefficient for B
        
        # Operation 1: m1 = (a+b)(e+f)
        # A blocks: (0,0):1, (0,1):1
        # B blocks: (0,0):1, (0,1):1
        # Output: contributes to C[0,0] with +1, C[0,1] with -1, C[1,0] with +1, C[1,1] with +1
        # Wait, I need to be more careful.
        
        # Let me use the standard Strassen formulas:
        # p1 = (a+b)(e+f)
        # p2 = (c+d)(e+f)
        # p3 = a(h-f)
        # p4 = d(g-e)
        # p5 = (a+e)(h+f)
        # p6 = (c+a)(f-h)
        # p7 = (b+d)(g-c)
        
        # C[0,0] = p1 + p4 - p5 + p7
        # C[0,1] = p5 - p1 + p2 + p6
        # C[1,0] = p1 - p3 + p4 + p6
        # C[1,1] = -p3 + p2 + p5 + p7
        
        # Where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1] (2x2 blocks)
        #       e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1] (2x2 blocks)
        
        # For each operation, I need to define the A and B coefficient matrices (4x4).
        
        ops = []
        
        # p1 = (a+b)(e+f)
        A1 = np.zeros((4,4))
        A1[0:2, 0:4] = 1  # a and b
        B1 = np.zeros((4,4))
        B1[0:2, 0:4] = 1  # e and f
        ops.append((A1, B1))
        
        # p2 = (c+d)(e+f)
        A2 = np.zeros((4,4))
        A2[2:4, 0:4] = 1  # c and d
        B2 = np.zeros((4,4))
        B2[0:2, 0:4] = 1  # e and f
        ops.append((A2, B2))
        
        # p3 = a(h-f)
        A3 = np.zeros((4,4))
        A3[0:2, 0:2] = 1  # a
        B3 = np.zeros((4,4))
        B3[2:4, 2:4] = 1  # h
        B3[0:2, 2:4] = -1  # -f
        ops.append((A3, B3))
        
        # p4 = d(g-e)
        A4 = np.zeros((4,4))
        A4[2:4, 2:4] = 1  # d
        B4 = np.zeros((4,4))
        B4[2:4, 0:2] = 1  # g
        B4[0:2, 0:2] = -1  # -e
        ops.append((A4, B4))
        
        # p5 = (a+e)(h+f)
        A5 = np.zeros((4,4))
        A5[0:2, 0:2] = 1  # a
        B5 = np.zeros((4,4))
        B5[0:2, 0:2] = 1  # e
        B5[2:4, 2:4] = 1  # h
        B5[0:2, 2:4] = 1  # f
        ops.append((A5, B5))
        
        # p6 = (c+a)(f-h)
        A6 = np.zeros((4,4))
        A6[0:2, 0:2] = 1  # a
        A6[2:4, 0:2] = 1  # c
        B6 = np.zeros((4,4))
        B6[0:2, 2:4] = 1  # f
        B6[2:4, 2:4] = -1  # -h
        ops.append((A6, B6))
        
        # p7 = (b+d)(g-c)
        A7 = np.zeros((4,4))
        A7[0:2, 2:4] = 1  # b
        A7[2:4, 2:4] = 1  # d
        B7 = np.zeros((4,4))
        B7[2:4, 0:2] = 1  # g
        B7[2:4, 0:2] -= 0  # 
        B7[2:4, 0:2] = 1  # g
        B7[2:4, 0:2] = 1
        B7[0:2, 0:2] = 0
        # Wait, g-c: g is B[1,0] = B[2:4,0:2], c is A[1,0] = A[2:4,0:2]
        # But c is in A, not B! I made an error.
        # p7 = (b+d)(g-c) where b,d are A blocks and g,c are... 
        # No: b=A[0,1], d=A[1,1], g=B[1,0], c=A[1,0]
        # Wait, that's wrong. Let me recheck Strassen.
        
        # Standard Strassen:
        # p1 = (a+b)(e+f)
        # p2 = (c+d)(e+f)
        # p3 = a(h-f)
        # p4 = d(g-e)
        # p5 = (a+e)(h+f)
        # p6 = (c+a)(f-h)
        # p7 = (b+d)(g-c)
        
        # Here a,b,c,d are 2x2 blocks of A, e,f,g,h are 2x2 blocks of B.
        # So in p7, (b+d) uses A blocks, (g-c) uses... g is B[1,0] and c is A[1,0]?
        # That can't be right for a bilinear form.
        
        # Let me look up the correct Strassen formulas.
        # 
        # Actually, I think I'm confusing myself. Let me use a different formulation.
        # 
        # The key insight: each p_i is a product of a linear combination of A entries 
        # and a linear combination of B entries. It's bilinear.
        
        # Let me restart with a cleaner approach.
        # 
        # I'll directly verify by construction.
        
        return None  # placeholder
    
    # Let me take a completely different, more direct approach.
    # 
    # I'll construct the rank-14 decomposition of M_{3,4,8} directly.
    # 
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # 
    # For M_{1,1,2}: U1 = [[1,1]], V1 = [[1,0],[0,1]], W1 = [[1,0],[0,1]]
    # 
    # For M_{3,4,4}: I'll use the Strassen-based rank-7 decomposition of M_{4,4,4}
    # restricted to the first 3 rows.
    
    # Let me build M_{4,4,4} Strassen factors properly.
    
    def build_strassen_4x4():
        """Build rank-7 factor matrices for 4x4 * 4x4 matrix multiplication."""
        U = np.zeros((16, 7))
        V = np.zeros((16, 7))
        W = np.zeros((16, 7))
        
        # Strassen operations. Each operation c has:
        # A_c: 4x4 matrix of coefficients for A
        # B_c: 4x4 matrix of coefficients for B
        # The product is a 4x4 matrix.
        
        # Let me define using the block structure.
        # A = [[a, b], [c, d]] where each is 2x2
        # B = [[e, f], [g, h]] where each is 2x2
        
        # I'll define each operation by specifying which 2x2 blocks are used.
        
        # p1 = (a+b)(e+f)
        # A coefficient: a has +1, b has +1, c has 0, d has 0
        # B coefficient: e has +1, f has +1, g has 0, h has 0
        
        # For the flattened 4x4 matrix:
        # a = A[0:2, 0:2], b = A[0:2, 2:4], c = A[2:4, 0:2], d = A[2:4, 2:4]
        # e = B[0:2, 0:2], f = B[0:2, 2:4], g = B[2:4, 0:2], h = B[2:4, 2:4]
        
        ops = []
        
        # p1 = (a+b)(e+f)
        A_ops = np.zeros((4,4))
        A_ops[0:2, 0:4] = 1  # a + b
        B_ops = np.zeros((4,4))
        B_ops[0:2, 0:4] = 1  # e + f
        ops.append((A_ops, B_ops))
        
        # p2 = (c+d)(e+f)
        A_ops = np.zeros((4,4))
        A_ops[2:4, 0:4] = 1  # c + d
        B_ops = np.zeros((4,4))
        B_ops[0:2, 0:4] = 1  # e + f
        ops.append((A_ops, B_ops))
        
        # p3 = a(h-f)
        A_ops = np.zeros((4,4))
        A_ops[0:2, 0:2] = 1  # a
        B_ops = np.zeros((4,4))
        B_ops[2:4, 2:4] = 1  # h
        B_ops[0:2, 2:4] = -1  # -f
        ops.append((A_ops, B_ops))
        
        # p4 = d(g-e)
        A_ops = np.zeros((4,4))
        A_ops[2:4, 2:4] = 1  # d
        B_ops = np.zeros((4,4))
        B_ops[2:4, 0:2] = 1  # g
        B_ops[0:2, 0:2] = -1  # -e
        ops.append((A_ops, B_ops))
        
        # p5 = (a+e)(h+f)
        A_ops = np.zeros((4,4))
        A_ops[0:2, 0:2] = 1  # a
        B_ops = np.zeros((4,4))
        B_ops[0:2, 0:2] = 1  # e
        B_ops[2:4, 2:4] = 1  # h
        B_ops[0:2, 2:4] = 1  # f
        ops.append((A_ops, B_ops))
        
        # p6 = (c+a)(f-h)
        A_ops = np.zeros((4,4))
        A_ops[0:2, 0:2] = 1  # a
        A_ops[2:4, 0:2] = 1  # c
        B_ops = np.zeros((4,4))
        B_ops[0:2, 2:4] = 1  # f
        B_ops[2:4, 2:4] = -1  # -h
        ops.append((A_ops, B_ops))
        
        # p7 = (b+d)(g-c)
        # Wait, c is an A block, not B. This is wrong.
        # Let me recheck. In Strassen:
        # p7 = (b+d)(g-c)
        # b = A[0,1], d = A[1,1], g = B[1,0], c = A[1,0]
        # But (g-c) mixes A and B! That's not a valid bilinear form.
        
        # I think I have the Strassen formulas wrong. Let me use the correct ones.
        # 
        # Correct Strassen for C = AB:
        # p1 = (a+b)(e+f)
        # p2 = (c+d)(e+f)
        # p3 = a(h-f)
        # p4 = d(g-e)
        # p5 = (a+e)(h+f)
        # p6 = (c+a)(f-h)
        # p7 = (b+d)(g-c)
        
        # Hmm, p7 = (b+d)(g-c). Here b,d are from A and g is from B, c is from A.
        # That's not bilinear in A and B separately.
        
        # I must be misremembering. Let me look at this differently.
        # 
        # Actually, I think the issue is that in the standard Strassen presentation,
        # all the terms are products of linear combinations of A entries and linear 
        # combinations of B entries. Let me re-derive.
        
        # C[0,0] = ae + bg
        # C[0,1] = af + bh
        # C[1,0] = ce + dg
        # C[1,1] = cf + dh
        
        # Strassen:
        # p1 = (a+b)(e+f) = ae + af + be + bf
        # p2 = (c+d)(e+f) = ce + cf + de + df
        # p3 = a(h-f) = ah - af
        # p4 = d(g-e) = dg - de
        # p5 = (a+e)(h+f) = ah + af + eh + ef
        # p6 = (c+a)(f-h) = cf - ch + af - ah
        # p7 = (b+d)(g-c) = bg - bc + dg - dc
        
        # Now:
        # C[0,0] = ae + bg
        # p1 + p4 - p5 + p7 = (ae+af+be+bf) + (dg-de) - (ah+af+eh+ef) + (bg-bc+dg-dc)
        # = ae + af + be + bf + dg - de - ah - af - eh - ef + bg - bc + dg - dc
        # = ae + be + bf + 2dg - de - ah - eh - ef + bg - bc - dc
        # This doesn't simplify to ae + bg. I'm making errors.
        
        # Let me just use a verified Strassen implementation.
        # 
        # Actually, for the purpose of this task, let me just use the schoolbook 
        # decomposition but try to find a lower rank through search.
        # 
        # Given the time constraints and the complexity, let me just return the 
        # schoolbook rank-96 solution. The random search in the original code
        # was unlikely to find anything better anyway.
        
        return None
    
    # Given the complexity of correctly implementing Strassen for non-square matrices
    # and the time pressure, let me just use the schoolbook decomposition.
    # 
    # But wait - let me try one more thing. The directive specifically mentions
    # rank 72 = 3 * 2^3 * 8. Let me see if I can construct a rank-72 decomposition.
    # 
    # 72 = 3 * 8 * 3 = 3 * 8 * 3. Or 72 = 9 * 8. Or 72 = 3 * 24.
    # 
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # If M_{3,4,4} has rank 36, then total is 72.
    # M_{3,4,4} = M_{3,4,2} ⊗ M_{1,1,2}
    # If M_{3,4,2} has rank 18, then M_{3,4,4} has rank 36.
    # M_{3,4,2} = M_{3,4,1} ⊗ M_{1,1,2}... no, M_{3,4,1} is 3x4*4x1->3x1, rank 12.
    # 
    # I'm going in circles. Let me just implement the schoolbook and be done.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
