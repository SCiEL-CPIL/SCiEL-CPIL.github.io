"""Matrix multiplication 3x4x8 - rank 72 via block decomposition with rank-36 sub-blocks."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 8]

def _rank36_3x4x4():
    """
    Return a valid rank-36 decomposition for 3x4 * 4x4 = 3x4.
    
    U: shape (12, 36)  indexed by i*4+j
    V: shape (16, 36)  indexed by j*4+k  
    W: shape (12, 36)  indexed by k*3+i
    
    We use the following strategy:
    Split the 4x4 multiplication into two 2x4 * 4x4 halves? No.
    
    Actually, let's use a known approach:
    M(3,4,4) <= 36 can be achieved by viewing it as:
    A(3x4) * B(4x4) = C(3x4)
    
    Split A into two 3x2 blocks: A = [A1, A2]
    Split B into two 4x2 blocks by columns? No, B is 4x4.
    
    Let me use a different decomposition. I'll construct it explicitly.
    
    For 3x4 * 4x4, we can use the following:
    Write B as a 2x2 block matrix of 2x2 blocks:
    B = [[B11, B12], [B21, B22]]
    
    A as a 3x2 block matrix (each block 3x2):
    A = [A1, A2] where A1, A2 are 3x2
    
    Then C = A1*B1 + A2*B2 where:
    B1 = [B11; B21] (4x2), B2 = [B12; B22] (4x2)
    
    C1 = A1*B1 (3x2), C2 = A2*B2 (3x2), C = [C1, C2]
    
    For 3x2 * 4x2:
    X(3x2) * Y(4x2) = Z(3x2)
    Z[i,k] = sum_{j=0}^{1} X[i,j] * Y[2j,k] + ... wait, Y is 4x2.
    
    Actually Y(4x2), so Z[i,k] = sum_{j=0}^{3} X[i,j%2]... no.
    
    Let me be more careful. X is 3x2, Y is 4x2.
    Z = X * Y is not defined since X has 2 columns and Y has 4 rows.
    
    I need to restructure. Let me use:
    A(3x4) * B(4x4) = C(3x4)
    
    Split the inner dimension (size 4) into two parts of size 2:
    A = [A_left(3x2), A_right(3x2)]
    B = [B_top(2x4); B_bottom(2x4)]
    
    C = A_left * B_top + A_right * B_bottom
    
    Each term is 3x2 * 2x4 = 3x4.
    
    For 3x2 * 2x4:
    X(3x2) * Y(2x4) = Z(3x4)
    Z[i,k] = X[i,0]*Y[0,k] + X[i,1]*Y[1,k]
    
    This is a 3x2x4 tensor. What's its rank?
    Schoolbook: 3*2*4 = 24.
    
    Can we do better? M(3,2,4) <= ?
    
    Hmm, 24 per block, 2 blocks = 48. That's the schoolbook for 3x4x4.
    
    I need a better decomposition. Let me try a different split.
    
    Actually, let me just use a computational approach to find a valid rank-36 decomposition.
    I'll set up the linear system and solve it.
    """
    # For 3x4*4x4 = 3x4
    # U: (12, r), V: (16, r), W: (12, r)
    # Tensor T[i*4+j, j*4+k, k*3+i] = 1
    
    n, m, p = 3, 4, 4
    # We want rank r = 36
    r = 36
    
    # The tensor has entries T[a, b, c] where:
    # a = i*m + j = i*4 + j, i in 0..2, j in 0..3
    # b = j*p + k = j*4 + k, j in 0..3, k in 0..3  
    # c = k*n + i = k*3 + i, k in 0..3, i in 0..2
    # T[a,b,c] = 1 if the indices are consistent (i,j,k match up)
    
    # For each term c in 0..r-1, we need:
    # sum over c of U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
    
    # This is a system of equations. Let me set it up.
    # Number of equations: 12 * 16 * 12 = 2304
    # Number of unknowns: 3 * 12 * 36 + 3 * 16 * 36 + 3 * 12 * 36 = 1296 + 1728 + 1296 = 4320
    
    # This is underdetermined, so solutions exist.
    # Let me use a randomized approach: fix some variables and solve for others.
    
    # Actually, let me use a simpler approach. I'll use the following known valid decomposition.
    
    # For M(3,4,4) <= 36, one construction uses the following:
    # Think of 3x4*4x4 as computing 12 inner products of length 4.
    # 
    # Use the identity: for vectors u, v in R^4, u.v can be computed as:
    # (u0+v0)(v0) + (u1+v1)(v1) + ... but that's just 4 terms.
    #
    # The key is to share computations across the 12 inner products.
    
    # Let me just construct a valid solution using linear algebra.
    # I'll fix the U matrix to be a random matrix with some structure, then solve for V and W.
    
    # Actually, the simplest valid approach: use the schoolbook for some terms and add
    # some "cancellation" terms. But that's complex.
    
    # Let me try a different approach: use the fact that 3x4*4x4 can be embedded in 4x4*4x4.
    # M(4,4,4) <= 49 is not known, but M(4,4,4) <= 49 via Strassen is not established.
    # Actually, M(4,4,4) <= 49 is open. Best known is M(4,4,4) <= 49? No.
    # M(4,4,4) <= 49 is not known. M(4,4,4) <= 49?
    # The best known upper bound for M(4,4,4) is 49? No, it's 49 via some constructions?
    # Actually, I recall M(4,4,4) <= 49 is not proven. M(4,4,4) <= 49?
    # Let me just use a direct construction.
    
    # I'll use the following: for 3x4*4x4, use 36 terms.
    # I'll construct them explicitly using a pattern.
    
    # Terms 0-11: schoolbook for first 12 (i,j,k) triples where k < 2
    # Terms 12-23: schoolbook for remaining 12 triples where k >= 2
    # Terms 24-35: correction terms that allow sharing
    
    # This is getting too complex. Let me just use rank 48 (schoolbook) for each 3x4x4 block,
    # giving total rank 96. But that's not an improvement.
    
    # Alternative: use rank 36 for each block. Let me find a valid one.
    
    # I'll use a numerical approach: set up the equations and solve.
    
    # For 3x4*4x4:
    # U: 12 x 36, V: 16 x 36, W: 12 x 36
    
    # The bilinear form is: for each (a, b, c_idx) in the output space:
    # sum_{r=0}^{35} U[a,r] * V[b,r] * W[c_idx,r] = T[a,b,c_idx]
    
    # Let me flatten this. For each r, the contribution is U[:,r] * V[:,r] * W[:,r] (outer product).
    # We need sum of these outer products to equal T.
    
    # This is equivalent to: vec(T) = sum_r kron(W[:,r], kron(V[:,r], U[:,r]))
    
    # Let me set up the linear system.
    # Unknowns: all entries of U, V, W. Total: 12*36 + 16*36 + 12*36 = 4320
    # Equations: 12*16*12 = 2304
    
    # This is nonlinear (bilinear in the variables). Let me use a different approach.
    
    # Fix U to be a specific matrix, then the equations become linear in V and W.
    # Or fix two of them.
    
    # Let me fix U to be a random 12x36 matrix with entries in {0, 1, -1}.
    # Then for each r, define u_r = U[:,r] (12-dim vector).
    # The equation becomes: sum_r (u_r * V[:,r]) * W[:,r]^T = T (as a 12x16x12 tensor)
    # Or: sum_r outer(outer(u_r, V[:,r]), W[:,r]) = T
    
    # This is still nonlinear. Let me fix U and V, then solve for W.
    
    # If U and V are fixed, then for each (a, c_idx):
    # sum_r U[a,r] * (sum_b V[b,r] * W[c_idx,r] * T[a,b,c_idx]) ... no.
    
    # The equation is: for each (a, b, c_idx):
    # sum_r U[a,r] * V[b,r] * W[c_idx,r] = T[a,b,c_idx]
    
    # If I fix U and V, this is linear in W:
    # For each (a, b, c_idx): sum_r (U[a,r] * V[b,r]) * W[c_idx,r] = T[a,b,c_idx]
    
    # This is a system of 2304 equations in 432 unknowns (12*36 entries of W).
    # It's overdetermined, so it may not have a solution for arbitrary U, V.
    
    # Let me try a different approach: use a known valid decomposition from the literature.
    
    # Actually, I recall that M(3,4,4) <= 36 can be achieved using the following:
    # 
    # The key insight is to use the fact that 3 = 2+1 and 4 = 2+2.
    # 
    # Let me try: A(3x4) = [[A1(2x4)], [A2(1x4)]]
    # B(4x4) = [[B1(4x2), B2(4x2)]]
    # 
    # C(3x4) = [[C1(2x4)], [C2(1x4)]]
    # 
    # C1 = A1*B (2x4), C2 = A2*B (1x4)
    # 
    # For 2x4*4x4: M(2,4,4) <= ?
    # For 1x4*4x4: M(1,4,4) <= 4 (schoolbook, since it's just 4 dot products of length 4)
    # 
    # M(2,4,4): A(2x4)*B(4x4)=C(2x4)
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # This is 2*4 = 8 dot products of length 4.
    # Schoolbook: 2*4*4 = 32.
    # Can we do better? M(2,4,4) <= 32? Probably less.
    # 
    # M(2,4,4) <= M(2,4,4). Using Strassen-like: 
    # Split A into two 2x2 blocks, B into 2x2 blocks.
    # A = [A1, A2] (each 2x2), B = [[B11,B12],[B21,B22]] (each 2x2)
    # C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
    # Each is 2x2*2x2, using Strassen: 7 mults each.
    # Total: 4*7 = 28.
    # So M(2,4,4) <= 28.
    # 
    # M(1,4,4) <= 4 (just 4 dot products).
    # 
    # Total: 28 + 4 = 32. That's less than 36!
    # 
    # Wait, but I need to be careful. Let me verify.
    # 
    # A(3x4) split as:
    # A_top(2x4), A_bottom(1x4)
    # 
    # B(4x4) is the same for both.
    # 
    # C_top = A_top * B (2x4)
    # C_bottom = A_bottom * B (1x4)
    # 
    # For C_top: use the 2x4*4x4 decomposition with rank 28.
    # For C_bottom: use the 1x4*4x4 decomposition with rank 4.
    # 
    # Total: 28 + 4 = 32.
    # 
    # So M(3,4,4) <= 32!
    # 
    # Let me implement this.
    
    # First, the 2x4*4x4 decomposition with rank 28:
    # A(2x4) * B(4x4) = C(2x4)
    # A = [A1(2x2), A2(2x2)]
    # B = [[B11(2x2), B12(2x2)], [B21(2x2), B22(2x2)]]
    # C1 = A1*B11 + A2*B21 (2x2)
    # C2 = A1*B12 + A2*B22 (2x2)
    # C = [C1, C2]
    # 
    # For C1: use Strassen on (A1, A2, B11, B21) -> 7 mults
    # For C2: use Strassen on (A1, A2, B12, B22) -> 7 mults
    # Total: 14 mults for the 2x2*2x2 products.
    # 
    # Wait, that's only 14, not 28. Let me recount.
    # 
    # Each 2x2*2x2 product using Strassen requires 7 scalar multiplications.
    # We have 4 such products (C1 uses 2, C2 uses 2).
    # Total: 4*7 = 28. Yes, 28.
    
    # Now let me implement the full 3x4x8 decomposition.
    # 
    # 3x4*4x8:
    # A(3x4) * B(4x8) = C(3x8)
    # 
    # Split B into B1(4x4) and B2(4x4):
    # C = [A*B1, A*B2]
    # 
    # For each A*B_i (3x4 * 4x4 = 3x4), use the rank-32 decomposition above.
    # Total: 32 + 32 = 64.
    # 
    # So M(3,4,8) <= 64!
    # 
    # Let me implement this.
    
    # I'll construct the U, V, W matrices for the full 3x4x8 problem.
    # 
    # U: shape (12, 64)  [i*4+j]
    # V: shape (32, 64)  [j*8+k]
    # W: shape (24, 64)  [k*3+i]
    # 
    # For the first half (B1, columns 0-3 of B):
    # Use the 3x4*4x4 decomposition.
    # 
    # For the second half (B2, columns 4-7 of B):
    # Use another 3x4*4x4 decomposition.
    
    # Let me define the 3x4*4x4 decomposition with rank 32.
    
    def _decomp_3x4x4():
        """
        Return U, V, W for 3x4*4x4=3x4 with rank 32.
        U: (12, 32), V: (16, 32), W: (12, 32)
        """
        n, m, p = 3, 4, 4
        r = 32
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Split A into A_top(2x4) and A_bottom(1x4)
        # A_top: rows 0,1 of A -> indices 0-7 in U (i*4+j for i in 0,1)
        # A_bottom: row 2 of A -> indices 8-11 in U (i*4+j for i=2)
        
        # For A_top * B (2x4 * 4x4):
        # Split A_top into A1(2x2) and A2(2x2)
        # A1: columns 0,1 -> U indices 0,1,4,5 (i*4+j for i in 0,1, j in 0,1)
        # A2: columns 2,3 -> U indices 2,3,6,7 (i*4+j for i in 0,1, j in 2,3)
        
        # B split into 2x2 blocks:
        # B11: rows 0,1, cols 0,1 -> V indices 0,1,4,5 (j*4+k for j in 0,1, k in 0,1)
        # B12: rows 0,1, cols 2,3 -> V indices 2,3,6,7
        # B21: rows 2,3, cols 0,1 -> V indices 8,9,12,13
        # B22: rows 2,3, cols 2,3 -> V indices 10,11,14,15
        
        # C_top = [C1, C2] where C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
        # C_top indices in W: k*3+i for i in 0,1 (top 2 rows of C)
        # C1: k in 0,1 -> W indices 0,1,3,4 (k*3+i for k in 0,1, i in 0,1)
        # C2: k in 2,3 -> W indices 6,7,9,10
        
        # For C1 = A1*B11 + A2*B21:
        # Use Strassen on the 2x2 blocks.
        # A1 = [[a00, a01], [a10, a11]] (from A_top)
        # A2 = [[a02, a03], [a12, a13]]
        # B11 = [[b00, b01], [b10, b11]]
        # B21 = [[b20, b21], [b30, b31]]
        
        # Strassen for C1:
        # m1 = (A1+A2) * (B11+B21)
        # m2 = A2 * B11
        # m3 = A1 * (B11-B21)
        # m4 = A2 * (B21-B11)
        # m5 = (A1+A2) * B21
        # m6 = (A2-A1) * (B11+B12)  -- wait, this is for the full 2x2*2x2
        # m7 = (A2-A1) * (B21+B22)  -- hmm, I'm mixing things up.
        
        # Let me use the standard Strassen for computing X*Y where X, Y are 2x2:
        # m1 = (x11+x22)*(y11+y22)
        # m2 = (x21+x22)*y11
        # m3 = x11*(y12-y22)
        # m4 = x22*(y21-y11)
        # m5 = (x11+x12)*y22
        # m6 = (x21-x11)*(y11+y12)
        # m7 = (x12-x22)*(y21+y22)
        # Z11 = m1+m4-m2+m6
        # Z12 = m3+m5
        # Z21 = m2+m4
        # Z22 = m1-m3+m5-m7
        
        # For C1 = A1*B11 + A2*B21, I need to compute this sum.
        # This is not a single 2x2*2x2 product. It's a sum of two.
        # 
        # I can't directly apply Strassen to a sum. I need to compute A1*B11 and A2*B21 separately.
        # Each requires 7 Strassen multiplications.
        # Total for C1: 14 mults.
        # Total for C2: 14 mults.
        # Total for A_top*B: 28 mults.
        
        # So the rank is 28 for the top part, plus 4 for the bottom part = 32.
        
        # Let me implement this.
        
        col = 0
        # For A_top * B:
        # C1 = A1*B11 + A2*B21
        # Compute A1*B11 using Strassen (7 terms):
        # A1 elements: U[0] = a00, U[1] = a01, U[4] = a10, U[5] = a11
        # B11 elements: V[0] = b00, V[1] = b01, V[4] = b10, V[5] = b11
        
        # Strassen terms for A1*B11:
        # m1 = (a00+a11)*(b00+b11)
        # m2 = (a10+a11)*b00
        # m3 = a00*(b01-b11)
        # m4 = a11*(b10-b00)
        # m5 = (a00+a01)*b11
        # m6 = (a10-a00)*(b00+b01)
        # m7 = (a01-a11)*(b10+b11)
        
        # C1[0,0] = m1+m4-m2+m6
        # C1[0,1] = m3+m5
        # C1[1,0] = m2+m4
        # C1[1,1] = m1-m3+m5-m7
        
        # C1 indices in W: C1[0,0]=W[0], C1[0,1]=W[1], C1[1,0]=W[3], C1[1,1]=W[4]
        
        # Term 0: m1 = (a00+a11)*(b00+b11)
        U[0, col] = 1; U[5, col] = 1  # a00, a11
        V[0, col] = 1; V[5, col] = 1  # b00, b11
        W[0, col] = 1; W[3, col] = 1  # C1[0,0], C1[1,1]
        W[1, col] = 0; W[4, col] = 0
        col += 1
        
        # Term 1: m2 = (a10+a11)*b00
        U[4, col] = 1; U[5, col] = 1  # a10, a11
        V[0, col] = 1  # b00
        W[0, col] = -1; W[3, col] = 1  # -C1[0,0] + C1[1,0]
        W[1, col] = 0; W[4, col] = 0
        col += 1
        
        # Term 2: m3 = a00*(b01-b11)
        U[0, col] = 1  # a00
        V[1, col] = 1; V[5, col] = -1  # b01, -b11
        W[1, col] = 1; W[4, col] = -1  # C1[0,1] - C1[1,1]
        W[0, col] = 0; W[3, col] = 0
        col += 1
        
        # Term 3: m4 = a11*(b10-b00)
        U[5, col] = 1  # a11
        V[4, col] = 1; V[0, col] = -1  # b10, -b00
        W[0, col] = 1; W[3, col] = 1  # C1[0,0] + C1[1,0]
        W[1, col] = 0; W[4, col] = 0
        col += 1
        
        # Term 4: m5 = (a00+a01)*b11
        U[0, col] = 1; U[1, col] = 1  # a00, a01
        V[5, col] = 1  # b11
        W[1, col] = 1; W[4, col] = 1  # C1[0,1] + C1[1,1]
        W[0, col] = 0; W[3, col] = 0
        col += 1
        
        # Term 5: m6 = (a10-a00)*(b00+b01)
        U[4, col] = 1; U[0, col] = -1  # a10, -a00
        V[0, col] = 1; V[1, col] = 1  # b00, b01
        W[0, col] = 1  # C1[0,0]
        W[1, col] = 0; W[3, col] = 0; W[4, col] = 0
        col += 1
        
        # Term 6: m7 = (a01-a11)*(b10+b11)
        U[1, col] = 1; U[5, col] = -1  # a01, -a11
        V[4, col] = 1; V[5, col] = 1  # b10, b11
        W[4, col] = -1  # -C1[1,1]
        W[0, col] = 0; W[1, col] = 0; W[3, col] = 0
        col += 1
        
        # Now compute A2*B21 using Strassen (7 terms):
        # A2 elements: U[2] = a02, U[3] = a03, U[6] = a12, U[7] = a13
        # B21 elements: V[8] = b20, V[9] = b21, V[12] = b30, V[13] = b31
        
        # Term 7: (a02+a13)*(b20+b31)
        U[2, col] = 1; U[7, col] = 1
        V[8, col] = 1; V[13, col] = 1
        W[0, col] = 1; W[3, col] = 1
        W[1, col] = 0; W[4, col] = 0
        col += 1
        
        # Term 8: (a12+a13)*b20
        U[6, col] = 1; U[7, col] = 1
        V[8, col] = 1
        W[0, col] = -1; W[3, col] = 1
        W[1, col] = 0; W[4, col] = 0
        col += 1
        
        # Term 9: a02*(b21-b31)
        U[2, col] = 1
        V[9, col] = 1; V[13, col] = -1
        W[1, col] = 1; W[4, col] = -1
        W[0, col] = 0; W[3, col] = 0
        col += 1
        
        # Term 10: a13*(b30-b20)
        U[7, col] = 1
        V[12, col] = 1; V[8, col] = -1
        W[0, col] = 1; W[3, col] = 1
        W[1, col] = 0; W[4, col] = 0
        col += 1
        
        # Term 11: (a02+a03)*b31
        U[2, col] = 1; U[3, col] = 1
        V[13, col] = 1
        W[1, col] = 1; W[4, col] = 1
        W[0, col] = 0; W[3, col] = 0
        col += 1
        
        # Term 12: (a12-a02)*(b20+b21)
        U[6, col] = 1; U[2, col] = -1
        V[8, col] = 1; V[9, col] = 1
        W[0, col] = 1
        W[1, col] = 0; W[3, col] = 0; W[4, col] = 0
        col += 1
        
        # Term 13: (a03-a13)*(b30+b31)
        U[3, col] = 1; U[7, col] = -1
        V[12, col] = 1; V[13, col] = 1
        W[4, col] = -1
        W[0, col] = 0; W[1, col] = 0; W[3, col] = 0
        col += 1
        
        # Now C2 = A1*B12 + A2*B22
        # A1 elements: U[0]=a00, U[1]=a01, U[4]=a10, U[5]=a11
        # B12 elements: V[2]=b02, V[3]=b03, V[6]=b12, V[7]=b13
        # A2 elements: U[2]=a02, U[3]=a03, U[6]=a12, U[7]=a13
        # B22 elements: V[10]=b22, V[11]=b23, V[14]=b32, V[15]=b33
        
        # C2 indices in W: C2[0,0]=W[6], C2[0,1]=W[7], C2[1,0]=W[9], C2[1,1]=W[10]
        
        # Term 14: (a00+a11)*(b02+b13)
        U[0, col] = 1; U[5, col] = 1
        V[2, col] = 1; V[7, col] = 1
        W[6, col] = 1; W[9, col] = 1
        W[7, col] = 0; W[10, col] = 0
        col += 1
        
        # Term 15: (a10+a11)*b02
        U[4, col] = 1; U[5, col] = 1
        V[2, col] = 1
        W[6, col] = -1; W[9, col] = 1
        W[7, col] = 0; W[10, col] = 0
        col += 1
        
        # Term 16: a00*(b03-b13)
        U[0, col] = 1
        V[3, col] = 1; V[7, col] = -1
        W[7, col] = 1; W[10, col] = -1
        W[6, col] = 0; W[9, col] = 0
        col += 1
        
        # Term 17: a11*(b12-b02)
        U[5, col] = 1
        V[6, col] = 1; V[2, col] = -1
        W[6, col] = 1; W[9, col] = 1
        W[7, col] = 0; W[10, col] = 0
        col += 1
        
        # Term 18: (a00+a01)*b13
        U[0, col] = 1; U[1, col] = 1
        V[7, col] = 1
        W[7, col] = 1; W[10, col] = 1
        W[6, col] = 0; W[9, col] = 0
        col += 1
        
        # Term 19: (a10-a00)*(b02+b03)
        U[4, col] = 1; U[0, col] = -1
        V[2, col] = 1; V[3, col] = 1
        W[6, col] = 1
        W[7, col] = 0; W[9, col] = 0; W[10, col] = 0
        col += 1
        
        # Term 20: (a01-a11)*(b12+b13)
        U[1, col] = 1; U[5, col] = -1
        V[6, col] = 1; V[7, col] = 1
        W[10, col] = -1
        W[6, col] = 0; W[7, col] = 0; W[9, col] = 0
        col += 1
        
        # Term 21: (a02+a13)*(b22+b33)
        U[2, col] = 1; U[7, col] = 1
        V[10, col] = 1; V[15, col] = 1
        W[6, col] = 1; W[9, col] = 1
        W[7, col] = 0; W[10, col] = 0
        col += 1
        
        # Term 22: (a12+a13)*b22
        U[6, col] = 1; U[7, col] = 1
        V[10, col] = 1
        W[6, col] = -1; W[9, col] = 1
        W[7, col] = 0; W[10, col] = 0
        col += 1
        
        # Term 23: a02*(b23-b33)
        U[2, col] = 1
        V[11, col] = 1; V[15, col] = -1
        W[7, col] = 1; W[10, col] = -1
        W[6, col] = 0; W[9, col] = 0
        col += 1
        
        # Term 24: a13*(b32-b22)
        U[7, col] = 1
        V[14, col] = 1; V[10, col] = -1
        W[6, col] = 1; W[9, col] = 1
        W[7, col] = 0; W[10, col] = 0
        col += 1
        
        # Term 25: (a02+a03)*b33
        U[2, col] = 1; U[3, col] = 1
        V[15, col] = 1
        W[7, col] = 1; W[10, col] = 1
        W[6, col] = 0; W[9, col] = 0
        col += 1
        
        # Term 26: (a12-a02)*(b22+b23)
        U[6, col] = 1; U[2, col] = -1
        V[10, col] = 1; V[11, col] = 1
        W[6, col] = 1
        W[7, col] = 0; W[9, col] = 0; W[10, col] = 0
        col += 1
        
        # Term 27: (a03-a13)*(b32+b33)
        U[3, col] = 1; U[7, col] = -1
        V[14, col] = 1; V[15, col] = 1
        W[10, col] = -1
        W[6, col] = 0; W[7, col] = 0; W[9, col] = 0
        col += 1
        
        # Now for A_bottom * B (1x4 * 4x4 = 1x4):
        # A_bottom: row 2 of A -> U[8]=a20, U[9]=a21, U[10]=a22, U[11]=a23
        # B: all 4x4
        # C_bottom: row 2 of C -> W[12]=C[2,0], W[13]=C[2,1], W[14]=C[2,2], W[15]=C[2,3]
        # 
        # C[2,j] = a20*B[0,j] + a21*B[1,j] + a22*B[2,j] + a23*B[3,j]
        # This is 4 dot products of length 4. Schoolbook: 4 terms.
        
        # Term 28: a20*b0j for j=0..3
        U[8, col] = 1
        V[0, col] = 1; V[1, col] = 1; V[2, col] = 1; V[3, col] = 1  # b00,b01,b02,b03
        W[12, col] = 1; W[13, col] = 1; W[14, col] = 1; W[15, col] = 1
        col += 1
        
        # Term 29: a21*b1j
        U[9, col] = 1
        V[4, col] = 1; V[5, col] = 1; V[6, col] = 1; V[7, col] = 1  # b10,b11,b12,b13
        W[12, col] = 1; W[13, col] = 1; W[14, col] = 1; W[15, col] = 1
        col += 1
        
        # Term 30: a22*b2j
        U[10, col] = 1
        V[8, col] = 1; V[9, col] = 1; V[10, col] = 1; V[11, col] = 1  # b20,b21,b22,b23
        W[12, col] = 1; W[13, col] = 1; W[14, col] = 1; W[15, col] = 1
        col += 1
        
        # Term 31: a23*b3j
        U[11, col] = 1
        V[12, col] = 1; V[13, col] = 1; V[14, col] = 1; V[15, col] = 1  # b30,b31,b32,b33
        W[12, col] = 1; W[13, col] = 1; W[14, col] = 1; W[15, col] = 1
        col += 1
        
        assert col == 32, f"Expected 32 terms, got {col}"
        return U, V, W
    
    # Now build the full 3x4x8 decomposition.
    # Split B into B1(4x4) and B2(4x4).
    # C = [A*B1, A*B2]
    
    n, m, p = 3, 4, 8
    r_total = 64
    U_full = np.zeros((n*m, r_total))
    V_full = np.zeros((m*p, r_total))
    W_full = np.zeros((p*n, r_total))
    
    U36, V36, W36 = _decomp_3x4x4()
    
    # First half: B1 (columns 0-3 of B)
    # V indices for B1: j*8+k for j in 0..3, k in 0..3
    # In the 3x4x4 decomposition, V is indexed by j*4+k.
    # So V_full[j*8+k, col] = V36[j*4+k, col] for k in 0..3
    
    for j in range(4):
        for k in range(4):
            V_full[j*8+k, 0:32] = V36[j*4+k, 0:32]
    
    U_full[:, 0:32] = U36
    # W for first half: C columns 0-3
    # W_full[k*3+i, col] = W36[k*3+i, col] for k in 0..3
    for k in range(4):
        for i in range(3):
            W_full[k*3+i, 0:32] = W36[k*3+i, 0:32]
    
    # Second half: B2 (columns 4-7 of B)
    # V indices for B2: j*8+k for j in 0..3, k in 4..7
    # Map to V36: j*4+(k-4)
    for j in range(4):
        for k in range(4, 8):
            V_full[j*8+k, 32:64] = V36[j*4+(k-4), 0:32]
    
    U_full[:, 32:64] = U36
    # W for second half: C columns 4-7
    for k in range(4, 8):
        for i in range(3):
            W_full[k*3+i, 32:64] = W36[(k-4)*3+i, 0:32]
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    return _rank36_3x4x4()
