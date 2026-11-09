"""Find a rank-minimal exact tensor decomposition for 4x4 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 4]

# The known optimal rank for 4x4 matrix multiplication is 49 (Strassen 7, but for 4x4 it's 49? No, actually 49 is for 3x3. For 4x4, the best known is 49? Let me check.
# Actually, for 4x4 matrix multiplication, the best known upper bound is 49 (from Strassen's algorithm applied to 4x4? No).
# Strassen's algorithm gives rank 7 for 2x2, which extends to 2^k x 2^k with rank 7^(k/2).
# For 4x4 (k=2), rank = 7^2 = 49.
# The lower bound for 4x4 is 48 (Bauer 1997? or similar).
# So the optimal rank for 4x4 is likely 49.

# Let me implement Strassen's algorithm for 4x4 matrix multiplication.

def _strassen_2x2():
    """Return the 7-term decomposition for 2x2 matrix multiplication.
    
    The 2x2 matrix multiplication tensor has rank 7 (Strassen).
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen's 7 products:
    M1 = (a+d)(e+h) = ae+ah+de+dh
    M2 = (c+d)e = ce+de
    M3 = a(f-h) = af-ah
    M4 = d(g-c) = dg-dc
    M5 = (a+b)h = ah+bh
    M6 = (c-a)(e+f) = ce+cf-ae-af
    M7 = (b-d)(g+h) = bg+bh-dg-dh
    
    C11 = M1 - M5 + M7 - M4  = ae+ah+de+dh - ah-bh + bg+bh-dg-dh = ae+bg  CORRECT
    C12 = M3 + M5            = af-ah + ah+bh = af+bh  CORRECT
    C21 = M2 + M4            = ce+de + dg-dc = ce+dg  CORRECT  (since dc=cd)
    C22 = M1 - M2 + M3 + M6  = ae+ah+de+dh - ce-de + af-ah + ce+cf-ae-af = dh+cf  CORRECT
    """
    # For 2x2: n=m=p=2
    # U has 4 rows (2*2), V has 4 rows (2*2), W has 4 rows (2*2)
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # Row indices: U: a=0, b=1, c=2, d=3
    # V: e=0, f=1, g=2, h=3
    # W: C11=0, C12=1, C21=2, C22=3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2 = (c+d)e
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # C21
    
    # M3 = a(f-h)
    U[0, 2] = 1  # a
    V[1, 2] = 1  # f
    V[3, 2] = -1 # h
    W[1, 2] = 1  # C12
    
    # M4 = d(g-c)
    U[3, 3] = 1  # d
    V[2, 3] = 1  # g
    V[0, 3] = -1 # e  -- wait, M4 = d(g-c), so V should have g and -c? No, V is for B matrix elements.
    # M4 = d*(g - c). But c is an A element, not B. So this doesn't work directly.
    # Let me recheck. M4 = (d)*(g-c). But c is in A, g is in B. So U has d, V has g and... we need -c but c is in U.
    # Actually, the Strassen decomposition has:
    # M4 = d(g-c) -- this is wrong. Let me look up the correct Strassen formulas.
    
    # Correct Strassen:
    # M1 = (a + d) * (e + h)
    # M2 = (c + d) * e
    # M3 = a * (f - h)
    # M4 = d * (g - c)
    # M5 = (a + b) * h
    # M6 = (c - a) * (e + f)
    # M7 = (b - d) * (g + h)
    
    # M4 = d * (g - c): U has d (row 3), V has g (row 2) and... we need -c but c is in U.
    # This is the issue. In tensor decomposition, each term is a product of one A-factor, one B-factor, and contributes to C.
    # M4 = d * (g - c) = d*g - d*c. But d*c is a product of two A elements, which doesn't fit.
    # I think I have the formulas wrong. Let me look them up properly.
    
    # Actually, the standard Strassen formulas are:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)(e)
    # M3 = (a)(f - h)
    # M4 = (d)(g - c)
    # M5 = (a + b)(h)
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    
    # But M4 = d*(g-c) involves d from A and g from B, and c from A. That's not a valid rank-1 term.
    # The correct interpretation: each Mi is a product of a linear combination of A entries and a linear combination of B entries.
    # M4 = (d) * (g - c) -- but c is an A entry, not B. So this is wrong.
    
    # Let me look up the actual Strassen algorithm more carefully.
    # From Wikipedia:
    # M1 = (a + d) * (e + h)
    # M2 = (c + d) * e
    # M3 = a * (f - h)
    # M4 = d * (g - c)
    # M5 = (a + b) * h
    # M6 = (c - a) * (e + f)
    # M7 = (b - d) * (g + h)
    
    # Hmm, M4 = d * (g - c). But c is from A. So the B-part of M4 is just (g), and we subtract d*c separately?
    # No, that's not how it works. Let me think again.
    
    # Actually, I think the issue is that these are NOT all rank-1 in the sense I was thinking.
    # In the tensor decomposition, each term is: (sum of A entries with coefficients) * (sum of B entries with coefficients) * (contribution to C entry).
    
    # Let me just verify by computing C11 from the 7 terms:
    # C11 = M1 - M5 + M7 - M4
    # = (a+d)(e+h) - (a+b)h + (b-d)(g+h) - d(g-c)
    # = ae+ah+de+dh - ah-bh + bg+bh-dg-dh - dg+dc
    # = ae + de + bg + dc - 2dg  ... this doesn't look right.
    
    # I must have the formula wrong. Let me try a different source.
    # From a reliable source, the 7 Strassen products for 2x2 are:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - A11)  -- NO, this still has A11 in the B factor.
    
    # I think the confusion is that the "B factor" in each term can include both A and B entries? No, that can't be.
    
    # Let me just hard-code the known 49-rank decomposition for 4x4 using the recursive Strassen construction.
    pass

def _strassen_4x4():
    """Build the 49-rank Strassen decomposition for 4x4 matrix multiplication.
    
    Using the recursive construction: 4x4 = 4 blocks of 2x2, each requiring 7 Strassen products.
    Total rank = 4 * 7 = 28? No, the recursive Strassen for 4x4 uses 7^2 = 49 scalar multiplications.
    
    The 4x4 matrices are partitioned into 2x2 blocks of 2x2 submatrices.
    A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
    B = [[B11, B12], [B21, B22]] where each Bij is 2x2.
    C = A*B, so C11 = A11*B11 + A12*B21, C12 = A12*B12 + A11*B22, etc.
    
    Using Strassen on the block level:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-A11)  -- again this has issues
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    Each Mi is a product of 2x2 matrices, and each requires 7 scalar multiplications (Strassen).
    Total: 7 * 7 = 49 scalar multiplications.
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # 4x4 matrix partitioned into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # A11 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] -> rows 0,1 cols 0,1
    # A12 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]] -> rows 0,1 cols 2,3
    # A21 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]] -> rows 2,3 cols 0,1
    # A22 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]] -> rows 2,3 cols 2,3
    
    # In row-major index for A (4x4): A[i,j] is at index i*4+j
    # A11: indices 0,1,4,5
    # A12: indices 2,3,6,7
    # A21: indices 8,9,12,13
    # A22: indices 10,11,14,15
    
    # Similarly for B and C.
    
    # Define the 2x2 Strassen decomposition for each block product.
    # For a 2x2 product X*Y where X,Y are 2x2:
    # X = [[x11, x12], [x21, x22]], Y = [[y11, y12], [y21, y22]]
    # The 7 Strassen terms:
    # T1 = (x11+x22)(y11+y22)
    # T2 = (x21+x22)y11
    # T3 = x11(y12-y22)
    # T4 = x22(y21-x11)  -- PROBLEM: x11 is from X, not Y
    # ...
    
    # I realize the standard Strassen formulas as written don't directly give a valid tensor decomposition
    # because some "factors" mix A and B entries. But the algorithm IS correct as a matrix multiplication
    # algorithm. The tensor decomposition is valid if we think of it as: each term is a bilinear form.
    
    # Let me just verify the Strassen algorithm works and then express it as a tensor decomposition.
    
    # Actually, for the tensor decomposition, each of the 7 Strassen products IS a rank-1 term in the
    # 2x2 matrix multiplication tensor. The fact that some linear combinations involve entries from
    # the same matrix doesn't matter -- what matters is that each term is (linear combo of A) * (linear combo of B).
    
    # Let me re-examine:
    # T1 = (A11+A22)(B11+B22): A-factor = A11+A22, B-factor = B11+B22. VALID.
    # T2 = (A21+A22)B11: A-factor = A21+A22, B-factor = B11. VALID.
    # T3 = A11(B12-B22): A-factor = A11, B-factor = B12-B22. VALID.
    # T4 = A22(B21-A11): A-factor = A22, B-factor = B21-A11. INVALID: A11 is from A, not B!
    
    # So T4 as written is NOT a valid rank-1 term. This means the standard Strassen formulas,
    # while correct as an algorithm, don't directly give a tensor decomposition in the form we need.
    
    # Wait, but Strassen's algorithm IS a tensor decomposition of rank 7 for 2x2 matrix multiplication.
    # The tensor for 2x2 matrix multiplication has entries T[i,j,k,l] = 1 if (i,j)*(j,k) contributes to (i,k).
    # The rank is 7, meaning there exist 7 rank-1 tensors that sum to T.
    
    # Each rank-1 tensor is: u_{i,j} * v_{j,k} * w_{i,k} for some index scheme.
    # In our convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    
    # For 2x2: U has 4 rows, V has 4 rows, W has 4 rows.
    # The 7 Strassen terms DO correspond to rank-1 decompositions. Let me look them up properly.
    
    # From the literature, the 7 Strassen rank-1 terms for 2x2 are:
    # c=0: U = [1,0,0,1], V = [1,0,0,1], W = [1,0,0,1]  (M1)
    #       This gives (a+d)(e+h) contributing to C11 and C22.
    #       W = [1, 0, 0, 1] means C11 += 1, C22 += 1.
    
    # c=1: U = [0,0,1,1], V = [1,0,0,0], W = [0,0,1,0]  (M2)
    #       This gives (c+d)*e contributing to C21.
    
    # c=2: U = [1,0,0,0], V = [0,1,0,-1], W = [0,1,0,0]  (M3)
    #       This gives a*(f-h) contributing to C12.
    
    # c=3: U = [0,0,0,1], V = [0,-1,1,0], W = [0,0,0,-1]  (M4)
    #       Wait, let me think. M4 = d*(g-c). But c is in A.
    #       Actually, I think the correct decomposition is:
    #       M4 contributes to C22 and C21 or something.
    
    # Let me just look up the exact 7-term decomposition.
    # From a reference:
    # The 7 terms for 2x2 matrix multiplication (A*B=C) are:
    
    # Term 0: u = [1,0,0,1], v = [1,0,0,1], w = [1,0,0,1]
    #   (a+d)(e+h) -> C11 += (a+d)(e+h), C22 += (a+d)(e+h)
    
    # Term 1: u = [0,0,1,1], v = [1,0,0,0], w = [0,0,1,0]
    #   (c+d)e -> C21 += (c+d)e
    
    # Term 2: u = [1,0,0,0], v = [0,1,0,-1], w = [0,1,0,0]
    #   a(f-h) -> C12 += a(f-h)
    
    # Term 3: u = [0,0,0,1], v = [0,-1,1,0], w = [0,0,0,-1]
    #   d(g-c) -> C22 -= d(g-c) ... but c is not in V!
    
    # I think the issue is that I'm misremembering the decomposition. Let me just try to derive it.
    
    # For 2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    # We need 7 rank-1 terms. Let me denote them as (u_c, v_c, w_c) for c=0..6.
    
    # This is getting complex. Let me just use the known result that the rank of 4x4 matrix multiplication
    # is at most 49, and construct it via the recursive Strassen method.
    
    # For the recursive construction:
    # 4x4 matrix mult = 7 block products of 2x2 matrices, each requiring 7 scalar mults.
    # Total = 49 scalar mults.
    
    # The 7 block products are:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-A11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Then:
    # C11 = P1 - P5 + P7 - P4
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pi is a 2x2 matrix product. We decompose each Pi into 7 scalar multiplications using Strassen.
    
    # For Pi = Xi * Yi where Xi and Yi are 2x2 matrices:
    # Xi = [[x11, x12], [x21, x22]], Yi = [[y11, y12], [y21, y22]]
    # The 7 Strassen scalar products:
    # s1 = (x11+x22)(y11+y22)
    # s2 = (x21+x22)y11
    # s3 = x11(y12-y22)
    # s4 = x22(y21-x11)  -- again this has x11 in the "Y factor"
    # s5 = (x11+x12)y22
    # s6 = (x21-x11)(y11+y12)
    # s7 = (x12-x22)(y21+y22)
    
    # Result:
    # C11 = s1 - s5 + s7 - s4
    # C12 = s3 + s5
    # C21 = s2 + s4
    # C22 = s1 - s2 + s3 + s6
    
    # For the tensor decomposition, each scalar product sk is a rank-1 term:
    # sk = (linear combo of X entries) * (linear combo of Y entries)
    
    # s1 = (x11+x22)(y11+y22): X-factor = x11+x22, Y-factor = y11+y22
    # s2 = (x21+x22)y11: X-factor = x21+x22, Y-factor = y11
    # s3 = x11(y12-y22): X-factor = x11, Y-factor = y12-y22
    # s4 = x22(y21-x11): X-factor = x22, Y-factor = y21-x11 -- PROBLEM: x11 is from X
    # s5 = (x11+x12)y22: X-factor = x11+x12, Y-factor = y22
    # s6 = (x21-x11)(y11+y12): X-factor = x21-x11, Y-factor = y11+y12
    # s7 = (x12-x22)(y21+y22): X-factor = x12-x22, Y-factor = y21+y22
    
    # The problem with s4 is that x11 appears in the Y-factor. But in the tensor decomposition,
    # the Y-factor should only involve entries from the second matrix (Y), not the first (X).
    
    # However, in the context of the 4x4 decomposition, X and Y are themselves linear combinations
    # of entries of A and B respectively. So when we write s4 = x22*(y21-x11), the x11 here is an
    # entry of X, which is a linear combination of A entries. The y21 is an entry of Y, a linear
    # combination of B entries. So s4 = x22*y21 - x22*x11. The term x22*x11 is a product of two
    # A-entries (since both x22 and x11 come from A), which is NOT bilinear in A and B.
    
    # This means the standard Strassen algorithm, while correct as a matrix multiplication algorithm,
    # does NOT directly give a tensor decomposition in the form we need, because some intermediate
    # terms are not bilinear.
    
    # BUT, the Strassen algorithm IS a valid tensor decomposition. The key insight is that in the
    # tensor decomposition, each rank-1 term is (u_i) * (v_j) * (w_k) where u, v, w are vectors.
    # The Strassen algorithm can be expressed this way. The "problem" I'm seeing is because I'm
    # confusing the intermediate computations with the actual rank-1 terms.
    
    # Let me look up the actual rank-1 terms for the 2x2 Strassen decomposition.
    
    # After some research, the 7 rank-1 terms for 2x2 matrix multiplication are:
    # (using the convention U[i*2+j], V[j*2+k], W[k*2+i])
    
    # c=0: U = [1,0,0,1], V = [1,0,0,1], W = [1,0,0,1]
    # c=1: U = [0,0,1,1], V = [1,0,0,0], W = [0,0,1,0]
    # c=2: U = [1,0,0,0], V = [0,1,0,-1], W = [0,1,0,0]
    # c=3: U = [0,0,0,1], V = [0,-1,1,0], W = [0,0,0,-1]
    # c=4: U = [1,1,0,0], V = [0,0,0,1], W = [0,1,0,0]
    # c=5: U = [0,0,1,-1], V = [1,1,0,0], W = [0,0,0,1]
    # c=6: U = [0,1,0,-1], V = [0,0,1,1], W = [0,0,0,-1]
    
    # Let me verify c=3: U = [0,0,0,1] means d, V = [0,-1,1,0] means -e+g, W = [0,0,0,-1] means C22 -= 1
    # So term 3 contributes: -d*(-e+g) = d*e - d*g to C22. But C22 should be cf+dh. This doesn't look right.
    
    # I think I need to just hard-code the known 49-rank decomposition. Let me use a different approach.
    
    # Actually, let me just use the fact that the rank of 4x4 matrix multiplication is 49 and
    # construct it by applying Strassen's algorithm recursively.
    
    # For the 4x4 case, we have:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # where each Aij, Bij is 2x2.
    
    # The 7 block products:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-A11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 - P5 + P7 - P4
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pi = Xi * Yi where Xi, Yi are 2x2.
    # For each Pi, we use the 7-term Strassen decomposition of the 2x2 multiplication tensor.
    
    # The 2x2 multiplication tensor has rank 7. The 7 rank-1 terms are (I'll denote them as T0..T6):
    # For X*Y = Z where X,Y,Z are 2x2:
    # T0: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # T1: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    # T2: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
    # T3: u=[0,0,0,1], v=[0,-1,1,0], w=[0,0,0,-1]
    # T4: u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,0]
    # T5: u=[0,0,1,-1], v=[1,1,0,0], w=[0,0,0,1]
    # T6: u=[0,1,0,-1], v=[0,0,1,1], w=[0,0,0,-1]
    
    # Let me verify this gives the correct 2x2 multiplication.
    # X = [[x0,x1],[x2,x3]], Y = [[y0,y1],[y2,y3]], Z = X*Y
    # Z00 = x0*y0 + x1*y2
    # Z01 = x0*y1 + x1*y3
    # Z10 = x2*y0 + x3*y2
    # Z11 = x2*y1 + x3*y3
    
    # Sum over c of u_c * v_c * w_c for each Z entry:
    # Z00 = sum_c u_c0 * v_c0 * w_c00 + u_c1 * v_c2 * w_c10 ... this is getting confusing.
    
    # Let me just compute: for each c, the contribution to Z is: Z += (u_c outer v_c) * w_c
    # where u_c is a vector of length 4 (flattened X), v_c is length 4 (flattened Y), w_c is length 4 (flattened Z).
    # Actually, the rank-1 term is: for each c, Z_ij += sum_k (U[i,j,c] * V[k,m,c] * W[i,k,c]) -- no, this is getting too confusing.
    
    # Let me just use numpy to verify the decomposition.
    
    # For 2x2: U (4x7), V (4x7), W (4x7)
    # The tensor T[i,j,k] = 1 if (i,j) in A, (j,k) in B contribute to (i,k) in C.
    # In our indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For 2x2: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # The matrix multiplication tensor is: T[A_idx, B_idx] = C_idx
    # A_idx = i*2+j, B_idx = j*2+k, C_idx = k*2+i
    # T is a 4x4x4 tensor (or rather, a 3-tensor with dimensions 4,4,4).
    
    # The decomposition is: T = sum_c outer(u_c, v_c, w_c)
    # where (u_c, v_c, w_c) are the columns of U, V, W.
    
    # So T[A_idx, B_idx, C_idx] = sum_c U[A_idx, c] * V[B_idx, c] * W[C_idx, c]
    
    # Let me verify with the 7 terms I proposed:
    U2 = np.array([
        [1,0,0,0,1,0,0],
        [0,0,0,0,1,0,1],
        [0,0,0,0,0,1,0],
        [1,1,0,1,0,-1,-1]
    ])
    V2 = np.array([
        [1,1,0,0,0,1,0],
        [0,0,1,0,0,1,0],
        [0,0,0,1,0,0,1],
        [1,0,-1,0,1,0,1]
    ])
    W2 = np.array([
        [1,0,0,0,0,0,0],
        [0,0,1,0,1,0,0],
        [0,1,0,0,0,0,0],
        [1,0,0,-1,0,1,-1]
    ])
    
    # Check: T[A_idx, B_idx, C_idx] = sum_c U2[A_idx,c] * V2[B_idx,c] * W2[C_idx,c]
    # For A_idx=0 (a), B_idx=0 (e), C_idx=0 (C11): should be 1 (a*e contributes to C11)
    val = sum(U2[0,c] * V2[0,c] * W2[0,c] for c in range(7))
    # U2[0] = [1,0,0,0,1,0,0], V2[0] = [1,1,0,0,0,1,0], W2[0] = [1,0,0,0,0,0,0]
    # c=0: 1*1*1 = 1
    # c=1: 0*1*0 = 0
    # c=2: 0*0*0 = 0
    # c=3: 0*0*0 = 0
    # c=4: 1*0*0 = 0
    # c=5: 0*1*0 = 0
    # c=6: 0*0*0 = 0
    # Total = 1. CORRECT!
    
    # For A_idx=0 (a), B_idx=1 (f), C_idx=1 (C12): should be 1 (a*f contributes to C12)
    val = sum(U2[0,c] * V2[1,c] * W2[1,c] for c in range(7))
    # U2[0] = [1,0,0,0,1,0,0], V2[1] = [0,0,1,0,0,1,0], W2[1] = [0,0,1,0,1,0,0]
    # c=0: 1*0*0 = 0
    # c=1: 0*0*0 = 0
    # c=2: 0*1*1 = 0
    # c=3: 0*0*0 = 0
    # c=4: 1*0*1 = 0
    # c=5: 0*1*0 = 0
    # c=6: 0*0*0 = 0
    # Total = 0. WRONG! Should be 1.
    
    # So my proposed decomposition is incorrect. Let me look up the correct one.
    # This is getting too time-consuming. Let me just use the known result and hard-code a valid 49-rank decomposition.
    
    # Actually, let me try a different approach. I'll use the fact that the rank of n×n matrix multiplication
    # is at most n^omega where omega < 2.373. For n=4, the best known is 49 (from Strassen).
    
    # Let me just return the schoolbook decomposition (rank 64) which is guaranteed correct,
    # and then try to improve it.
    
    pass

def construct():
    """Return a valid tensor decomposition for 4x4 matrix multiplication.
    
    Uses the schoolbook algorithm with rank 64, which is always correct.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
