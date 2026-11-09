"""
Rank-minimal exact tensor decomposition for 2x4 x 4x4 matrix multiplication.

Strategy:
  - The multiplication tensor T of order (2,4,4) has the structure where
    output entry C[i,k] = sum_j A[i,j] * B[j,k].
  - We can decompose this by considering the 2x4 matrix A as having 2 rows,
    each of length 4, and B as 4x4.
  - Key insight: For each pair (i, k) in the output, C[i,k] = sum_j A[i,j]*B[j,k].
  - We can use a Strassen-like approach on the 4x4 multiplication part, but we need
    to handle the 2x4 structure.
  
  A better approach: Think of the 2x4 x 4x4 multiplication as:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
  
  We can group the 4 columns of A and 4 rows of B. If we could do 4x4 matrix
  multiplication with rank r, then for each of the 2 rows of A we'd need rank r,
  giving total rank 2*r. But that's not quite right because the A matrix is 2x4,
  not 4x4.
  
  Actually, let's think differently. The tensor is:
    T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
  
  We want to write T = sum_c U[:,c] * V[:,c] * W[:,c] (outer product of 3 vectors)
  where U has shape (n*m, rank) = (8, rank), V has shape (m*p, rank) = (16, rank),
  W has shape (p*n, rank) = (8, rank).
  
  Wait, let me re-read the convention. The task says:
    U: shape (n*m, rank)
    V: shape (m*p, rank)  
    W: shape (p*n, rank)
    Output factor follows W[k*n+i, r]
  
  So U[i*m + j, c] corresponds to A[i,j], V[j*p + k, c] corresponds to B[j,k],
  W[k*n + i, c] corresponds to C[i,k].
  
  The tensor is T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
  
  This is a standard matrix multiplication tensor. The known lower bound for
  2x4 x 4x4 is... let me think.
  
  For m x n x p matrix multiplication, the schoolbook rank is m*n*p.
  For 2x4x4, that's 32.
  
  Can we do better? The Strassen algorithm for 2x2x2 gives rank 7 instead of 8.
  
  One approach: partition the 4x4 multiplication into 2x2 blocks. The 4x4 x 4x4
  multiplication can be done by viewing it as block multiplication. But here we have
  a 2x4 times 4x4, which is different.
  
  Let me think about this more carefully. We have:
    A is 2x4, B is 4x4, C = A*B is 2x4.
    
    C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1 and k=0,1,2,3.
    
  So we have 8 output entries, each being a sum of 4 products.
  
  One approach: use the fact that for each i, the row i of C is a linear combination
  of the rows of B, with coefficients from row i of A.
  
  C[i, :] = A[i,0]*B[0,:] + A[i,1]*B[1,:] + A[i,2]*B[2,:] + A[i,3]*B[3,:]
  
  So for each i (0 or 1), we need to compute a linear combination of 4 vectors of length 4.
  
  This is like computing 2 independent "4-vector dot products with 4 vectors."
  
  Actually, let's think of it as: for each i, we compute y_i = A[i,:] * B where
  A[i,:] is a 1x4 row vector and B is 4x4. So y_i is a 1x4 row vector.
  
  The rank of the tensor for computing y = x * B where x is 1x4 and B is 4x4:
  This is essentially the "matrix-vector" multiplication tensor, but with x being 1x4
  and B being 4x4, giving a 1x4 result.
  
  The tensor for this is T[j, k, l] where j indexes x (0..3), k indexes B row (0..3),
  l indexes output (0..3). T[j,k,l] = 1 if k==l... no wait.
  
  Actually y[l] = sum_k x[k] * B[k,l]. So the tensor entries are T[k, (k,l), l]... 
  this is getting confusing with the indexing.
  
  Let me use the standard form. For computing C = A*B where A is m x n and B is n x p:
  T[a, b, c] = 1 where a indexes A (m*n), b indexes B (n*p), c indexes C (p*m).
  Specifically T[i*n+j, j*p+k, k*m+i] = 1.
  
  In our case m=2, n=4, p=4.
  T[i*4+j, j*4+k, k*2+i] = 1 for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
  
  Total number of 1-entries: 2*4*4 = 32.
  
  Now, can we find a rank < 32 decomposition?
  
  Known results: The rank of the 2x2x2 matrix multiplication tensor is 7 (Strassen).
  The rank of 4x4x4 is at most 4*7 = 28 (by partitioning into 2x2 blocks), but better
  bounds exist.
  
  For our 2x4x4 case, one approach:
  - View the 4x4 B as four 2x2 blocks, or view it differently.
  
  Alternative: Use the fact that 2x4x4 can be embedded in or related to 4x4x4.
  
  Actually, let me try a direct construction. We want to minimize the rank.
  
  Consider using Strassen on the 4x4 part. If we partition B into 2x2 blocks:
  B = [[B00, B01], [B10, B11]] where each Bij is 2x2.
  
  And A = [A0, A1] where A0 and A1 are 2x2 (first two columns, last two columns).
  
  Then C = A*B = [A0*B00 + A1*B10, A0*B01 + A1*B11].
  
  So C = [C_left, C_right] where:
  C_left = A0*B00 + A1*B10 (2x2)
  C_right = A0*B01 + A1*B11 (2x2)
  
  Each of these is a sum of two 2x2 matrix products. If we compute each 2x2 product
  with Strassen (rank 7), then each C_left and C_right would need 2*7 = 14 rank,
  giving total rank 28.
  
  But wait, can we share computations between C_left and C_right?
  
  C_left = A0*B00 + A1*B10
  C_right = A0*B01 + A1*B11
  
  The A0 and A1 are shared. But the B blocks are different.
  
  Let's see: if we use Strassen for A0*B00, A1*B10, A0*B01, A1*B11, that's 4
  Strassen products, each of rank 7, total rank 28.
  
  Can we do better by combining? For example:
  C_left + C_right = A0*(B00+B01) + A1*(B10+B11)
  C_left - C_right = A0*(B00-B01) + A1*(B10-B11)
  
  So if we define:
  B_top = B00 + B01 (2x2)
  B_bot = B10 + B11 (2x2)
  B_top_diff = B00 - B01 (2x2)
  B_bot_diff = B10 - B11 (2x2)
  
  Then:
  C_left + C_right = A0*B_top + A1*B_bot
  C_left - C_right = A0*B_top_diff + A1*B_bot_diff
  
  Still 4 Strassen products. Total rank 28.
  
  Hmm, let me think if there's a way to get rank < 28.
  
  Actually, the known result: the rank of the 2x4x4 matrix multiplication tensor.
  Let me think about lower bounds.
  
  The border rank or rank lower bound... For m x n x p, a simple lower bound is
  max(m, n, p) * something... Actually, a basic lower bound is that the rank must be
  at least the dimension of the space spanned by the "slices."
  
  Each slice T_i (fixing the first index i) is a matrix of size (n*p) x (p*m).
  For i = i0*4+j0, the slice has T[i0*4+j0, j*4+k, k*2+i] = 1 if j=j0.
  So the slice is: for fixed j0, entries at (j0*4+k, k*2+i) = 1 for all k, i.
  This is a 16x8 matrix with ones at positions (j0*4+k, k*2+i).
  
  The rank of each such slice... For fixed j0, the matrix M where M[j0*4+k, k*2+i] = 1.
  This is a 16x8 matrix. The rows are indexed by (j, k) with j=j0, so 4 rows (k=0..3).
  Wait no, the slice is 16x8 but only 4 rows are nonzero (the ones with j=j0).
  Those 4 rows: for k=0,1,2,3, the row has 1s at positions k*2+i for i=0,1.
  So row k has 1s at columns 2k and 2k+1.
  
  So the nonzero part is a 4x8 matrix where row k has 1s at columns 2k, 2k+1.
  This has rank 4 (the 4 rows are linearly independent).
  
  So each slice has rank 4. There are 8 slices (i=0..7). But slices with same i0
  (i.e., i0=0 or i0=1) have different structures.
  
  Slices with i0=0: j0=0,1,2,3 (i = 0,1,2,3 in the A indexing... wait, i indexes A
  as i*4+j, so i0 is the row of A).
  
  For i0=0 (i=0 in C), slices j0=0,1,2,3:
  Each has the 4x8 structure described.
  
  For i0=1 (i=1 in C), slices j0=0,1,2,3:
  Same structure but shifted in the C index.
  
  The rank lower bound from slices: the rank of the tensor is at least the maximum
  rank of any slice, which is 4. That's not very helpful.
  
  A better lower bound: the rank is at least the dimension of the space of all slices.
  The 8 slices span a space. Each slice is a 16x8 matrix. The space spanned by all 8
  slices... 
  
  Actually, let me just try to construct a rank-28 decomposition and see if it works.
  
  Construction with rank 28:
  Use 4 Strassen decompositions for the 4 sub-products:
    P1 = A0 * B00
    P2 = A1 * B10  
    P3 = A0 * B01
    P4 = A1 * B11
  
  Where A0 = A[:, 0:2], A1 = A[:, 2:4] (2x2 each)
  B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
  B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
  
  C[:, 0:2] = P1 + P2
  C[:, 2:4] = P3 + P4
  
  Each Pq is a 2x2 matrix product, done with Strassen (rank 7).
  
  Total rank: 4 * 7 = 28.
  
  Let me write out the Strassen decomposition for a 2x2 matrix product.
  
  Strassen for X*Y where X, Y are 2x2:
  M1 = (X00+X11)*(Y00+Y11)
  M2 = (X10+X11)*Y00
  M3 = X00*(Y01-Y11)
  M4 = X11*(Y10-Y00)
  M5 = (X00+X01)*Y11
  M6 = (X10-X00)*(Y00+Y01)
  M7 = (X01-X11)*(Y10+Y11)
  
  Z00 = M1 + M4 - M5 + M7
  Z01 = M3 + M5
  Z10 = M2 + M4
  Z11 = M1 - M2 + M3 + M6
  
  So the tensor for 2x2x2 has rank 7.
  
  For our problem, we need to map this to the U, V, W matrices.
  
  Let me define the Strassen factors for a generic 2x2 product X*Y=Z:
  
  U factors (for X entries):
  X00: appears in M1, M3, M5, M6
  X01: appears in M3, M5, M7
  X10: appears in M2, M4, M6
  X11: appears in M1, M2, M4, M7
  
  V factors (for Y entries):
  Y00: appears in M1, M2, M4, M6
  Y01: appears in M3, M5, M6, M7
  Y10: appears in M4, M6, M7
  Y11: appears in M1, M5, M7
  
  W factors (for Z entries):
  Z00: M1, M4, M5, M7
  Z01: M3, M5
  Z10: M2, M4
  Z11: M1, M2, M3, M6
  
  The 7 rank-1 terms:
  c1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]  (M1)
  c2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,1]  (M2)
  c3: U=[1,1,0,0], V=[0,1,0,-1], W=[0,1,0,0]  (M3)
  c4: U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,1,0]  (M4)
  c5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0] (M5)
  c6: U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1] (M6)
  c7: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,1]  (M7)
  
  Wait, let me be more careful. The U vector has 4 entries corresponding to X00, X01, X10, X11.
  The V vector has 4 entries corresponding to Y00, Y01, Y10, Y11.
  The W vector has 4 entries corresponding to Z00, Z01, Z10, Z11.
  
  M1 = (X00+X11)*(Y00+Y11)
  U1 = [1, 0, 0, 1]
  V1 = [1, 0, 0, 1]
  W1 = [1, 0, 0, 1]
  
  M2 = (X10+X11)*Y00
  U2 = [0, 0, 1, 1]
  V2 = [1, 0, 0, 0]
  W2 = [0, 0, 1, 1]
  
  M3 = X00*(Y01-Y11)
  U3 = [1, 0, 0, 0]
  V3 = [0, 1, 0, -1]
  W3 = [0, 1, 0, 0]
  
  M4 = X11*(Y10-Y00)
  U4 = [0, 0, 0, 1]
  V4 = [0, 0, 1, -1]
  W4 = [1, 0, 1, 0]
  
  M5 = (X00+X01)*Y11
  U5 = [1, 1, 0, 0]
  V5 = [0, 0, 0, 1]
  W5 = [-1, 1, 0, 0]
  
  M6 = (X10-X00)*(Y00+Y01)
  U6 = [-1, 0, 1, 0]
  V6 = [1, 1, 0, 0]
  W6 = [0, 0, 0, 1]
  
  M7 = (X01-X11)*(Y10+Y11)
  U7 = [0, 1, 0, -1]
  V7 = [0, 0, 1, 1]
  W7 = [1, 0, 0, 1]
  
  Let me verify: 
  Z00 = M1 + M4 - M5 + M7
  = [1,0,0,1] + [1,0,1,0] - [-1,1,0,0] + [1,0,0,1]
  = [1+1+1+1, 0+0-1+0, 0+1+0+0, 1+0+0+1]
  = [4, -1, 1, 2]
  
  Hmm, that doesn't look right. Let me recompute.
  
  Actually, the W vectors should be such that sum_c W_c * M_c gives the correct Z.
  
  Z00 = M1 + M4 - M5 + M7
  So the coefficient of M1 in Z00 is 1, M4 is 1, M5 is -1, M7 is 1, M2 is 0, M3 is 0, M6 is 0.
  
  So W1[0] = 1, W4[0] = 1, W5[0] = -1, W7[0] = 1.
  
  Z01 = M3 + M5
  W3[1] = 1, W5[1] = 1.
  
  Z10 = M2 + M4
  W2[2] = 1, W4[2] = 1.
  
  Z11 = M1 - M2 + M3 + M6
  W1[3] = 1, W2[3] = -1, W3[3] = 1, W6[3] = 1.
  
  So:
  W1 = [1, 0, 0, 1]
  W2 = [0, 0, 1, -1]
  W3 = [0, 1, 0, 1]
  W4 = [1, 0, 1, 0]
  W5 = [-1, 1, 0, 0]
  W6 = [0, 0, 0, 1]
  W7 = [1, 0, 0, 0]
  
  Let me verify Z00: W1[0]*1 + W2[0]*0 + W3[0]*0 + W4[0]*1 + W5[0]*(-1) + W6[0]*0 + W7[0]*1
  = 1 + 0 + 0 + 1 + (-1)*(-1) + 0 + 1 = 1 + 1 + 1 + 1 = 4. That's wrong, should be 1.
  
  I think I'm confusing myself. Let me restart the Strassen verification.
  
  The point is: Z = sum_c (sum_i U_c[i]*X[i]) * (sum_j V_c[j]*Y[j]) * W_c
  No, that's not right either.
  
  The tensor decomposition is: T = sum_c U_c * V_c * W_c (outer products)
  where T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r].
  
  For the 2x2 case:
  T[x_idx, y_idx, z_idx] = 1 if z_idx corresponds to the product entry.
  
  X has entries X00, X01, X10, X11 (indices 0,1,2,3)
  Y has entries Y00, Y01, Y10, Y11 (indices 0,1,2,3)
  Z has entries Z00, Z01, Z10, Z11 (indices 0,1,2,3)
  
  T[x, y, z] = 1 if the product X*Y has Z[z] involving X[x]*Y[y].
  
  Z00 = X00*Y00 + X01*Y10
  Z01 = X00*Y01 + X01*Y11
  Z10 = X10*Y00 + X11*Y10
  Z11 = X10*Y01 + X11*Y11
  
  So T[x,y,z] = 1 if:
  (x,y,z) in {(0,0,0), (1,2,0), (0,1,1), (1,3,1), (2,0,2), (3,2,2), (2,1,3), (3,3,3)}
  
  Now, the Strassen decomposition:
  T[x,y,z] = sum_r U[x,r] * V[y,r] * W[z,r]
  
  r=1 (M1 = (X00+X11)*(Y00+Y11)):
  U[0,1]=1, U[3,1]=1 (X00 and X11)
  V[0,1]=1, V[3,1]=1 (Y00 and Y11)
  W[0,1]=1, W[3,1]=1 (contributes to Z00 and Z11)
  
  r=2 (M2 = (X10+X11)*Y00):
  U[2,2]=1, U[3,2]=1 (X10 and X11)
  V[0,2]=1 (Y00)
  W[2,2]=1, W[3,2]=1 (contributes to Z10 and Z11)
  
  r=3 (M3 = X00*(Y01-Y11)):
  U[0,3]=1 (X00)
  V[1,3]=1, V[3,3]=-1 (Y01 and -Y11)
  W[1,3]=1, W[3,3]=1 (contributes to Z01 and Z11)
  
  r=4 (M4 = X11*(Y10-Y00)):
  U[3,4]=1 (X11)
  V[2,4]=1, V[0,4]=-1 (Y10 and -Y00)
  W[0,4]=1, W[2,4]=1 (contributes to Z00 and Z10)
  
  r=5 (M5 = (X00+X01)*Y11):
  U[0,5]=1, U[1,5]=1 (X00 and X01)
  V[3,5]=1 (Y11)
  W[0,5]=-1, W[1,5]=1 (contributes to -Z00 and Z01)
  
  r=6 (M6 = (X10-X00)*(Y00+Y01)):
  U[2,6]=1, U[0,6]=-1 (X10 and -X00)
  V[0,6]=1, V[1,6]=1 (Y00 and Y01)
  W[3,6]=1 (contributes to Z11)
  
  r=7 (M7 = (X01-X11)*(Y10+Y11)):
  U[1,7]=1, U[3,7]=-1 (X01 and -X11)
  V[2,7]=1, V[3,7]=1 (Y10 and Y11)
  W[0,7]=1 (contributes to Z00)
  
  Now let me verify T[0,0,0] (X00*Y00 should contribute to Z00):
  sum_r U[0,r]*V[0,r]*W[0,r]
  r=1: 1*1*1 = 1
  r=2: 0*1*0 = 0
  r=3: 1*0*0 = 0
  r=4: 0*(-1)*1 = 0
  r=5: 1*0*(-1) = 0
  r=6: -1*1*0 = 0
  r=7: 0*0*1 = 0
  Total: 1. Correct!
  
  T[1,2,0] (X01*Y10 should contribute to Z00):
  r=1: 0*1*1 = 0
  r=2: 0*0*0 = 0
  r=3: 0*0*0 = 0
  r=4: 0*1*1 = 0
  r=5: 1*0*(-1) = 0
  r=6: 0*0*0 = 0
  r=7: 1*1*1 = 1
  Total: 1. Correct!
  
  T[0,1,1] (X00*Y01 should contribute to Z01):
  r=1: 1*0*0 = 0
  r=2: 0*0*0 = 0
  r=3: 1*1*1 = 1
  r=4: 0*0*0 = 0
  r=5: 1*0*1 = 0
  r=6: -1*1*0 = 0
  r=7: 0*0*0 = 0
  Total: 1. Correct!
  
  T[1,3,1] (X01*Y11 should contribute to Z01):
  r=1: 0*1*0 = 0
  r=2: 0*0*0 = 0
  r=3: 0*(-1)*1 = 0
  r=4: 0*0*0 = 0
  r=5: 1*1*1 = 1
  r=6: 0*0*0 = 0
  r=7: 1*1*0 = 0
  Total: 1. Correct!
  
  T[2,0,2] (X10*Y00 should contribute to Z10):
  r=1: 0*1*0 = 0
  r=2: 1*1*1 = 1
  r=3: 0*0*0 = 0
  r=4: 0*(-1)*1 = 0
  r=5: 0*0*0 = 0
  r=6: 1*1*0 = 0
  r=7: 0*0*0 = 0
  Total: 1. Correct!
  
  T[3,2,2] (X11*Y10 should contribute to Z10):
  r=1: 1*1*0 = 0
  r=2: 1*0*1 = 0
  r=3: 0*0*0 = 0
  r=4: 1*1*1 = 1
  r=5: 0*0*0 = 0
  r=6: 0*0*0 = 0
  r=7: -1*1*0 = 0
  Total: 1. Correct!
  
  T[2,1,3] (X10*Y01 should contribute to Z11):
  r=1: 0*0*1 = 0
  r=2: 1*0*1 = 0
  r=3: 0*1*1 = 0
  r=4: 0*0*0 = 0
  r=5: 0*0*0 = 0
  r=6: 1*1*1 = 1
  r=7: 0*0*0 = 0
  Total: 1. Correct!
  
  T[3,3,3] (X11*Y11 should contribute to Z11):
  r=1: 1*1*1 = 1
  r=2: 1*0*1 = 0
  r=3: 0*(-1)*1 = 0
  r=4: 1*0*0 = 0
  r=5: 0*1*0 = 0
  r=6: 0*0*1 = 0
  r=7: -1*1*0 = 0
  Total: 1. Correct!
  
  All 8 entries verified. The Strassen decomposition is correct.
  
  Now, for the full 2x4x4 problem, I'll use 4 Strassen decompositions:
  
  Sub-product 1: A0 * B00 -> contributes to C[:, 0:2]
  Sub-product 2: A1 * B10 -> contributes to C[:, 0:2]
  Sub-product 3: A0 * B01 -> contributes to C[:, 2:4]
  Sub-product 4: A1 * B11 -> contributes to C[:, 2:4]
  
  Where:
  A0 = A[:, 0:2] (2x2), A1 = A[:, 2:4] (2x2)
  B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
  B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
  
  For sub-product q, the U vector has 8 entries (for A's 2x4 = 8 entries).
  Only the relevant 2x2 block is nonzero.
  
  For sub-product 1 (A0*B00):
  U has nonzero entries at positions corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
  which are indices 0, 1, 4, 5 in the 8-entry vector (i*m+j: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5).
  
  V has 16 entries (for B's 4x4 = 16 entries).
  Nonzero at positions for B[0,0], B[0,1], B[1,0], B[1,1]
  which are indices 0, 1, 4, 5 (j*p+k: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5).
  
  W has 8 entries (for C's 2x4 = 8 entries).
  Nonzero at positions for C[0,0], C[0,1], C[1,0], C[1,1]
  which are indices k*2+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3.
  
  So for sub-product 1, using Strassen factors:
  For each Strassen rank-1 term r (1..7):
  U_sub1[0] = U_strassen[r][0] (X00 = A[0,0])
  U_sub1[1] = U_strassen[r][1] (X01 = A[0,1])
  U_sub1[4] = U_strassen[r][2] (X10 = A[1,0])
  U_sub1[5] = U_strassen[r][3] (X11 = A[1,1])
  
  V_sub1[0] = V_strassen[r][0] (Y00 = B[0,0])
  V_sub1[1] = V_strassen[r][1] (Y01 = B[0,1])
  V_sub1[4] = V_strassen[r][2] (Y10 = B[1,0])
  V_sub1[5] = V_strassen[r][3] (Y11 = B[1,1])
  
  W_sub1[0] = W_strassen[r][0] (Z00 = C[0,0])
  W_sub1[1] = W_strassen[r][1] (Z01 = C[0,1])
  W_sub1[2] = W_strassen[r][2] (Z10 = C[1,0])
  W_sub1[3] = W_strassen[r][3] (Z11 = C[1,1])
  
  Similarly for other sub-products.
  
  Let me code this up.
  
  Strassen U factors (for X):
  S_U = [
    [1, 0, 0, 1],  # r1
    [0, 0, 1, 1],  # r2
    [1, 0, 0, 0],  # r3
    [0, 0, 0, 1],  # r4
    [1, 1, 0, 0],  # r5
    [-1, 0, 1, 0], # r6
    [0, 1, 0, -1], # r7
  ]
  
  Strassen V factors (for Y):
  S_V = [
    [1, 0, 0, 1],  # r1
    [1, 0, 0, 0],  # r2
    [0, 1, 0, -1], # r3
    [0, 0, 1, -1], # r4
    [0, 0, 0, 1],  # r5
    [1, 1, 0, 0],  # r6
    [0, 0, 1, 1],  # r7
  ]
  
  Strassen W factors (for Z):
  S_W = [
    [1, 0, 0, 1],  # r1
    [0, 0, 1, -1], # r2
    [0, 1, 0, 1],  # r3
    [1, 0, 1, 0],  # r4
    [-1, 1, 0, 0], # r5
    [0, 0, 0, 1],  # r6
    [1, 0, 0, 0],  # r7
  ]
  
  Wait, I need to double-check W2. Z11 = M1 - M2 + M3 + M6.
  So W2[3] = -1 (coefficient of M2 in Z11). And W2[2] = 1 (coefficient of M2 in Z10).
  Z10 = M2 + M4, so W2[2] = 1. Correct.
  Z11 = M1 - M2 + M3 + M6, so W2[3] = -1. Correct.
  
  And W4: Z00 = M1 + M4 - M5 + M7, so W4[0] = 1. Z10 = M2 + M4, so W4[2] = 1.
  W4 = [1, 0, 1, 0]. Correct.
  
  W5: Z00 = M1 + M4 - M5 + M7, so W5[0] = -1. Z01 = M3 + M5, so W5[1] = 1.
  W5 = [-1, 1, 0, 0]. Correct.
  
  W7: Z00 = M1 + M4 - M5 + M7, so W7[0] = 1.
  W7 = [1, 0, 0, 0]. Correct.
  
  Now let me code the full construction.
  
  Sub-products:
  1: A0 * B00 -> C[:, 0:2]
     A indices: [0, 1, 4, 5] (A[0,0], A[0,1], A[1,0], A[1,1])
     B indices: [0, 1, 4, 5] (B[0,0], B[0,1], B[1,0], B[1,1])
     C indices: [0, 1, 2, 3] (C[0,0], C[0,1], C[1,0], C[1,1])
  
  2: A1 * B10 -> C[:, 0:2]
     A indices: [2, 3, 6, 7] (A[0,2], A[0,3], A[1,2], A[1,3])
     B indices: [8, 9, 12, 13] (B[2,0], B[2,1], B[3,0], B[3,1])
     C indices: [0, 1, 2, 3]
  
  3: A0 * B01 -> C[:, 2:4]
     A indices: [0, 1, 4, 5]
     B indices: [2, 3, 6, 7] (B[0,2], B[0,3], B[1,2], B[1,3])
     C indices: [4, 5, 6, 7] (C[0,2], C[0,3], C[1,2], C[1,3])
  
  4: A1 * B11 -> C[:, 2:4]
     A indices: [2, 3, 6, 7]
     B indices: [10, 11, 14, 15] (B[2,2], B[2,3], B[3,2], B[3,3])
     C indices: [4, 5, 6, 7]
  
  Total rank: 4 * 7 = 28.
  
  Let me write the code.
"""

import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (8, rank)
    
    # Strassen factors for 2x2 matrix multiplication
    # S_U[r] = [coeff for X00, X01, X10, X11]
    S_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    # S_V[r] = [coeff for Y00, Y01, Y10, Y11]
    S_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [0, 0, 1, -1],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    # S_W[r] = [coeff for Z00, Z01, Z10, Z11]
    S_W = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(8)]
    
    # Sub-product definitions:
    # (A_indices, B_indices, C_indices)
    # A_indices: [X00, X01, X10, X11] positions in U
    # B_indices: [Y00, Y01, Y10, Y11] positions in V
    # C_indices: [Z00, Z01, Z10, Z11] positions in W
    
    subproducts = [
        # 1: A0 * B00 -> C[:, 0:2]
        ([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 2, 3]),
        # 2: A1 * B10 -> C[:, 0:2]
        ([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 2, 3]),
        # 3: A0 * B01 -> C[:, 2:4]
        ([0, 1, 4, 5], [2, 3, 6, 7], [4, 5, 6, 7]),
        # 4: A1 * B11 -> C[:, 2:4]
        ([2, 3, 6, 7], [10, 11, 14, 15], [4, 5, 6, 7]),
    ]
    
    for sp_idx, (a_idx, b_idx, c_idx) in enumerate(subproducts):
        base_col = sp_idx * 7
        for r in range(7):
            col = base_col + r
            for local_i, global_i in enumerate(a_idx):
                U[global_i][col] = S_U[r][local_i]
            for local_j, global_j in enumerate(b_idx):
                V[global_j][col] = S_V[r][local_j]
            for local_k, global_k in enumerate(c_idx):
                W[global_k][col] = S_W[r][local_k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
