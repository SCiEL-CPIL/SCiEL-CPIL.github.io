"""Program for matrix multiplication tensor decomposition 2x3 by 3x7."""
DIMENSIONS = [2, 3, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0])
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                T[i, j, k] = val
    return np.allclose(T, 1.0)

def _strassen_2x2(U_rows, V_cols, W_rows, rank_offset, n, m, p):
    """
    Add Strassen 2x2x2 decomposition terms starting at rank_offset.
    U_rows: list of 2 row indices in U (i*m+j for i=0,1 and fixed j)
    V_cols: list of 2 column indices in V (j*p+k for fixed j and k=0,1)
    W_rows: list of 2 row indices in W (k*n+i for k=0,1 and i=0,1)
    
    Returns the number of ranks used.
    """
    # Strassen 2x2x2: 7 multiplications
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    u0, u1 = U_rows  # rows for i=0, i=1 (same j)
    v0, v1 = V_cols  # cols for k=0, k=1 (same j)
    w00, w01 = W_rows[0], W_rows[1]  # W[k*n+i]: k=0,i=0 and k=0,i=1
    w10, w11 = W_rows[2], W_rows[3]  # W[k*n+i]: k=1,i=0 and k=1,i=1
    
    # Wait, let me reconsider the indexing.
    # W index is k*n + i where n=2
    # So for k=0: W[0*2+i] = W[i] -> i=0: W[0], i=1: W[1]
    # For k=1: W[1*2+i] = W[2+i] -> i=0: W[2], i=1: W[3]
    
    # Actually, let me re-derive this more carefully.
    # The tensor is T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For a 2x2x2 block with fixed j, we have:
    #   i in {0,1}, k in {0,1}
    #   U[i*m+j, c], V[j*p+k, c], W[k*2+i, c]
    
    # Strassen computes C = A*B where A is 2x2, B is 2x2
    # A[i, k] corresponds to... hmm, this is tricky because the indices are mixed.
    
    # Let me think differently. For fixed j, the sub-problem is:
    #   C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*2+i, c]
    # This is NOT a standard matrix multiplication because W has k and i mixed.
    
    # Actually, for fixed j, we have a bilinear form:
    #   f(x, y) = sum_i sum_k x_i * y_k * W[k*2+i] * (sum_c U[i*m+j,c] * V[j*p+k,c])
    
    # This is equivalent to: let M[i,k] = sum_c U[i*m+j,c] * V[j*p+k,c]
    # Then f = sum_{i,k} M[i,k] * W[k*2+i] * x_i * y_k
    
    # So the "output" is a 2x2 matrix C where C[i,k] = M[i,k] * W[k*2+i]
    # But wait, that's not right either. The tensor value is:
    #   T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For fixed j: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*2+i,c]
    
    # This is a 3-tensor with indices (i, k, c) where the "matrix multiplication" interpretation
    # is: think of U[:,j] as a 2xR matrix (rows i), V[j,:] as a 2xR matrix (cols k),
    # and W as a 4xR matrix (rows k*2+i).
    
    # The standard schoolbook would use rank 4 for this 2x2x2 block:
    #   T[i,j,k] = sum_{c=0}^{3} U[i*m+j, c] * V[j*p+k, c] * W[k*2+i, c]
    # where U[i*m+j, i*2+k] = 1, V[j*p+k, i*2+k] = 1, W[k*2+i, i*2+k] = 1
    
    # Strassen can reduce this to rank 7 for 2x2x2.
    # The key insight: for fixed j, define:
    #   A[i, k] = U[i*m+j, :]  (row i of U for this j)
    #   B[i, k] = V[j*p+k, :]  (col k of V for this j)
    #   W_mat[k*2+i, :] = W[k*2+i, :]
    
    # We want: T[i,j,k] = <A[i,:], B[k,:], W[k*2+i,:]> summed over c
    
    # This is equivalent to computing C = A^T * B where A is R x 2 (transposed), B is R x 2
    # No wait, let me just use the standard Strassen approach.
    
    # For the 2x2x2 tensor with T[i,k] = sum_c a_i(c) * b_k(c) * w_{k,i}(c)
    # where a_i = U[i*m+j,:], b_k = V[j*p+k,:], w_{k,i} = W[k*2+i,:]
    
    # Standard Strassen for 2x2 matrix mult C = A*B:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # In our case, for fixed j, we can define:
    # "A" matrix: A[i, ?] -- but the structure is different.
    
    # Let me just directly construct the rank-7 Strassen decomposition.
    # The 7 products are:
    # P1 = (a00 + a11)(b00 + b11)  -> contributes to c00, c11
    # P2 = (a10 + a11)b00           -> contributes to c10
    # P3 = a00(b10 - b11)           -> contributes to c00, c01
    # P4 = a11(b01 - b00)           -> contributes to c00, c11
    # P5 = (a00 + a01)b11           -> contributes to c01, c11
    # P6 = (a10 - a00)(b00 + b01)   -> contributes to c11
    # P7 = (a01 - a11)(b10 + b11)   -> contributes to c00, c10
    
    # But in our tensor, the "output" C[i,k] is not directly a matrix product.
    # Instead, T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*2+i,c]
    
    # Let me define for fixed j:
    # u_i = U[i*m+j, :]  (vector of length R)
    # v_k = V[j*p+k, :]  (vector of length R)
    # w_{k,i} = W[k*2+i, :]  (vector of length R)
    
    # T[i,j,k] = sum_c u_i[c] * v_k[c] * w_{k,i}[c]
    
    # This is a 3-tensor. To apply Strassen, I need to recognize it as a matrix mult.
    # Note that if we define:
    #   X[i, k] = u_i  (so X is a 2 x R matrix, but each "entry" is a vector)
    #   Y[k, i] = v_k * w_{k,i} (element-wise product)
    # Then T[i,j,k] = <X[i,:], Y[k,i,:]> ... this is getting complicated.
    
    # Alternative approach: just use the known Strassen 2x2x2 tensor decomposition.
    # The 2x2x2 matrix multiplication tensor has rank 7 (Strassen).
    # The tensor is: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c] where the indices
    # correspond to (row of A, col of A / row of B, col of B).
    
    # In our problem, for fixed middle index j, we have a 2x2x2 sub-tensor with:
    #   first index: i (0,1) - row of "A"
    #   second index: k (0,1) - col of "B"  
    #   The "W" part mixes i and k: W[k*2+i]
    
    # Actually, I think the cleanest approach is:
    # For each fixed j, the sub-problem is to compute a 2x2 "matrix" C where:
    #   C[i,k] = T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*2+i,c]
    
    # This IS a 2x2x2 tensor. The standard Strassen decomposition applies.
    # Let me denote:
    #   a_i = U[i*m+j, :]  for i=0,1
    #   b_k = V[j*p+k, :]  for k=0,1
    #   w_{ki} = W[k*2+i, :] for k=0,1, i=0,1
    
    # The tensor is T[i,k] = sum_c a_i[c] * b_k[c] * w_{ki}[c]
    
    # For Strassen, we need to express this as a matrix multiplication.
    # Note: if we define matrices A and B (2x2) with:
    #   A[i, ?] and B[?, k]
    # The issue is that w_{ki} depends on both i and k.
    
    # Actually, I realize the standard Strassen applies to T[i,j,k] where
    # i indexes rows of A, j indexes cols of A/rows of B, k indexes cols of B.
    # In our case, for fixed middle j, we have indices i (row of A) and k (col of B),
    # and the "W" tensor W[k*2+i] plays the role of the output coefficient.
    
    # Let me just directly write out the rank-7 Strassen terms.
    # The 2x2x2 multiplication tensor can be written as:
    # T = sum_{s=1}^{7} u_s * v_s * w_s
    # where u_s, v_s, w_s are vectors in R^2, R^2, R^4 respectively.
    
    # Standard Strassen:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # Strassen products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(g-h)
    # p4 = d(f-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    # In our tensor, for fixed j:
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*2+i,c]
    
    # Map: i -> row of A (a,c for i=0,1), k -> col of B (e,g for k=0,1)
    # The "B" matrix is transposed in some sense.
    
    # Let me define:
    # A[0,0] = U[0*m+j], A[0,1] = U[0*m+j] ... no, A should be 2x2.
    
    # I think the issue is that in standard matrix mult, T[i,j,k] = sum_c A[i,c]*B[c,k]*1
    # but in our case, the "W" part is W[k*2+i] which is the output coefficient.
    
    # Let me just directly construct it. For the 2x2x2 sub-tensor (fixed j),
    # I'll use 7 terms. Let me define the 7 rank-1 tensors:
    
    # Term 1: p1 = (a+d)(e+h)
    #   U contribution: (a+d) for row 0, (a+d) for row 1 -> [1, 1] in U-space
    #   V contribution: (e+h) for col 0, (e+h) for col 1 -> [1, 1] in V-space
    #   W contribution: needs to give C11 += 1, C12 += 0, C21 += 0, C22 += 1
    #   So W = [1, 0, 0, 1] (for indices k*2+i: (0,0)=1, (0,1)=0, (1,0)=0, (1,1)=1)
    
    # Hmm, this is getting complex. Let me just use a known construction.
    
    # Actually, for the purposes of this problem, let me use a simpler approach:
    # Split the 3x7 into a 2x7 and 1x7 part.
    # Use Strassen for the 2x2x7 part (splitting 7 into 2+2+2+1)
    # and schoolbook for the remaining 1x7 part.
    
    # 2x2x2 Strassen: rank 7
    # 2x2x1 schoolbook: rank 2
    # 1x2x1 schoolbook: rank 1 (for the remaining 1x1)
    # 1x1x7 schoolbook: rank 7
    
    # Total: 7*2 + 7 + 7 = 14 + 7 + 7 = 28? That doesn't seem right.
    
    # Let me reconsider. The matrix is 2x3 times 3x7.
    # Split the middle dimension 3 into 2+1:
    #   C = A1*B1 + A2*B2
    # where A1 is 2x2, B1 is 2x7, A2 is 2x1, B2 is 1x7.
    
    # A1*B1: 2x2 * 2x7. Split 7 into 2+2+2+1:
    #   A1*B11 (2x2*2x2): Strassen rank 7
    #   A1*B12 (2x2*2x2): Strassen rank 7
    #   A1*B13 (2x2*2x2): Strassen rank 7
    #   A1*B14 (2x2*2x1): schoolbook rank 4
    #   Subtotal: 7+7+7+4 = 25
    
    # A2*B2: 2x1 * 1x7: schoolbook rank 14
    
    # Total: 25 + 14 = 39
    
    # That gives rank 39! Let me implement this.
    
    return 0  # placeholder, will be restructured

def construct():
    """
    Returns a valid tensor decomposition for matrix multiplication 2x3 * 3x7
    using a hybrid Strassen/Schoolbook approach with rank 39.
    
    Strategy: Split middle dimension 3 into 2+1.
    - For the 2x2 * 2x7 part: split 7 into 2+2+2+1
      * Three 2x2*2x2 blocks: Strassen (rank 7 each) = 21
      * One 2x2*2x1 block: schoolbook (rank 4)
    - For the 2x1 * 1x7 part: schoolbook (rank 14)
    Total rank: 21 + 4 + 14 = 39
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 6 x 39
    V = [[0] * rank for _ in range(m * p)]  # 21 x 39
    W = [[0] * rank for _ in range(p * n)]  # 14 x 39
    
    c = 0  # current rank offset
    
    # Part 1: A1 (2x2) * B1 (2x7), where A1 uses rows i=0,1 and cols j=0,1 of A
    #         B1 uses rows j=0,1 and cols k=0..6 of B
    # Split B1's columns into 4 groups: k={0,1}, {2,3}, {4,5}, {6}
    
    # For each 2x2 * 2x2 block, use Strassen (rank 7)
    # Strassen for 2x2*2x2:
    #   Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    #   p1 = (a+d)(e+h)
    #   p2 = (c+d)e
    #   p3 = a(g-h)
    #   p4 = d(f-e)
    #   p5 = (a+b)h
    #   p6 = (c-a)(e+f)
    #   p7 = (b-d)(g+h)
    #   C11 = p1 + p4 - p5 + p7
    #   C12 = p3 + p5
    #   C21 = p2 + p4
    #   C22 = p1 - p2 + p3 + p6
    
    # In our tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For a 2x2*2x2 block with i in {0,1}, j in {0,1}, k in {k0, k0+1}:
    #   U[i*m+j, c] is the "A" coefficient
    #   V[j*p+k, c] is the "B" coefficient
    #   W[k*n+i, c] is the "output" coefficient
    
    # Strassen terms for A(i,j) * B(j,k) -> C(i,k):
    # The "A" matrix is A[i,j] = U[i*3+j, c] (for i,j in {0,1})
    # The "B" matrix is B[j,k] = V[j*7+k, c] (for j in {0,1}, k in {k0,k0+1})
    # The "C" output is C[i,k] = W[k*2+i, c] (for i in {0,1}, k in {k0,k0+1})
    
    # Wait, W index is k*n+i = k*2+i, but k here is the global k, not local.
    # For local k'=0,1 corresponding to global k=k0, k0+1:
    #   W index = (k0+k')*2 + i
    
    # Let me define a helper to add Strassen terms for a 2x2*2x2 block.
    
    def add_strassen_2x2x2(j0, k0, rank_offset):
        """
        Add Strassen 2x2x2 terms for A[i, j0..j0+1] * B[j0..j0+1, k0..k0+1] -> C[i, k0..k0+1]
        where i in {0,1}, j in {j0, j0+1}, k in {k0, k0+1}.
        Returns new rank offset.
        """
        # A[i,j] = U[i*3+j, c]
        # B[j,k] = V[j*7+k, c]
        # C[i,k] = W[k*2+i, c]
        
        # Strassen 7 terms:
        # p1 = (A00+A11)(B00+B11)
        # p2 = (A10+A11)B00
        # p3 = A00(B10-B11)
        # p4 = A11(B01-B00)
        # p5 = (A00+A01)B11
        # p6 = (A10-A00)(B00+B01)
        # p7 = (A01-A11)(B10+B11)
        
        # Where Aij = U[i*3+j0+j, :], Bij = V[(j0+j)*7+(k0+i), :]
        # And C output: W[(k0+i)*2+i', :] for i' in {0,1}
        
        # Let me be more careful with indices.
        # A[0,0] = U[0*3+j0, c] = U[j0, c]
        # A[0,1] = U[0*3+j0+1, c] = U[j0+1, c]
        # A[1,0] = U[1*3+j0, c] = U[3+j0, c]
        # A[1,1] = U[1*3+j0+1, c] = U[3+j0+1, c]
        
        # B[0,0] = V[j0*7+k0, c]
        # B[0,1] = V[j0*7+k0+1, c]
        # B[1,0] = V[(j0+1)*7+k0, c]
        # B[1,1] = V[(j0+1)*7+k0+1, c]
        
        # C[0,0] = W[k0*2+0, c] = W[2*k0, c]
        # C[0,1] = W[(k0+1)*2+0, c] = W[2*k0+2, c]
        # C[1,0] = W[k0*2+1, c] = W[2*k0+1, c]
        # C[1,1] = W[(k0+1)*2+1, c] = W[2*k0+3, c]
        
        terms = []
        
        # p1 = (A00+A11)(B00+B11)
        # U: [1, 0, 1, 0] for rows [j0, j0+1, 3+j0, 3+j0+1]
        # V: [1, 1, 1, 1] for cols [j0*7+k0, j0*7+k0+1, (j0+1)*7+k0, (j0+1)*7+k0+1]
        # W: C11 += 1, C22 += 1 -> [1, 0, 0, 1] for W[2*k0, 2*k0+1, 2*k0+2, 2*k0+3]
        # Wait: C[0,0] = W[2*k0], C[0,1] = W[2*k0+2], C[1,0] = W[2*k0+1], C[1,1] = W[2*k0+3]
        # C11 means i=0,k=0 -> W[2*k0+0] = W[2*k0]
        # C12 means i=0,k=1 -> W[2*(k0+1)+0] = W[2*k0+2]
        # C21 means i=1,k=0 -> W[2*k0+1]
        # C22 means i=1,k=1 -> W[2*k0+3]
        
        # p1 contributes to C11 and C22: W = [1, 0, 0, 1] (indices 2k0, 2k0+1, 2k0+2, 2k0+3)
        terms.append({
            'U': {j0: 1, 3+j0+1: 1},  # A00 + A11
            'V': {j0*7+k0: 1, (j0+1)*7+k0+1: 1},  # B00 + B11
            'W': {2*k0: 1, 2*k0+3: 1}  # C11 + C22
        })
        
        # p2 = (A10+A11)B00
        # U: [0, 0, 1, 1] for rows [j0, j0+1, 3+j0, 3+j0+1]
        # V: [1, 0, 0, 0] for cols [j0*7+k0, j0*7+k0+1, (j0+1)*7+k0, (j0+1)*7+k0+1]
        # W: C21 += 1 -> W[2*k0+1]
        terms.append({
            'U': {3+j0: 1, 3+j0+1: 1},  # A10 + A11
            'V': {j0*7+k0: 1},  # B00
            'W': {2*k0+1: 1}  # C21
        })
        
        # p3 = A00(B10-B11)
        # U: [1, 0, 0, 0]
        # V: [0, -1, 1, 0] for cols [j0*7+k0, j0*7+k0+1, (j0+1)*7+k0, (j0+1)*7+k0+1]
        # W: C12 += 1, C11 += 1 -> W[2*k0+2] + W[2*k0]
        # Wait: p3 = A00*(B10-B11)
        # B10 = V[(j0+1)*7+k0], B11 = V[(j0+1)*7+k0+1]
        # This contributes to C[i,k] where... 
        # In Strassen: p3 = a*(g-h) where a=A00, g=B10, h=B11
        # C11 += p3, C12 += p3
        # So W: [1, 0, 1, 0] for [2k0, 2k0+1, 2k0+2, 2k0+3]
        terms.append({
            'U': {j0: 1},  # A00
            'V': {(j0+1)*7+k0: 1, (j0+1)*7+k0+1: -1},  # B10 - B11
            'W': {2*k0: 1, 2*k0+2: 1}  # C11 + C12
        })
        
        # p4 = A11(B01-B00)
        # U: [0, 0, 0, 1]
        # V: [-1, 1, 0, 0] for cols [j0*7+k0, j0*7+k0+1, (j0+1)*7+k0, (j0+1)*7+k0+1]
        # B00 = V[j0*7+k0], B01 = V[j0*7+k0+1]
        # C11 += p4, C22 += p4
        # W: [1, 0, 0, 1]
        terms.append({
            'U': {3+j0+1: 1},  # A11
            'V': {j0*7+k0: -1, j0*7+k0+1: 1},  # B01 - B00
            'W': {2*k0: 1, 2*k0+3: 1}  # C11 + C22
        })
        
        # p5 = (A00+A01)B11
        # U: [1, 1, 0, 0]
        # V: [0, 0, 0, 1] for cols [j0*7+k0, j0*7+k0+1, (j0+1)*7+k0, (j0+1)*7+k0+1]
        # B11 = V[(j0+1)*7+k0+1]
        # C12 += p5, C22 += p5
        # W: [0, 0, 1, 1]
        terms.append({
            'U': {j0: 1, j0+1: 1},  # A00 + A01
            'V': {(j0+1)*7+k0+1: 1},  # B11
            'W': {2*k0+2: 1, 2*k0+3: 1}  # C12 + C22
        })
        
        # p6 = (A10-A00)(B00+B01)
        # U: [-1, 0, 1, 0]
        # V: [1, 1, 0, 0]
        # C22 += p6
        # W: [0, 0, 0, 1]
        terms.append({
            'U': {j0: -1, 3+j0: 1},  # A10 - A00
            'V': {j0*7+k0: 1, j0*7+k0+1: 1},  # B00 + B01
            'W': {2*k0+3: 1}  # C22
        })
        
        # p7 = (A01-A11)(B10+B11)
        # U: [0, 1, 0, -1]
        # V: [0, 0, 1, 1]
        # C11 += p7, C21 += p7
        # W: [1, 1, 0, 0]
        terms.append({
            'U': {j0+1: 1, 3+j0+1: -1},  # A01 - A11
            'V': {(j0+1)*7+k0: 1, (j0+1)*7+k0+1: 1},  # B10 + B11
            'W': {2*k0: 1, 2*k0+1: 1}  # C11 + C21
        })
        
        for term in terms:
            for u_idx, u_val in term['U'].items():
                U[u_idx][rank_offset] += u_val
            for v_idx, v_val in term['V'].items():
                V[v_idx][rank_offset] += v_val
            for w_idx, w_val in term['W'].items():
                W[w_idx][rank_offset] += w_val
            rank_offset += 1
        
        return rank_offset
    
    # Add Strassen for three 2x2*2x2 blocks: k0=0, 2, 4
    c = add_strassen_2x2x2(j0=0, k0=0, rank_offset=c)  # 7 terms, c=7
    c = add_strassen_2x2x2(j0=0, k0=2, rank_offset=c)  # 7 terms, c=14
    c = add_strassen_2x2x2(j0=0, k0=4, rank_offset=c)  # 7 terms, c=21
    
    # Part 2: A1 (2x2) * B14 (2x1) where B14 has k=6
    # Schoolbook: 4 terms
    # T[i,j,6] = sum_c U[i*3+j,c] * V[j*7+6,c] * W[6*2+i,c] = sum_c U[i*3+j,c] * V[j*7+6,c] * W[12+i,c]
    # For i in {0,1}, j in {0,1}:
    for i in range(2):
        for j in range(2):
            U[i*3+j][c] = 1
            V[j*7+6][c] = 1
            W[12+i][c] = 1
            c += 1
    # c = 25
    
    # Part 3: A2 (2x1) * B2 (1x7) where A2 uses j=2, B2 uses j=2
    # Schoolbook: 14 terms
    # T[i,2,k] = sum_c U[i*3+2,c] * V[2*7+k,c] * W[k*2+i,c] = sum_c U[i*3+2,c] * V[14+k,c] * W[k*2+i,c]
    # For i in {0,1}, k in {0..6}:
    for i in range(2):
        for k in range(7):
            U[i*3+2][c] = 1
            V[14+k][c] = 1
            W[k*2+i][c] = 1
            c += 1
    # c = 25 + 14 = 39
    
    # Verify
    assert c == rank, f"Expected rank {rank}, got {c}"
    
    if time.time() > deadline:
        pass  # Shouldn't happen, but just in case
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][idx] = 1
                    V_fb[j*p+k][idx] = 1
                    W_fb[k*n+i][idx] = 1
                    idx += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return result
# EVOLVE-BLOCK-END

# Verification helper (not part of the required interface, but useful for local debugging)
def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies that the tensor decomposition is exact.
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = len(U[0])
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                T[i, j, k] = val
    return np.allclose(T, 1.0)

if __name__ == "__main__":
    result = construct()
    U, V, W = result["U"], result["V"], result["W"]
    n, m, p = DIMENSIONS
    if verify_decomposition(U, V, W, n, m, p):
        print("Decomposition is valid.")
    else:
        print("Decomposition is INVALID.")
    print(f"Rank: {len(U[0])}")
