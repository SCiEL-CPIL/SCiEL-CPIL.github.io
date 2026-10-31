"""Matrix multiplication 4x4x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-49 exact decomposition of the 4x4x4 matrix multiplication tensor
    using the Strassen algorithm (rank 7 for 2x2x2) applied to 2x2 blocks.
    
    The Strassen algorithm gives rank 7 for 2x2x2, so for 4x4x4 with 2x2 blocks:
    - We split each 4x4 matrix into four 2x2 blocks
    - Apply Strassen's 7 multiplications at the block level
    - Each 2x2 multiplication is done with rank 4 (schoolbook)
    - Total rank: 7 * 4 = 28? No, that's not right.
    
    Actually, the standard way: Strassen for n=2 gives rank 7.
    For n=4, we can use Strassen recursively: 4 = 2*2, so rank = 7 * 7 = 49? No.
    
    Let me think more carefully. The matrix multiplication tensor M_{n,m,p} has rank r(n,m,p).
    Strassen showed r(2,2,2) = 7 (and it's known to be 7).
    For r(4,4,4), we can use the block structure: split 4x4 into 2x2 blocks of 2x2 matrices.
    Then M_{4,4,4} can be decomposed using 7 copies of M_{2,2,2} (from Strassen's algorithm).
    Since r(2,2,2) = 7, we get r(4,4,4) <= 7 * 7 = 49.
    
    But wait, that's not how it works. The rank of a tensor product is the product of ranks.
    If we have a decomposition of M_{4,4,4} as a sum of 7 tensor products where each factor
    involves M_{2,2,2}, then the total rank is 7 * r(2,2,2) = 7 * 7 = 49.
    
    Actually, I need to be more careful. Let me construct the decomposition explicitly.
    
    Strassen's algorithm for 2x2 matrix multiplication:
    Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    m1 = (a+d)(e+h)
    m2 = (c+d)e
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)h
    m6 = (c-a)(e+f)
    m7 = (b-d)(g+h)
    
    Then:
    AB = [[m1+m4-m5+m7, m3+m5],
          [m2+m4, m1-m2+m3+m6]]
    
    For 4x4 matrices, split into 2x2 blocks:
    A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    
    Then AB = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
               [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    Using Strassen at the block level:
    m1 = (A11+A22)(B11+B22)
    m2 = (A21+A22)B11
    m3 = A11(B12-B22)
    m4 = A22(B21-B11)
    m5 = (A11+A12)B22
    m6 = (A21-A11)(B11+B12)
    m7 = (A12-A22)(B21+B22)
    
    Then:
    C11 = m1 + m4 - m5 + m7
    C12 = m3 + m5
    C21 = m2 + m4
    C22 = m1 - m2 + m3 + m6
    
    Each mij is a product of two 2x2 matrices (with coefficients ±1, 0).
    Each 2x2 product can be done with rank 4 (schoolbook).
    
    So total rank = 7 * 4 = 28? No, that's not right either.
    
    Wait, I think I'm confusing myself. Let me reconsider.
    
    The tensor M_{4,4,4} can be viewed as a tensor in (R^4 ⊗ R^4) ⊗ (R^4 ⊗ R^4) ⊗ (R^4 ⊗ R^4).
    Using the block decomposition, we can write it as a sum of 7 terms, each of which is a
    "scaled" version of M_{2,2,2} embedded in the 4x4x4 space.
    
    More precisely, each of the 7 Strassen multiplications corresponds to multiplying two
    2x2 matrices (which are blocks of the 4x4 matrices). The product of two 2x2 matrices
    is a 2x2 matrix, which is then placed in the appropriate block of the 4x4 result.
    
    So the decomposition is:
    M_{4,4,4} = sum_{t=1}^{7} u_t ⊗ v_t ⊗ w_t
    
    where each (u_t, v_t, w_t) corresponds to one of the 7 Strassen multiplications, and
    each of these is itself a decomposition of M_{2,2,2} (with rank 4 if we use schoolbook).
    
    So the total rank is 7 * 4 = 28.
    
    But wait, is that correct? Let me verify with a small example.
    
    Actually, I think the issue is that when we do Strassen at the block level, each "multiplication"
    is of 2x2 matrices, and we need to decompose each 2x2 multiplication. If we use rank 4 for
    each 2x2 multiplication, then the total rank is 7 * 4 = 28.
    
    However, I recall that the best known upper bound for r(4,4,4) is 49 (from recursive Strassen)
    or possibly lower. Let me check: Strassen's algorithm gives r(n,n,n) <= 7^(log2(n)).
    For n=4, log2(4) = 2, so r(4,4,4) <= 7^2 = 49.
    
    But that's using recursive Strassen, which means we apply Strassen twice. The first time
    splits 4x4 into 2x2 blocks (2x2 blocks), and the second time splits each 2x2 block into
    1x1 blocks.
    
    So the rank is 7 * 7 = 49.
    
    But can we do better? If we use schoolbook for the inner 2x2 multiplications, we get
    7 * 4 = 28. But that's not valid because the inner multiplications are of 2x2 matrices,
    and we need to decompose them exactly.
    
    Wait, I think I was wrong. Let me reconsider.
    
    When we apply Strassen at the block level for 4x4 matrices, we get 7 "block multiplications"
    of 2x2 matrices. Each block multiplication is a 2x2 matrix multiplication, which has rank 4
    (using schoolbook) or rank 7 (using Strassen, but that's overkill).
    
    So if we use schoolbook for the inner 2x2 multiplications, the total rank is 7 * 4 = 28.
    
    But is that a valid decomposition? Let me think about it more carefully.
    
    The tensor M_{4,4,4} is in (R^4 ⊗ R^4) ⊗ (R^4 ⊗ R^4) ⊗ (R^4 ⊗ R^4).
    Using the block decomposition, we can identify R^4 with R^2 ⊗ R^2 (where the first R^2
    is the block row/col index and the second R^2 is the element index within the block).
    
    Then M_{4,4,4} can be written as a sum of 7 terms, each of which is of the form:
    (u_block ⊗ u_elem) ⊗ (v_block ⊗ v_elem) ⊗ (w_block ⊗ w_elem)
    
    where u_block, v_block, w_block are in R^2 (the block level) and u_elem, v_elem, w_elem
    are in R^2 (the element level).
    
    Each such term is a rank-1 tensor in the 4x4x4 space. So the total rank is the number
    of such terms.
    
    For each of the 7 Strassen block multiplications, we need to decompose the 2x2 element-level
    multiplication. If we use schoolbook (rank 4), then each block multiplication contributes
    4 rank-1 terms. So the total rank is 7 * 4 = 28.
    
    So the rank is 28, not 49.
    
    But wait, I recall that the best known upper bound for r(4,4,4) is 49, and the lower bound
    is 27. So 28 might be achievable.
    
    Let me construct the decomposition explicitly.
    """
    n, m, p = DIMENSIONS
    
    # Strassen's 7 multiplications for 2x2 block level
    # Each multiplication is of the form: (linear_combo of A blocks) * (linear_combo of B blocks)
    # The result is placed in the appropriate C block with coefficients.
    
    # Let's define the 7 Strassen multiplications at the block level.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # m1 = (A11+A22)(B11+B22)
    # m2 = (A21+A22)B11
    # m3 = A11(B12-B22)
    # m4 = A22(B21-B11)
    # m5 = (A11+A12)B22
    # m6 = (A21-A11)(B11+B12)
    # m7 = (A12-A22)(B21+B22)
    
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Each mij is a product of two 2x2 matrices. We decompose each using schoolbook (rank 4).
    # For a 2x2 matrix product X*Y where X = [[x1,x2],[x3,x4]] and Y = [[y1,y2],[y3,y4]]:
    # XY = [[x1*y1+x2*y3, x1*y2+x2*y4],
    #       [x3*y1+x4*y3, x3*y2+x4*y4]]
    # This has rank 4 (schoolbook).
    
    # So for each of the 7 block multiplications, we have 4 rank-1 terms (from schoolbook).
    # Total rank = 7 * 4 = 28.
    
    # Now let's construct the U, V, W matrices.
    # U has shape (n*m, rank) = (16, 28)
    # V has shape (m*p, rank) = (16, 28)
    # W has shape (p*n, rank) = (16, 28)
    
    # The indices:
    # A is indexed by (i, j) where i is row (0-3), j is col (0-3). Index in U: i*m+j = i*4+j.
    # B is indexed by (j, k) where j is row (0-3), k is col (0-3). Index in V: j*p+k = j*4+k.
    # C is indexed by (k, i) where k is row (0-3), i is col (0-3). Index in W: k*n+i = k*4+i.
    # Note: the output factor follows W[k*n+i, r] convention.
    
    # Let's define the block structure:
    # Block (br, bc) of A: rows br*2 to br*2+1, cols bc*2 to bc*2+1.
    # So A11: rows 0-1, cols 0-1. A12: rows 0-1, cols 2-3.
    # A21: rows 2-3, cols 0-1. A22: rows 2-3, cols 2-3.
    
    # Similarly for B and C.
    
    # For each of the 7 Strassen multiplications, we need to:
    # 1. Define the linear combinations of A blocks and B blocks.
    # 2. For each of the 4 schoolbook terms in the 2x2 product, create a rank-1 term.
    
    # Let's define the 7 Strassen multiplications as (A_combo, B_combo) pairs.
    # Each A_combo is a 2x2 matrix of coefficients for the 4 A blocks.
    # Each B_combo is a 2x2 matrix of coefficients for the 4 B blocks.
    
    # Actually, let's think of it differently. Each Strassen multiplication is of the form:
    # m_t = (sum_{br,bc} a_{t,br,bc} * A[br,bc]) * (sum_{br,bc} b_{t,br,bc} * B[br,bc])
    # where A[br,bc] and B[br,bc] are 2x2 blocks.
    
    # The product m_t is a 2x2 matrix, and it contributes to C blocks with certain coefficients.
    
    # Let me define the 7 multiplications more explicitly.
    
    # m1: A_combo = [[1, 0], [0, 1]] (i.e., A11 + A22), B_combo = [[1, 0], [0, 1]] (i.e., B11 + B22)
    # m2: A_combo = [[0, 0], [1, 1]] (i.e., A21 + A22), B_combo = [[1, 0], [0, 0]] (i.e., B11)
    # m3: A_combo = [[1, 0], [0, 0]] (i.e., A11), B_combo = [[0, 1], [0, -1]] (i.e., B12 - B22)
    # m4: A_combo = [[0, 0], [0, 1]] (i.e., A22), B_combo = [[0, 0], [1, -1]] (i.e., B21 - B11)
    # m5: A_combo = [[1, 1], [0, 0]] (i.e., A11 + A12), B_combo = [[0, 0], [0, 1]] (i.e., B22)
    # m6: A_combo = [[-1, 0], [1, 0]] (i.e., A21 - A11), B_combo = [[1, 1], [0, 0]] (i.e., B11 + B12)
    # m7: A_combo = [[0, 1], [0, -1]] (i.e., A12 - A22), B_combo = [[0, 0], [1, 1]] (i.e., B21 + B22)
    
    # The contributions to C:
    # C11 += m1 + m4 - m5 + m7
    # C12 += m3 + m5
    # C21 += m2 + m4
    # C22 += m1 - m2 + m3 + m6
    
    # Now, for each multiplication m_t, we decompose the 2x2 product using schoolbook.
    # Let X = A_combo (2x2 matrix of coefficients) and Y = B_combo (2x2 matrix of coefficients).
    # The product X*Y is a 2x2 matrix. We decompose it using schoolbook:
    # (X*Y)[r,c] = sum_{d=0}^{1} X[r,d] * Y[d,c]
    # This gives 4 terms (for r=0,1 and c=0,1, and d=0,1).
    
    # Wait, that's not right. The schoolbook decomposition of a 2x2 matrix product is:
    # For X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]]:
    # XY = [[x00*y00 + x01*y10, x00*y01 + x01*y11],
    #       [x10*y00 + x11*y10, x10*y01 + x11*y11]]
    # This is a sum of 4 rank-1 terms (each term is a product of two 2x2 matrices, but actually
    # each term is a single product of two scalars, so it's rank-1 in the tensor sense).
    
    # Actually, I think I need to be more careful. The 2x2 matrix product X*Y can be written as:
    # X*Y = sum_{d=0}^{1} (X[:,d] * Y[d,:])
    # where X[:,d] is the d-th column of X and Y[d,:] is the d-th row of Y.
    # Each term X[:,d] * Y[d,:] is a 2x2 matrix that is the outer product of a 2x1 vector and a 1x2 vector.
    # So it's rank-1 in the 2x2 matrix space.
    
    # But in the context of the 4x4x4 tensor, each such term corresponds to a rank-1 tensor
    # in the 4x4x4 space.
    
    # Let me think about this differently. The 4x4x4 tensor M_{4,4,4} has entries:
    # M[i,j,k,l,m,n] = 1 if i*l + j*m = k and j*n + l*m = ... wait, no.
    
    # Actually, the matrix multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # M[i,j,k] = 1 if the (i,j) entry of A times the (j,k) entry of B contributes to the (i,k) entry of C.
    # More precisely, M_{i,j,k} = 1 for all i,j,k (since every (i,j) of A and (j,k) of B
    # contributes to (i,k) of C).
    
    # Wait, that's not right. The tensor is:
    # M in (R^{n*m}) ⊗ (R^{m*p}) ⊗ (R^{p*n})
    # M[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries are 0.
    
    # So the tensor has n*m*p non-zero entries, each equal to 1.
    
    # A rank-r decomposition is:
    # M = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    # where u_t in R^{n*m}, v_t in R^{m*p}, w_t in R^{p*n}
    
    # For the decomposition to be exact, we need:
    # sum_{t=1}^{r} u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] = 1 for all i,j,k
    # and 0 otherwise.
    
    # Now, let's construct the decomposition using Strassen at the block level.
    
    # The 4x4 matrix A is divided into 2x2 blocks:
    # A[br*2+r, bc*2+c] for br, bc, r, c in {0, 1}
    # So the index in U is i*m+j = (br*2+r)*4 + (bc*2+c) = br*8 + r*4 + bc*2 + c
    
    # Similarly for B: index in V is j*p+k = (br*2+r)*4 + (bc*2+c) = br*8 + r*4 + bc*2 + c
    # Wait, j is the column of A / row of B, and k is the column of B.
    # So B[j,k] where j = br*2+r, k = bc*2+c. Index in V: j*p+k = (br*2+r)*4 + (bc*2+c).
    
    # And C[i,k] where i = br*2+r, k = bc*2+c. Index in W: k*n+i = (bc*2+c)*4 + (br*2+r).
    
    # Now, for each of the 7 Strassen multiplications, we have:
    # m_t = (sum_{br,bc} a_{t,br,bc} * A[br,bc]) * (sum_{br,bc} b_{t,br,bc} * B[br,bc])
    
    # where A[br,bc] is the 2x2 block of A at block position (br, bc).
    
    # The product m_t is a 2x2 matrix, and it contributes to C blocks with coefficients.
    
    # Let's define the 7 multiplications:
    # t=0: m0 = (A11+A22)(B11+B22)
    #   a_0 = [[1, 0], [0, 1]] (A11 + A22)
    #   b_0 = [[1, 0], [0, 1]] (B11 + B22)
    #   Contributes to: C11 (+1), C22 (+1)
    
    # t=1: m1 = (A21+A22)B11
    #   a_1 = [[0, 0], [1, 1]] (A21 + A22)
    #   b_1 = [[1, 0], [0, 0]] (B11)
    #   Contributes to: C21 (+1), C22 (-1)
    
    # t=2: m2 = A11(B12-B22)
    #   a_2 = [[1, 0], [0, 0]] (A11)
    #   b_2 = [[0, 1], [0, -1]] (B12 - B22)
    #   Contributes to: C12 (+1), C22 (+1)
    
    # t=3: m3 = A22(B21-B11)
    #   a_3 = [[0, 0], [0, 1]] (A22)
    #   b_3 = [[0, 0], [1, -1]] (B21 - B11)
    #   Contributes to: C11 (+1), C21 (+1)
    
    # t=4: m4 = (A11+A12)B22
    #   a_4 = [[1, 1], [0, 0]] (A11 + A12)
    #   b_4 = [[0, 0], [0, 1]] (B22)
    #   Contributes to: C11 (-1), C12 (+1)
    
    # t=5: m5 = (A21-A11)(B11+B12)
    #   a_5 = [[-1, 0], [1, 0]] (A21 - A11)
    #   b_5 = [[1, 1], [0, 0]] (B11 + B12)
    #   Contributes to: C22 (+1)
    
    # t=6: m6 = (A12-A22)(B21+B22)
    #   a_6 = [[0, 1], [0, -1]] (A12 - A22)
    #   b_6 = [[0, 0], [1, 1]] (B21 + B22)
    #   Contributes to: C11 (+1)
    
    # Now, for each multiplication m_t, we decompose the 2x2 product using schoolbook.
    # The 2x2 product X*Y where X is a 2x2 matrix of coefficients and Y is a 2x2 matrix
    # of coefficients. The product is a 2x2 matrix.
    
    # But wait, I think I'm overcomplicating this. Let me think about it in terms of the
    # actual tensor entries.
    
    # The key insight is that each of the 7 Strassen multiplications corresponds to a
    # "scaled" copy of the 2x2x2 matrix multiplication tensor, embedded in the 4x4x4 space.
    
    # For each of the 7 multiplications, we have a rank-4 decomposition (schoolbook) of the
    # 2x2x2 tensor. So the total rank is 7 * 4 = 28.
    
    # Let me construct the decomposition explicitly.
    
    # For each of the 7 multiplications t=0..6, and for each of the 4 schoolbook terms d=0..1
    # (actually, 4 terms for a 2x2 product), we create a rank-1 term.
    
    # Wait, a 2x2 matrix product has 4 terms in schoolbook:
    # (X*Y)[r,c] = sum_{d=0}^{1} X[r,d] * Y[d,c]
    # So for each (r,c) in {0,1}^2, we have 2 terms (d=0 and d=1).
    # But that's not a rank-1 decomposition. The rank-1 decomposition of a 2x2 matrix product
    # is:
    # X*Y = sum_{d=0}^{1} (X[:,d] ⊗ Y[d,:])
    # where X[:,d] is the d-th column of X (a 2x1 vector) and Y[d,:] is the d-th row of Y (a 1x2 vector).
    # Each term X[:,d] ⊗ Y[d,:] is a 2x2 matrix that is the outer product of a 2x1 and a 1x2 vector.
    # So it's rank-1 in the 2x2 matrix space.
    
    # But in the context of the 4x4x4 tensor, each such term corresponds to a rank-1 tensor.
    
    # Let me define the 4 schoolbook terms for a 2x2 product:
    # Term 0: X[:,0] ⊗ Y[0,:] = [[X[0,0]*Y[0,0], X[0,0]*Y[0,1]], [X[1,0]*Y[0,0], X[1,0]*Y[0,1]]]
    # Term 1: X[:,1] ⊗ Y[1,:] = [[X[0,1]*Y[1,0], X[0,1]*Y[1,1]], [X[1,1]*Y[1,0], X[1,1]*Y[1,1]]]
    
    # So the 2x2 product X*Y = Term 0 + Term 1.
    
    # Now, for each of the 7 Strassen multiplications, we have 2 such terms (not 4).
    # So the total rank is 7 * 2 = 14? That seems too low.
    
    # I think I'm making a mistake. Let me reconsider.
    
    # The 2x2 matrix multiplication tensor M_{2,2,2} has rank 4 (schoolbook) or rank 7 (Strassen).
    # The schoolbook decomposition is:
    # M_{2,2,2} = sum_{i=0}^{1} sum_{j=0}^{1} sum_{k=0}^{1} e_i ⊗ e_j ⊗ f_k
    # where e_i is the i-th standard basis vector in R^2 (for A), e_j is the j-th standard basis
    # vector in R^2 (for B), and f_k is the k-th standard basis vector in R^2 (for C).
    # Wait, that's not right. The tensor M_{2,2,2} has entries M[i,j,k] = 1 for all i,j,k in {0,1}.
    # So the schoolbook decomposition is:
    # M_{2,2,2} = sum_{i=0}^{1} sum_{j=0}^{1} sum_{k=0}^{1} u_{i,j,k} ⊗ v_{i,j,k} ⊗ w_{i,j,k}
    # where u_{i,j,k}[i'] = 1 if i'=i else 0, v_{i,j,k}[j'] = 1 if j'=j else 0, w_{i,j,k}[k'] = 1 if k'=k else 0.
    # But that's not a valid decomposition because the indices are mixed up.
    
    # Actually, the matrix multiplication tensor M_{n,m,p} is defined as:
    # M in (R^{n*m}) ⊗ (R^{m*p}) ⊗ (R^{p*n})
    # M[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # All other entries are 0.
    
    # So for M_{2,2,2}, the non-zero entries are:
    # M[0*2+0, 0*2+0, 0*2+0] = M[0,0,0] = 1
    # M[0*2+0, 0*2+1, 1*2+0] = M[0,1,2] = 1
    # M[0*2+1, 1*2+0, 0*2+0] = M[1,2,0] = 1
    # M[0*2+1, 1*2+1, 1*2+0] = M[1,3,2] = 1
    # M[1*2+0, 0*2+0, 0*2+1] = M[2,0,1] = 1
    # M[1*2+0, 0*2+1, 1*2+1] = M[2,1,3] = 1
    # M[1*2+1, 1*2+0, 0*2+1] = M[3,2,1] = 1
    # M[1*2+1, 1*2+1, 1*2+1] = M[3,3,3] = 1
    
    # So the schoolbook decomposition is:
    # M_{2,2,2} = sum_{i=0}^{1} sum_{j=0}^{1} sum_{k=0}^{1} u_{i,j,k} ⊗ v_{i,j,k} ⊗ w_{i,j,k}
    # where u_{i,j,k}[i*m+j'] = 1 if j'=j else 0 (wait, that's not right).
    
    # Let me redefine. For the schoolbook decomposition:
    # For each (i,j,k) in {0,1}^3:
    # u_{i,j,k} is a vector in R^{2*2} = R^4 with u_{i,j,k}[i*2+j] = 1, all other entries 0.
    # v_{i,j,k} is a vector in R^{2*2} = R^4 with v_{i,j,k}[j*2+k] = 1, all other entries 0.
    # w_{i,j,k} is a vector in R^{2*2} = R^4 with w_{i,j,k}[k*2+i] = 1, all other entries 0.
    
    # Then the sum of u_{i,j,k} ⊗ v_{i,j,k} ⊗ w_{i,j,k} over all (i,j,k) gives the tensor M_{2,2,2}.
    
    # So the rank is 8 (schoolbook), not 4. I was wrong earlier.
    
    # So for M_{4,4,4}, the schoolbook rank is 4*4*4 = 64.
    # Using Strassen at the block level, we get 7 * 8 = 56? No, that's not right either.
    
    # Let me reconsider. The Strassen algorithm for 2x2 matrices uses 7 multiplications, each of
    # which is a product of two 2x2 matrices (with linear combinations). But each "multiplication"
    # is not a single scalar multiplication; it's a 2x2 matrix multiplication.
    
    # So the rank of the 2x2x2 tensor using Strassen is 7 (not 7*4=28).
    
    # Wait, I'm confusing myself. Let me look this up.
    
    # The rank of the matrix multiplication tensor M_{n,m,p} is the minimum number of rank-1 tensors
    # needed to decompose it.
    
    # For M_{2,2,2}, the rank is 7 (Strassen). The schoolbook rank is 8.
    
    # For M_{4,4,4}, the rank is at most 49 (using recursive Strassen: 7*7=49).
    # The best known upper bound is 49, and the lower bound is 27.
    
    # So the current best known upper bound for r(4,4,4) is 49.
    
    # Let me construct the rank-49 decomposition using recursive Strassen.
    
    # Strassen for 2x2: rank 7.
    # For 4x4: split into 2x2 blocks, apply Strassen at the block level (7 block multiplications),
    # and for each block multiplication, apply Strassen again (7 scalar multiplications).
    # Total rank: 7 * 7 = 49.
    
    # Let me construct this decomposition.
    
    # First, let's define the Strassen decomposition for 2x2 matrices.
    # The 7 multiplications are:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # Where A = [[a,b],[c,d]], B = [[e,f],[g,h]].
    
    # The result is:
    # C = [[m1+m4-m5+m7, m3+m5],
    #      [m2+m4, m1-m2+m3+m6]]
    
    # Now, for the 4x4 case, we split A and B into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2.
    
    # Apply Strassen at the block level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mj is a 2x2 matrix (product of two 2x2 matrices).
    
    # Now, for each Mj, we apply Strassen again to decompose the 2x2 matrix product.
    # Each Mj = X*Y where X and Y are 2x2 matrices.
    # Using Strassen, we can decompose X*Y using 7 scalar multiplications.
    
    # So the total rank is 7 * 7 = 49.
    
    # Let me construct the U, V, W matrices.
    
    # The indices:
    # A is indexed by (i, j) where i is row (0-3), j is col (0-3). Index in U: i*4+j.
    # B is indexed by (j, k) where j is row (0-3), k is col (0-3). Index in V: j*4+k.
    # C is indexed by (i, k) where i is row (0-3), k is col (0-3). Index in W: k*4+i.
    # Wait, the output factor follows W[k*n+i, r] convention, so W[k*4+i, r].
    
    # Let's define the block structure:
    # A[br*2+r, bc*2+c] for br, bc, r, c in {0, 1}
    # Index in U: (br*2+r)*4 + (bc*2+c) = br*8 + r*4 + bc*2 + c
    
    # B[br*2+r, bc*2+c] for br, bc, r, c in {0, 1}
    # Index in V: (br*2+r)*4 + (bc*2+c) = br*8 + r*4 + bc*2 + c
    
    # C[br*2+r, bc*2+c] for br, bc, r, c in {0, 1}
    # Index in W: (bc*2+c)*4 + (br*2+r) = bc*8 + c*4 + br*2 + r
    
    # Now, for each of the 7 block-level Strassen multiplications (t=0..6), and for each of the
    # 7 element-level Strassen multiplications (s=0..6), we create a rank-1 term.
    
    # Total rank: 7 * 7 = 49.
    
    # Let's define the 7 block-level Strassen multiplications:
    # t=0: M0 = (A11+A22)(B11+B22)
    #   A_combo: A11 + A22 -> coefficients: A11:1, A22:1, A12:0, A21:0
    #   B_combo: B11 + B22 -> coefficients: B11:1, B22:1, B12:0, B21:0
    #   Contributes to: C11 (+1), C22 (+1)
    
    # t=1: M1 = (A21+A22)B11
    #   A_combo: A21 + A22 -> coefficients: A21:1, A22:1, A11:0, A12:0
    #   B_combo: B11 -> coefficients: B11:1, B12:0, B21:0, B22:0
    #   Contributes to: C21 (+1), C22 (-1)
    
    # t=2: M2 = A11(B12-B22)
    #   A_combo: A11 -> coefficients: A11:1, A12:0, A21:0, A22:0
    #   B_combo: B12 - B22 -> coefficients: B12:1, B22:-1, B11:0, B21:0
    #   Contributes to: C12 (+1), C22 (+1)
    
    # t=3: M3 = A22(B21-B11)
    #   A_combo: A22 -> coefficients: A22:1, A11:0, A12:0, A21:0
    #   B_combo: B21 - B11 -> coefficients: B21:1, B11:-1, B12:0, B22:0
    #   Contributes to: C11 (+1), C21 (+1)
    
    # t=4: M4 = (A11+A12)B22
    #   A_combo: A11 + A12 -> coefficients: A11:1, A12:1, A21:0, A22:0
    #   B_combo: B22 -> coefficients: B22:1, B11:0, B12:0, B21:0
    #   Contributes to: C11 (-1), C12 (+1)
    
    # t=5: M5 = (A21-A11)(B11+B12)
    #   A_combo: A21 - A11 -> coefficients: A21:1, A11:-1, A12:0, A22:0
    #   B_combo: B11 + B12 -> coefficients: B11:1, B12:1, B21:0, B22:0
    #   Contributes to: C22 (+1)
    
    # t=6: M6 = (A12-A22)(B21+B22)
    #   A_combo: A12 - A22 -> coefficients: A12:1, A22:-1, A11:0, A21:0
    #   B_combo: B21 + B22 -> coefficients: B21:1, B22:1, B11:0, B12:0
    #   Contributes to: C11 (+1)
    
    # Now, for each block-level multiplication t, we need to decompose the 2x2 matrix product
    # using Strassen at the element level.
    
    # Let X be the 2x2 matrix of A coefficients and Y be the 2x2 matrix of B coefficients.
    # The product X*Y is a 2x2 matrix. We decompose it using Strassen.
    
    # Strassen for 2x2:
    # m1 = (x00+x11)(y00+y11)
    # m2 = (x10+x11)y00
    # m3 = x00(y01-y11)
    # m4 = x11(y10-y00)
    # m5 = (x00+x01)y11
    # m6 = (x10-x00)(y00+y01)
    # m7 = (x01-x11)(y10+y11)
    
    # C = [[m1+m4-m5+m7, m3+m5],
    #      [m2+m4, m1-m2+m3+m6]]
    
    # So for each block-level multiplication t, we have 7 element-level multiplications.
    # Each element-level multiplication is a scalar multiplication (product of two scalars).
    
    # So the total rank is 7 * 7 = 49.
    
    # Let me construct the U, V, W matrices.
    
    # For each (t, s) where t is the block-level multiplication (0-6) and s is the element-level
    # multiplication (0-6), we create a rank-1 term.
    
    # The U vector for (t, s) has non-zero entries at the positions corresponding to the A blocks
    # involved in the t-th block-level multiplication, with coefficients from the s-th element-level
    # multiplication.
    
    # Similarly for V and W.
    
    # Let's define the 7 block-level multiplications as (A_coeffs, B_coeffs, C_coeffs) where:
    # A_coeffs is a 4x4 matrix of coefficients for the 4 A blocks (each 2x2).
    # B_coeffs is a 4x4 matrix of coefficients for the 4 B blocks.
    # C_coeffs is a 4x4 matrix of coefficients for the 4 C blocks.
    
    # Actually, let's define it more carefully.
    
    # For each block-level multiplication t, we have:
    # A_t = sum_{br,bc} a_{t,br,bc} * A[br,bc]
    # B_t = sum_{br,bc} b_{t,br,bc} * B[br,bc]
    # C_t = sum_{br,bc} c_{t,br,bc} * C[br,bc]
    
    # where A[br,bc], B[br,bc], C[br,bc] are 2x2 blocks.
    
    # The product A_t * B_t is a 2x2 matrix, and it contributes to C with coefficients c_{t,br,bc}.
    
    # Now, for each t, we decompose the 2x2 matrix product A_t * B_t using Strassen.
    
    # Let A_t = [[a00, a01], [a10, a11]] and B_t = [[b00, b01], [b10, b11]].
    # These are 2x2 matrices of coefficients (each coefficient is a scalar).
    
    # Wait, no. A_t is a 2x2 matrix where each entry is a 2x2 block of A.
    # So A_t is a 4x4 matrix (2x2 blocks of 2x2 matrices).
    
    # I think I need to be more careful with the indexing.
    
    # Let me define the decomposition in terms of the actual tensor entries.
    
    # The tensor M_{4,4,4} has entries M[i*4+j, j*4+k, k*4+i] = 1 for all i,j,k in {0,1,2,3}.
    
    # A rank-r decomposition is:
    # M = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    
    # where u_t in R^{16}, v_t in R^{16}, w_t in R^{16}.
    
    # For the decomposition to be exact, we need:
    # sum_{t=1}^{r} u_t[i*4+j] * v_t[j*4+k] * w_t[k*4+i] = 1 for all i,j,k
    # and 0 otherwise.
    
    # Now, let's construct the rank-49 decomposition using recursive Strassen.
    
    # I'll define the 7 block-level Strassen multiplications and the 7 element-level Strassen
    # multiplications, and then combine them.
    
    # Let's define the 7 block-level multiplications:
    # t=0: A0 = A11 + A22, B0 = B11 + B22, C0 = C11 + C22
    # t=1: A1 = A21 + A22, B1 = B11, C1 = C21 - C22
    # t=2: A2 = A11, B2 = B12 - B22, C2 = C12 + C22
    # t=3: A3 = A22, B3 = B21 - B11, C3 = C11 + C21
    # t=4: A4 = A11 + A12, B4 = B22, C4 = -C11 + C12
    # t=5: A5 = A21 - A11, B5 = B11 + B12, C5 = C22
    # t=6: A6 = A12 - A22, B6 = B21 + B22, C6 = C11
    
    # Where A11, A12, A21, A22 are the 2x2 blocks of A, etc.
    
    # Now, for each t, we decompose the 2x2 matrix product A_t * B_t using Strassen.
    
    # Let A_t = [[a00, a01], [a10, a11]] and B_t = [[b00, b01], [b10, b11]].
    # These are 2x2 matrices where each entry is a 2x2 block.
    
    # Wait, I think I'm overcomplicating this. Let me just construct the decomposition directly.
    
    # I'll use the following approach:
    # 1. Define the 7 block-level Strassen multiplications.
    # 2. For each block-level multiplication, define the 7 element-level Strassen multiplications.
    # 3. Combine them to get the 49 rank-1 terms.
    
    # Let's define the block-level multiplications as (A_blocks, B_blocks, C_blocks) where:
    # A_blocks is a dict mapping (br, bc) to coefficient.
    # B_blocks is a dict mapping (br, bc) to coefficient.
    # C_blocks is a dict mapping (br, bc) to coefficient.
    
    block_level = [
        # (A_blocks, B_blocks, C_blocks)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # t=0
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),  # t=1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),  # t=2
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),  # t=3
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),  # t=4
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),  # t=5
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),  # t=6
    ]
    
    # Now, for each block-level multiplication, we need to decompose the 2x2 matrix product
    # using Strassen at the element level.
    
    # Let's define the 7 element-level Strassen multiplications for a 2x2 matrix product.
    # Given X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]]:
    # m1 = (x00+x11)(y00+y11)
    # m2 = (x10+x11)y00
    # m3 = x00(y01-y11)
    # m4 = x11(y10-y00)
    # m5 = (x00+x01)y11
    # m6 = (x10-x00)(y00+y01)
    # m7 = (x01-x11)(y10+y11)
    
    # C = [[m1+m4-m5+m7, m3+m5],
    #      [m2+m4, m1-m2+m3+m6]]
    
    # So for each element-level multiplication s, we have:
    # X_s = linear combo of X entries
    # Y_s = linear combo of Y entries
    # C_s = linear combo of C entries
    
    # Let's define the 7 element-level multiplications:
    element_level = [
        # (X_coeffs, Y_coeffs, C_coeffs)
        # X_coeffs is a dict mapping (r,c) to coefficient.
        # Y_coeffs is a dict mapping (r,c) to coefficient.
        # C_coeffs is a dict mapping (r,c) to coefficient.
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # s=0: m1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),  # s=1: m2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),  # s=2: m3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),  # s=3: m4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),  # s=4: m5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),  # s=5: m6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),  # s=6: m7
    ]
    
    # Now, for each (t, s), we create a rank-1 term.
    # The U vector for (t, s) has non-zero entries at the positions corresponding to the A blocks
    # involved in the t-th block-level multiplication, with coefficients from the s-th element-level
    # multiplication.
    
    # Let's construct the U, V, W matrices.
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    for t in range(7):
        A_blocks, B_blocks, C_blocks = block_level[t]
        for s in range(7):
            X_coeffs, Y_coeffs, C_coeffs = element_level[s]
            
            # For each (br, bc) in A_blocks, and for each (r, c) in X_coeffs:
            # The A block at (br, bc) has entries A[br*2+r, bc*2+c].
            # The coefficient is A_blocks[(br,bc)] * X_coeffs[(r,c)].
            # The index in U is (br*2+r)*4 + (bc*2+c).
            
            for (br, bc), a_coeff in A_blocks.items():
                for (r, c), x_coeff in X_coeffs.items():
                    i = br * 2 + r
                    j = bc * 2 + c
                    U[i * 4 + j][col] += a_coeff * x_coeff
            
            # Similarly for V.
            for (br, bc), b_coeff in B_blocks.items():
                for (r, c), y_coeff in Y_coeffs.items():
                    j = br * 2 + r
                    k = bc * 2 + c
                    V[j * 4 + k][col] += b_coeff * y_coeff
            
            # And for W.
            for (br, bc), c_coeff in C_blocks.items():
                for (r, c), w_coeff in C_coeffs.items():
                    i = br * 2 + r
                    k = bc * 2 + c
                    W[k * 4 + i][col] += c_coeff * w_coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
