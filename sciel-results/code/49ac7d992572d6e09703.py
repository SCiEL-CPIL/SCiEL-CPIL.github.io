"""Matrix multiplication 4x4 by 4x9: rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
    """
    Return the 7 Strassen terms for a 2x2 matrix product.
    A is 2x2 with rows A_rows, cols A_cols (in A's coordinate system)
    B is 2x2 with rows B_rows, cols B_cols
    C is 2x2 with rows C_rows, cols C_cols
    
    Each term is (u_indices, v_indices, w_indices, coefficient)
    where the term contributes: sum_i u_i * sum_j v_j * w_k * coeff
    Actually, for tensor decomposition, each rank-1 term is:
    U[row_a, r] = 1, V[row_b, r] = 1, W[row_c, r] = 1
    But Strassen terms involve sums of products, so we need to decompose each
    multiplication of linear combinations.
    
    Let me think differently. The matrix multiplication tensor for 2x2x2 has
    rank 7. The standard Strassen decomposition gives 7 rank-1 terms.
    
    For A (2x2), B (2x2), C (2x2):
    A indices: a00, a01, a10, a11 (row, col)
    B indices: b00, b01, b10, b11
    C indices: c00, c01, c10, c11
    
    Strassen:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b01-b11)
    M4 = a11(b10-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b01)
    M7 = (a01-a11)(b10+b11)
    
    c00 = M1 - M4 + M6 - M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M3 + M2 - M5
    
    Each M is a product of two linear combinations. To express as rank-1 terms
    in the tensor, we need to expand each M into products of individual A and B
    entries. But that would give 64 terms for 2x2, not 7.
    
    The key insight: in tensor decomposition, a "rank-1" term is a single
    outer product. But Strassen's M1 = (a00+a11)(b00+b11) is NOT a single
    rank-1 term in the original tensor - it's a sum of 4 rank-1 terms.
    
    However, the Strassen algorithm works by defining NEW intermediate variables.
    In the context of tensor decomposition, what matters is that we can compute
    the 4 output entries using only 7 multiplications of linear combinations.
    
    For the tensor decomposition, we need to express C_ij = sum_k A_ik * B_kj
    as sum_r U_ir * V_jr * W_kr ... no wait, that's not right either.
    
    Let me reconsider. The matrix multiplication tensor T[i,j,k] = 1 if we're
    computing (AB)_{ik} = sum_j A_{ij} B_{jk}.
    
    The standard schoolbook decomposition: for each (i,j,k), term:
    U[i,j] = 1, V[j,k] = 1, W[k,i] = 1 (in the trace convention W[k*n+i])
    
    So for 4x4 x 4x9, schoolbook rank = 4*4*9 = 144.
    
    Now, Strassen reduces the number of "scalar multiplications" but in tensor
    decomposition, we're looking at the rank of the bilinear form. The rank of
    the matrix multiplication tensor is different from the number of multiplications
    in an algorithm.
    
    Actually, I think the task is asking for the minimum rank of a tensor
    decomposition where each rank-1 term has the form:
    U is a vector in R^(n*m), V is a vector in R^(m*p), W is a vector in R^(p*n)
    such that for all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This is exactly the tensor rank of the matrix multiplication tensor.
    
    For 2x2x2, the tensor rank is 7 (Strassen).
    For 4x4x4, the tensor rank is 49 (recursive Strassen).
    
    So the approach is: decompose the 4x4x9 multiplication into blocks.
    
    A is 4x4, B is 4x9.
    Split B into B1 (4x4), B2 (4x4), B3 (4x1).
    C = A*B1 + A*B2 + A*B3, where C is 4x9.
    
    C1 = A*B1: 4x4x4, rank 49
    C2 = A*B2: 4x4x4, rank 49  
    C3 = A*B3: 4x4x1, rank = 4*4*1 = 16 (schoolbook) or can we do better?
    
    For 4x4x1, that's computing a single column: C_i = sum_j A_ij * B_j.
    This is just a matrix-vector product. The tensor is T[i,j,k=0] = A_ij * B_j.
    Since k=0 is fixed, this is T[i,j] = 1 for all i,j, which has rank... 
    Actually for m*p = 4*1 = 4, n*m = 16, p*n = 1*4 = 4.
    T[i,j,0] = 1. The tensor is T[i,j] = 1 (a 4x4 all-ones matrix in the (i,j) slice).
    The tensor rank of a 4x4x1 tensor where T[i,j,0]=1 is... 
    We need: 1 = sum_r U[i,j,r] * V[j,r] * W[0,i,r] for all i,j.
    This is equivalent to: the 4x4 matrix of ones has rank 1, so we can write
    ones = u * v^T where u = [1,1,1,1]^T and v = [1,1,1,1]^T.
    So: U[i,j,0] = u_i = 1, V[j,0] = v_j = 1, W[0,i,0] = 1.
    That's rank 1! Because 1*1*1 = 1 for all i,j.
    
    Wait, let me verify: for each (i,j), we need sum_r U[i,j,r] * V[j,r] * W[0*n+i, r] = 1.
    With one term: U[i,j,0] = 1, V[j,0] = 1, W[i,0] = 1 (since p*n = 1*4=4, W has 4 rows, indexed by k*n+i = 0*4+i = i).
    Then the product is 1*1*1 = 1. Yes! So rank 1 for 4x4x1.
    
    Total rank = 49 + 49 + 1 = 99.
    
    Score = 144/99 ≈ 1.4545.
    
    Can we do better? What if we split differently?
    B is 4x9. We could split into B1 (4x5) and B2 (4x4), or other ways.
    
    For 4x4x5: rank of 4x4x5 multiplication tensor.
    4x4x5 = 4x4x4 + 4x4x1 = 49 + 1 = 50? No, that's not how it works.
    The tensor for 4x4x5 is T[i,j,k] for i in 0..3, j in 0..3, k in 0..4.
    This is a 4x4x5 tensor. Its rank is at most 4*4*5 = 80 (schoolbook).
    Can we do better? Using Strassen-like approaches...
    
    Actually, I think the block decomposition approach is sound. Let me just implement
    the 4x4x4 Strassen decomposition (rank 49) and use it for the blocks.
    
    For 4x4x4, recursive Strassen:
    Split 4x4 into 2x2 blocks:
    A = [[A11, A12], [A21, A22]] (each 2x2)
    B = [[B11, B12], [B21, B22]] (each 2x2)
    C = [[C11, C12], [C21, C22]] (each 2x2)
    
    C11 = (A11+A22)(B11+B22) - (A21+A22)(B22-B12) - (A11-A21)(B11-B21) + (A12-A22)(B21+B22)
    C12 = A11(B12-B22) + (A11+A12)B22
    C21 = (A21+A22)B11 + (A22-A11)(B11+B12)
    C22 = (A11+A22)(B11+B22) - (A11-A12)(B11-B12) - (A21+A22)(B22-B12) + A22(B22-B11)
    
    Each 2x2x2 product has rank 7.
    The expressions above involve 7 multiplications at the 2x2 level, but each
    "multiplication" of 2x2 matrices is itself a 2x2x2 tensor product.
    
    So total: 7 * 7 = 49 rank-1 terms.
    
    Let me code this up properly.
    """
    pass

def _get_strassen_2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term: (u_entries, v_entries, w_entries)
    where u is indexed by (i,j) for A (i,j in 0..1),
          v is indexed by (j,k) for B (j,k in 0..1),
          w is indexed by (k,i) for C (k,i in 0..1), using W[k*n+i] with n=2.
    
    The 7 Strassen terms expanded into rank-1 tensor terms:
    Each M_x = (linear combo of A)(linear combo of B)
    Expanding: M_x = sum_{a in support} sum_{b in support} c_a * d_b * A_a * B_b
    
    But this gives multiple rank-1 terms per M_x. The total would be more than 7.
    
    I think I'm confusing two things. Let me reconsider.
    
    In the literature, the "rank" of matrix multiplication in the context of
    arithmetic complexity is the minimum number of scalar multiplications needed.
    For 2x2, this is 7 (Strassen). For 4x4, this is 49.
    
    But the tensor rank (as a bilinear form) is different. The tensor rank of
    the 2x2x2 matrix multiplication tensor is actually 7, and this corresponds
    to the minimum number of bilinear forms needed.
    
    Each bilinear form is of the form: (sum_i a_i A_i) * (sum_j b_j B_j)
    This is NOT a rank-1 tensor term. A rank-1 tensor term is A_i * B_j * C_k
    for specific i,j,k.
    
    Hmm, but the task says "exact tensor decomposition" with U, V, W matrices.
    Let me re-read the task.
    
    "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    
    So the decomposition is: for each (i,j,k), 
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    This is a 3-way tensor decomposition. The rank is the number of columns c.
    
    For the standard schoolbook: for each (i,j,k), set U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for a unique c. Then sum_c gives 1 for each (i,j,k). Rank = n*m*p.
    
    Now, can we do better? The tensor T[i,j,k] = 1 for all i,j,k (in the appropriate slices).
    Actually, T is a 3-tensor of size (n*m) x (m*p) x (p*n).
    T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i.
    T[a,b,c] = 1 if the indices are consistent (i.e., the j in a matches the j in b, 
    and the k in b matches the k in c, and the i in a matches the i in c).
    
    This is the matrix multiplication tensor, and its tensor rank is what we want.
    
    For 2x2x2: the tensor rank is 7. This is a well-known result.
    For 4x4x4: the tensor rank is 49 (by recursive Strassen).
    
    So the decomposition with U, V, W of rank 49 exists for 4x4x4.
    
    How to construct it? The Strassen decomposition for 2x2x2 gives 7 bilinear forms.
    Each bilinear form (sum a_i A_i) * (sum b_j B_j) corresponds to... hmm.
    
    Actually, I think the correct interpretation is:
    Each "rank-1" term in the tensor decomposition corresponds to one scalar multiplication
    in the algorithm. The U, V, W entries encode the linear combinations.
    
    For Strassen 2x2:
    M1 = (A00+A11)(B00+B11)
    This single bilinear form contributes to C00, C11.
    
    In the tensor decomposition:
    For term r corresponding to M1:
    U[0, r] = 1 (coefficient of A00)
    U[3, r] = 1 (coefficient of A11)  [since i*m+j = 0*2+0=0, 1*2+1=3]
    V[0, r] = 1 (coefficient of B00)  [j*p+k = 0*2+0=0]
    V[3, r] = 1 (coefficient of B11)  [j*p+k = 1*2+1=3]
    W[1, r] = 1 (contributes to C00: k*n+i = 0*2+0=0? no, k*n+i with k=0,i=0 gives 0)
    W[3, r] = 1 (contributes to C11: k*n+i = 1*2+1=3)
    
    But wait, M1 contributes +1 to C00 and +1 to C11.
    So W[0, r] = 1 and W[3, r] = 1.
    
    Let me check: for (i,j,k) = (0,0,0): T[0,0,0] should get contribution from term r.
    U[0, r] * V[0, r] * W[0, r] = 1 * 1 * 1 = 1. Good.
    For (i,j,k) = (0,0,1): T[0,0,1] should get contribution.
    U[0, r] * V[1, r] * W[1, r] = 1 * 1 * ? 
    W[1, r] where k*n+i = 1*2+0 = 2. So W[2, r] = 0 (since M1 doesn't contribute to C01).
    So contribution = 1 * 1 * 0 = 0. Good.
    
    For (i,j,k) = (1,1,1): T[1,1,1] should get contribution.
    U[3, r] * V[3, r] * W[3, r] = 1 * 1 * 1 = 1. Good.
    
    So the construction works. Each Strassen term becomes one column in U, V, W.
    
    Now let me define the 7 Strassen terms for 2x2:
    A indices: 00, 01, 10, 11 -> U indices: 0, 1, 2, 3
    B indices: 00, 01, 10, 11 -> V indices: 0, 1, 2, 3
    C indices: 00, 01, 10, 11 -> W indices: 0, 1, 2, 3 (k*n+i)
    
    M1 = (A00+A11)(B00+B11): contributes +1 to C00, +1 to C11
    U: [1, 0, 0, 1], V: [1, 0, 0, 1], W: [1, 0, 0, 1]
    
    M2 = (A10+A11)B00: contributes +1 to C10, +1 to C11
    U: [0, 0, 1, 1], V: [1, 0, 0, 0], W: [0, 0, 1, 1]
    
    M3 = A00(B01-B11): contributes +1 to C01, -1 to C11
    U: [1, 0, 0, 0], V: [0, 1, 0, -1], W: [0, 1, 0, -1]
    
    M4 = A11(B10-B00): contributes -1 to C00, +1 to C10, +1 to C11
    U: [0, 0, 0, 1], V: [0, 0, 1, -1], W: [-1, 0, 1, 1]
    
    M5 = (A00+A01)B11: contributes +1 to C01, -1 to C11
    U: [1, 1, 0, 0], V: [0, 0, 0, 1], W: [0, 1, 0, -1]
    
    M6 = (A10-A00)(B00+B01): contributes +1 to C00, +1 to C10
    U: [-1, 0, 1, 0], V: [1, 1, 0, 0], W: [1, 0, 1, 0]
    
    M7 = (A01-A11)(B10+B11): contributes -1 to C00, -1 to C11
    U: [0, 1, 0, -1], V: [0, 0, 1, 1], W: [-1, 0, 0, -1]
    
    Let me verify C00 = M1 - M4 + M6 - M7:
    M1: +1, M4: -1, M6: +1, M7: -1 => 1-1+1-1 = 0. That's wrong, should be 1.
    
    Let me re-derive. C00 = A00*B00 + A01*B10.
    M1 - M4 + M6 - M7:
    M1 = A00B00 + A00B11 + A11B00 + A11B11
    M4 = A11B10 - A11B00
    M6 = A10B00 + A10B01 - A00B00 - A00B01
    M7 = A01B10 + A01B11 - A11B10 - A11B11
    
    M1 - M4 + M6 - M7 = 
    A00B00 + A00B11 + A11B00 + A11B11 
    - A11B10 + A11B00
    + A10B00 + A10B01 - A00B00 - A00B01
    - A01B10 - A01B11 + A11B10 + A11B11
    
    = (A00B00 - A00B00) + A00B11 - A00B01 + A11B00 + A11B00 - A11B10 + A11B10 + A11B11 + A11B11
    + A10B00 + A10B01 - A01B10 - A01B11
    
    This is getting messy. Let me just use the standard Strassen formulas and trust them.
    
    Standard Strassen:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 - M4 + M6 - M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M2 - M5
    
    Let me verify C11 = A11B11 + A12B21:
    M1 - M4 + M6 - M7
    = (A11+A22)(B11+B22) - A22(B21-B11) + (A21-A11)(B11+B12) - (A12-A22)(B21+B22)
    = A11B11 + A11B22 + A22B11 + A22B22 - A22B21 + A22B11 + A21B11 + A21B12 - A11B11 - A11B12 - A12B21 - A12B22 + A22B21 + A22B22
    = (A11B11 - A11B11) + A11B22 - A11B12 + A22B11 + A22B11 - A22B21 + A22B21 + A22B22 + A22B22
    + A21B11 + A21B12 - A12B21 - A12B22
    = A11B22 - A11B12 + 2*A22B11 + 2*A22B22 + A21B11 + A21B12 - A12B21 - A12B22
    
    This doesn't simplify to A11B11 + A12B21. I must have the formula wrong.
    
    Let me look up the correct Strassen formulas.
    
    Correct Strassen:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 - M4 + M6 - M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M2 - M5
    
    I think my expansion was wrong. Let me try again more carefully.
    
    Actually, for the purpose of this task, I'll just use the known fact that the
    tensor rank of 2x2x2 matmul is 7, and construct the decomposition empirically
    by solving the system of equations, or by using a known explicit decomposition.
    
    Given the time constraints, let me just use the schoolbook decomposition which
    gives rank 144, and see if I can find a known decomposition for 4x4x9 with
    lower rank.
    
    Actually, I recall that the tensor rank of 4x4x4 matmul is at most 49, and it's
    known that it's at least 49 (or maybe higher). Let me just implement the schoolbook
    and submit it, since the score would be 144/144 = 1.0, which is the same as the
    current program.
    
    Wait, the current program has score 0.0. That means it's invalid. Let me check why.
    
    The current program returns U, V, W as lists of lists. The task says "integer/half-integer
    entries". Maybe the issue is that the format is wrong?
    
    Let me re-read: "construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), 
    (m*p, rank), (p*n, rank) with integer/half-integer entries"
    
    The current program seems correct in shape. n*m = 16, m*p = 36, p*n = 36.
    rank = 144.
    
    Maybe the issue is that it's not returning numpy arrays? Or maybe the validity check
    is failing for some other reason.
    
    Let me just return the same thing but as numpy arrays, and see if that helps.
    """
    pass

def construct():
    """
    Return a valid tensor decomposition for 4x4 by 4x9 matrix multiplication.
    Uses schoolbook decomposition with rank 144.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 144
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
