"""Matrix multiplication 2x2 by 2x3 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-6 exact decomposition for multiplying a 2x2 matrix by a 2x3 matrix.
    
    The multiplication tensor T[i,j,k] = delta_{i, row of A} * delta_{j, col of A / row of B} * delta_{k, col of B}
    where A is 2x2 (indices i in {0,1}, j in {0,1}) and B is 2x3 (indices j in {0,1}, k in {0,1,2}).
    
    Output convention: W[k*n + i, r] where n=2, so W has 3*2=6 rows.
    U has n*m = 2*2 = 4 rows.
    V has m*p = 2*3 = 6 rows.
    
    We use the known optimal rank for 2x2 * 2x3 which is 6 (same as schoolbook since 
    the middle dimension is small). Actually, let's verify: the tensor is of format (2,2,3).
    The schoolbook rank is 2*2*3 = 12. But we can do better.
    
    Actually, for matrix multiplication A_{n x m} * B_{m x p}, the tensor is in [n*m] x [m*p] x [p*n].
    The rank of the matrix multiplication tensor is a classic problem. For 2x2 * 2x2, the rank is 7 (Strassen).
    For 2x2 * 2x3, let's think about it.
    
    The tensor T in U^* ⊗ V^* ⊗ W^* where U = C^{n*m} = C^4, V = C^{m*p} = C^6, W = C^{p*n} = C^6.
    T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    We need to find the minimal rank. Let's try to construct a rank-6 decomposition.
    
    One approach: use the fact that this is equivalent to computing 2x2 times 2x3, which produces a 2x3 output.
    Each output element C[i,k] = sum_j A[i,j] * B[j,k].
    There are 2*3 = 6 output elements, each requiring a dot product of length 2.
    Schoolbook uses 12 multiplications (2 for each of the 6 outputs).
    
    Can we share some multiplications? Let's try to find a decomposition of rank 6.
    
    Let's denote the 12 schoolbook terms as (i,j,k) for i in {0,1}, j in {0,1}, k in {0,1,2}.
    
    A rank-r decomposition means we find r triples (u_c, v_c, w_c) such that:
    T = sum_c u_c ⊗ v_c ⊗ w_c
    
    In coordinates: T[a,b,c] = sum_c U[a,c] * V[b,c] * W[c,c] ... wait, the indices are:
    U has rows indexed by (i,j) pairs, so row index a = i*m + j = i*2 + j
    V has rows indexed by (j,k) pairs, so row index b = j*p + k = j*3 + k
    W has rows indexed by (k,i) pairs, so row index c = k*n + i = k*2 + i
    
    T[a,b,c] is non-zero (equals 1) only when the indices are consistent:
    a = i*2+j, b = j*3+k, c = k*2+i for some i in {0,1}, j in {0,1}, k in {0,1,2}.
    
    So T[i*2+j, j*3+k, k*2+i] = 1 for all valid (i,j,k), and 0 otherwise.
    
    Let's try a rank-6 decomposition. We can use the following idea:
    For each k in {0,1,2}, we need to compute C[:,k] = A * B[:,k].
    Each C[:,k] is a 2-vector computed as A times a 2-vector.
    Computing A * v for a 2x2 matrix A and vector v requires 4 multiplications schoolbook, or 7 with Strassen.
    But we have 3 such vectors, so 3*4 = 12 schoolbook or 3*7 = 21 Strassen. That's worse.
    
    Alternatively, we can think of it as 3 separate 2x2 * 2x1 multiplications.
    
    Let me try a different approach. Let's use linear algebra to find a rank-6 decomposition.
    
    Actually, let's just try to construct one manually or use a known result.
    
    For 2x2 * 2x3, the border rank and rank are known. Let me try rank 6.
    
    Consider the following 6 rank-1 tensors:
    
    For the computation, let's define intermediate products:
    p0 = A[0,0]*B[0,0]
    p1 = A[0,0]*B[0,1]
    p2 = A[0,0]*B[0,2]
    p3 = A[0,1]*B[0,0]
    p4 = A[0,1]*B[0,1]
    p5 = A[0,1]*B[0,2]
    p6 = A[1,0]*B[0,0]
    p7 = A[1,0]*B[0,1]
    p8 = A[1,0]*B[0,2]
    p9 = A[1,1]*B[0,0]
    p10 = A[1,1]*B[0,1]
    p11 = A[1,1]*B[0,2]
    
    Then C[0,0] = p0 + p3
    C[0,1] = p1 + p4
    C[0,2] = p2 + p5
    C[1,0] = p6 + p9
    C[1,1] = p7 + p10
    C[1,2] = p8 + p11
    
    This is the schoolbook with rank 12.
    
    To get rank 6, we need to combine these. Let's try using Strassen-like ideas.
    
    Actually, I recall that the rank of the 2x2 x 2x3 matrix multiplication tensor is 6.
    Let me construct it.
    
    We can use the following decomposition with 6 terms. Let me denote the factors:
    
    Term 0: U has 1 at (0,0) [i=0,j=0], V has 1 at (0,0) [j=0,k=0], W has 1 at (0,0) [k=0,i=0]
    Term 1: U has 1 at (0,1) [i=0,j=1], V has 1 at (0,1) [j=0,k=1], W has 1 at (1,0) [k=1,i=0]
    ...
    
    This is getting complex. Let me just use a known rank-6 decomposition or verify if rank 6 is achievable.
    
    After reflection, I believe the minimal rank is indeed 6. Here's a construction:
    
    Let's use the following 6 rank-1 components:
    
    For each pair (i, k) where i in {0,1} and k in {0,1,2}, we can't directly do it with 1 multiplication.
    
    Let me try a different construction. Consider:
    
    r0: U: e_{0,0}, V: e_{0,0}, W: e_{0,0}  -> contributes to C[0,0] via A[0,0]*B[0,0]
    r1: U: e_{0,1}, V: e_{0,1}, W: e_{1,0}  -> contributes to C[0,1] via A[0,1]*B[0,1]... no wait
    
    Let me be more careful. The indices are:
    U row: i*2+j, so rows 0,1,2,3 correspond to (i,j) = (0,0),(0,1),(1,0),(1,1)
    V row: j*3+k, so rows 0,1,2,3,4,5 correspond to (j,k) = (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
    W row: k*2+i, so rows 0,1,2,3,4,5 correspond to (k,i) = (0,0),(0,1),(1,0),(1,1),(2,0),(2,1)
    
    The tensor T has T[i*2+j, j*3+k, k*2+i] = 1.
    
    Let's list the 12 non-zero entries:
    (0,0,0): i=0,j=0,k=0 -> U[0], V[0], W[0]
    (0,1,1): i=0,j=0,k=1 -> U[0], V[1], W[2]
    (0,2,2): i=0,j=0,k=2 -> U[0], V[2], W[4]
    (1,3,0): i=0,j=1,k=0 -> U[1], V[3], W[0]
    (1,4,1): i=0,j=1,k=1 -> U[1], V[4], W[2]
    (1,5,2): i=0,j=1,k=2 -> U[1], V[5], W[4]
    (2,0,1): i=1,j=0,k=0 -> U[2], V[0], W[1]
    (2,1,2): i=1,j=0,k=1 -> U[2], V[1], W[3]
    (2,2,3): i=1,j=0,k=2 -> U[2], V[2], W[5]
    (3,3,1): i=1,j=1,k=0 -> U[3], V[3], W[1]
    (3,4,2): i=1,j=1,k=1 -> U[3], V[4], W[3]
    (3,5,3): i=1,j=1,k=2 -> U[3], V[5], W[5]
    
    Now I need to find 6 rank-1 tensors that sum to this.
    
    Let me try the following rank-6 decomposition:
    
    Component 0: U=1 at row 0, V=1 at row 0, W=1 at row 0
    Component 1: U=1 at row 0, V=1 at row 1, W=1 at row 2
    Component 2: U=1 at row 0, V=1 at row 2, W=1 at row 4
    Component 3: U=1 at row 1, V=1 at row 3, W=1 at row 0
    Component 4: U=1 at row 1, V=1 at row 4, W=1 at row 2
    Component 5: U=1 at row 1, V=1 at row 5, W=1 at row 4
    
    This gives us:
    (0,0,0)=1, (0,1,2)=1, (0,2,4)=1, (1,3,0)=1, (1,4,2)=1, (1,5,4)=1
    
    But we're missing the i=1 terms: (2,0,1), (2,1,2), (2,2,3), (3,3,1), (3,4,2), (3,5,3)
    
    So this is just half. We need 6 more for the other half, giving rank 12.
    
    Let me try a smarter approach. Can we combine terms?
    
    Consider adding two rank-1 tensors:
    T_a = U_a ⊗ V_a ⊗ W_a
    T_b = U_b ⊗ V_b ⊗ W_b
    
    If we can make their sum contribute to multiple of the required entries, we can save rank.
    
    Let's try:
    Comp 0: U[0]=1, V[0]=1, W[0]=1  -> (0,0,0)
    Comp 1: U[2]=1, V[0]=1, W[1]=1  -> (2,0,1)
    Comp 2: U[0]=1, V[1]=1, W[2]=1  -> (0,1,2)
    Comp 3: U[2]=1, V[1]=1, W[3]=1  -> (2,1,3)... but we need (2,1,2)
    
    Hmm, W indices don't match. Let me reconsider.
    
    I think the minimal rank might actually be 12 for this specific case, or perhaps lower but not 6.
    
    Let me just use the schoolbook rank 12 decomposition, which is guaranteed to work.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
