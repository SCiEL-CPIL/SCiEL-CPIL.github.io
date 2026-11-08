"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x4 * 4x6 multiplication using Kronecker product structure.
    
    We view the 4x4 middle dimension as a 2x2 block structure where each block is 2x2.
    Actually, let's think differently.
    
    The matrix multiplication tensor for (n, m, p) can be decomposed if we can write
    the multiplication as a Kronecker product of smaller multiplications.
    
    For 2 x 4 x 6:
    Let n=2, m=4, p=6.
    
    We can write m = 2 * 2 and p = 3 * 2.
    So we are multiplying (2 x 4) by (4 x 6).
    
    Consider the tensor T_{n,m,p} which has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    T_{i,j,k} = 1 if the corresponding entry in the product is A[i,j] * B[j,k].
    
    If we can write the multiplication as a Kronecker product, we can combine decompositions.
    
    Let's try: n = 2, m = 2*2, p = 3*2.
    This doesn't directly factor into a clean Kronecker product of two independent multiplications
    because the middle dimension m is shared.
    
    However, there is a known technique: if we have a decomposition of size (n1, m1, p1) with rank r1
    and a decomposition of size (n2, m2, p2) with rank r2, we can get a decomposition of
    (n1*n2, m1*m2, p1*p2) with rank r1*r2.
    
    Here we have (2, 4, 6). Can we write 2 = n1*n2, 4 = m1*m2, 6 = p1*p2?
    Options:
    1) n1=1, n2=2; m1=1, m2=4; p1=1, p2=6 -> (1,1,1) x (2,4,6). Rank 1 * rank(2,4,6). No gain.
    2) n1=2, n2=1; m1=4, m2=1; p1=6, p2=1 -> (2,4,6) x (1,1,1). No gain.
    3) n1=1, n2=2; m1=2, m2=2; p1=3, p2=2 -> (1,2,3) x (2,2,2).
       Rank = rank(1,2,3) * rank(2,2,2).
       rank(1,2,3): Multiplying 1x2 by 2x3. This is just a linear map from R^2 to R^3.
       The tensor is sum_{j=0}^1 e_{0,j} ⊗ e_{j,0} ⊗ ... wait.
       For (1, 2, 3): i=0, j in {0,1}, k in {0,1,2}.
       T_{0,j,k} = 1 if we are computing (A*B)_{0,k} = sum_j A_{0,j} B_{j,k}.
       This is just the standard basis vectors for the linear map.
       The rank of the (1, m, p) tensor is m*p because each (j,k) pair is independent?
       Let's check. The tensor is sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{0,j} ⊗ e_{j,k} ⊗ e_{k,0}? No.
       The indices are (i, j, k). The term is e_i ⊗ e_j ⊗ e_k if A_{i,j} B_{j,k} contributes to C_{i,k}.
       For (1, 2, 3):
       C_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}
       C_{0,1} = A_{0,0}B_{0,1} + A_{0,1}B_{1,1}
       C_{0,2} = A_{0,0}B_{0,2} + A_{0,1}B_{1,2}
       
       The tensor is:
       e_{0,0} ⊗ e_{0,0} ⊗ e_{0,0}  (from A_{0,0}B_{0,0})
       e_{0,0} ⊗ e_{0,1} ⊗ e_{0,0}  (from A_{0,0}B_{1,0}? No, index j=1 for B)
       
       Let's be precise.
       T = sum_{i,j,k} T_{i,j,k} e_i ⊗ e_j ⊗ e_k
       T_{i,j,k} = 1 if the entry (i,k) of the product depends on A_{i,j} and B_{j,k}.
       Yes, T_{i,j,k} = 1 for all valid i,j,k.
       
       For (1, 2, 3):
       i=0. j=0,1. k=0,1,2.
       Terms:
       j=0, k=0: e_{0,0} ⊗ e_{0,0} ⊗ e_{0,0}
       j=0, k=1: e_{0,0} ⊗ e_{0,0} ⊗ e_{0,1}
       j=0, k=2: e_{0,0} ⊗ e_{0,0} ⊗ e_{0,2}
       j=1, k=0: e_{0,0} ⊗ e_{1,0} ⊗ e_{1,0}  <-- Wait, the middle index is j.
       
       The tensor is:
       sum_{j=0}^1 sum_{k=0}^2 e_{0} ⊗ e_{j} ⊗ e_{k}  ... No, the first index is i=0.
       The vector in U is indexed by (i,j). The vector in V is indexed by (j,k). The vector in W is indexed by (k,i).
       
       For (1, 2, 3):
       U is 1*2 = 2 rows. V is 2*3 = 6 rows. W is 3*1 = 3 rows.
       The tensor is sum_{j=0}^1 sum_{k=0}^2 (e_j in U) ⊗ (e_{j,k} in V) ⊗ (e_{k,0} in W).
       Wait, U index is (i,j). Since i=0, U index is j.
       V index is (j,k).
       W index is (k,i). Since i=0, W index is (k,0).
       
       So the terms are:
       j=0, k=0: U[0], V[0], W[0]
       j=0, k=1: U[0], V[1], W[1]
       j=0, k=2: U[0], V[2], W[2]
       j=1, k=0: U[1], V[3], W[0]
       j=1, k=1: U[1], V[4], W[1]
       j=1, k=2: U[1], V[5], W[2]
       
       These 6 terms are linearly independent?
       U vectors: u0, u1.
       V vectors: v0..v5.
       W vectors: w0, w1, w2.
       
       The rank is at most 6. Can it be less?
       The matrix of the tensor flattened is 2x6x3.
       Since the support is full (all j,k pairs exist), and the indices are distinct in a way that prevents cancellation, the rank is 6.
       Actually, for (1, m, p), the rank is m*p.
       So rank(1,2,3) = 6.
       
       rank(2,2,2) = 7 (Strassen).
       
       Total rank = 6 * 7 = 42.
       
       This is a valid decomposition.
       Schoolbook rank is 2*4*6 = 48.
       42 < 48.
       
       Let's implement the Kronecker product of the two decompositions.
       
       Decomposition 1: (1, 2, 3) with rank 6.
       U1, V1, W1.
       U1 shape: (1*2, 6) = (2, 6)
       V1 shape: (2*3, 6) = (6, 6)
       W1 shape: (3*1, 6) = (3, 6)
       
       U1[j, c] = 1 if c corresponds to (j, k_c).
       V1[j*3+k, c] = 1 if c corresponds to (j_c, k).
       W1[k*1+0, c] = 1 if c corresponds to (j_c, k).
       
       Let's define the indices for (1,2,3):
       c = 0: j=0, k=0
       c = 1: j=0, k=1
       c = 2: j=0, k=2
       c = 3: j=1, k=0
       c = 4: j=1, k=1
       c = 5: j=1, k=2
       
       U1:
       Row 0 (i=0, j=0): [1, 1, 1, 0, 0, 0]
       Row 1 (i=0, j=1): [0, 0, 0, 1, 1, 1]
       
       V1:
       Row 0 (j=0, k=0): [1, 0, 0, 0, 0, 0]
       Row 1 (j=0, k=1): [0, 1, 0, 0, 0, 0]
       Row 2 (j=0, k=2): [0, 0, 1, 0, 0, 0]
       Row 3 (j=1, k=0): [0, 0, 0, 1, 0, 0]
       Row 4 (j=1, k=1): [0, 0, 0, 0, 1, 0]
       Row 5 (j=1, k=2): [0, 0, 0, 0, 0, 1]
       
       W1:
       Row 0 (k=0, i=0): [1, 0, 0, 1, 0, 0]
       Row 1 (k=1, i=0): [0, 1, 0, 0, 1, 0]
       Row 2 (k=2, i=0): [0, 0, 1, 0, 0, 1]
       
       Decomposition 2: (2, 2, 2) with rank 7 (Strassen).
       Standard Strassen decomposition for 2x2 matrix multiplication.
       Indices: i in {0,1}, j in {0,1}, k in {0,1}.
       U2 shape: (2*2, 7) = (4, 7)
       V2 shape: (2*2, 7) = (4, 7)
       W2 shape: (2*2, 7) = (4, 7)
       
       Strassen variables:
       M1 = (A00 + A11) * (B00 + B11)
       M2 = (A10 + A11) * B00
       M3 = A00 * (B01 - B11)
       M4 = A11 * (B10 - B00)
       M5 = (A00 + A01) * B11
       M6 = (A10 - A00) * (B00 + B01)
       M7 = (A01 - A11) * (B10 + B11)
       
       C00 = M1 + M4 - M5 + M7
       C01 = M3 + M5
       C10 = M2 + M4
       C11 = M1 - M2 + M3 + M6
       
       Let's map these to U, V, W.
       U index: (i, j) -> i*2 + j
       V index: (j, k) -> j*2 + k
       W index: (k, i) -> k*2 + i
       
       M1:
       A part: A00 + A11. U vectors: e_{0,0} + e_{1,1}.
       B part: B00 + B11. V vectors: e_{0,0} + e_{1,1}.
       C part: C00 + C11. W vectors: e_{0,0} + e_{1,1}.
       Wait, W index is (k, i).
       C00 is (i=0, k=0) -> W index 0*2+0 = 0.
       C11 is (i=1, k=1) -> W index 1*2+1 = 3.
       So W vector for M1 is e_0 + e_3.
       
       Let's list the 7 terms:
       
       Term 0 (M1):
       U: e_{0,0} + e_{1,1} -> [1, 0, 0, 1]
       V: e_{0,0} + e_{1,1} -> [1, 0, 0, 1]
       W: e_{0,0} + e_{1,1} -> [1, 0, 0, 1] (indices 0 and 3)
       
       Term 1 (M2):
       A part: A10 + A11. U: e_{1,0} + e_{1,1} -> [0, 0, 1, 1]
       B part: B00. V: e_{0,0} -> [1, 0, 0, 0]
       C part: C10. W: e_{0,1} -> index 0*2+1 = 1. -> [0, 1, 0, 0]
       
       Term 2 (M3):
       A part: A00. U: e_{0,0} -> [1, 0, 0, 0]
       B part: B01 - B11. V: e_{0,1} - e_{1,1} -> [0, 1, 0, -1]
       C part: C01. W: e_{1,0} -> index 1*2+0 = 2. -> [0, 0, 1, 0]
       
       Term 3 (M4):
       A part: A11. U: e_{1,1} -> [0, 0, 0, 1]
       B part: B10 - B00. V: e_{1,0} - e_{0,0} -> [-1, 0, 1, 0]
       C part: C00 + C10. W: e_{0,0} + e_{0,1} -> [1, 1, 0, 0]
       
       Term 4 (M5):
       A part: A00 + A01. U: e_{0,0} + e_{0,1} -> [1, 1, 0, 0]
       B part: B11. V: e_{1,1} -> [0, 0, 0, 1]
       C part: C00 + C01. W: e_{0,0} + e_{1,0} -> [1, 0, 1, 0]
       
       Term 5 (M6):
       A part: A10 - A00. U: e_{1,0} - e_{0,0} -> [-1, 0, 1, 0]
       B part: B00 + B01. V: e_{0,0} + e_{0,1} -> [1, 1, 0, 0]
       C part: C11. W: e_{1,1} -> [0, 0, 0, 1]
       
       Term 6 (M7):
       A part: A01 - A11. U: e_{0,1} - e_{1,1} -> [0, 1, 0, -1]
       B part: B10 + B11. V: e_{1,0} + e_{1,1} -> [0, 0, 1, 1]
       C part: C00 + C11. W: e_{0,0} + e_{1,1} -> [1, 0, 0, 1]
       
       Now we combine (1,2,3) and (2,2,2) using Kronecker product.
       The combined tensor is (1*2, 2*2, 3*2) = (2, 4, 6).
       
       The combined U has shape (2*4, 6*7) = (8, 42).
       U_combined[(i1*2+i2, j1*2+j2), c1*7+c2] = U1[i1*2+j1, c1] * U2[i2*2+j2, c2]
       Wait, the indices are:
       Original (n,m,p) = (2,4,6).
       Factor 1: (n1,m1,p1) = (1,2,3).
       Factor 2: (n2,m2,p2) = (2,2,2).
       
       U index: (i, j) where i in [0,2), j in [0,4).
       i = i1*2 + i2, where i1 in [0,1), i2 in [0,2).
       j = j1*2 + j2, where j1 in [0,2), j2 in [0,2).
       
       So U_combined[(i1*2+i2) * 4 + (j1*2+j2), c1*7+c2] = U1[i1*2+j1, c1] * U2[i2*2+j2, c2]
       
       Similarly for V:
       V index: (j, k) where j in [0,4), k in [0,6).
       j = j1*2 + j2.
       k = k1*2 + k2, where k1 in [0,3), k2 in [0,2).
       
       V_combined[(j1*2+j2) * 6 + (k1*2+k2), c1*7+c2] = V1[j1*3+k1, c1] * V2[j2*2+k2, c2]
       
       Similarly for W:
       W index: (k, i) where k in [0,6), i in [0,2).
       k = k1*2 + k2.
       i = i1*2 + i2.
       
       W_combined[(k1*2+k2) * 2 + (i1*2+i2), c1*7+c2] = W1[k1*1+i1, c1] * W2[k2*2+i2, c2]
       
       Let's construct this.
       """
    
    n, m, p = DIMENSIONS
    # Factor 1: (1, 2, 3)
    n1, m1, p1 = 1, 2, 3
    r1 = 6
    U1 = [[0]*r1 for _ in range(n1*m1)]
    V1 = [[0]*r1 for _ in range(m1*p1)]
    W1 = [[0]*r1 for _ in range(p1*n1)]
    
    # c: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # U1[i1*m1+j1, c]
    # i1=0 always.
    U1[0][0] = 1; U1[0][1] = 1; U1[0][2] = 1
    U1[1][3] = 1; U1[1][4] = 1; U1[1][5] = 1
    
    # V1[j1*p1+k1, c]
    V1[0][0] = 1
    V1[1][1] = 1
    V1[2][2] = 1
    V1[3][3] = 1
    V1[4][4] = 1
    V1[5][5] = 1
    
    # W1[k1*n1+i1, c]
    W1[0][0] = 1; W1[0][3] = 1
    W1[1][1] = 1; W1[1][4] = 1
    W1[2][2] = 1; W1[2][5] = 1
    
    # Factor 2: (2, 2, 2)
    n2, m2, p2 = 2, 2, 2
    r2 = 7
    U2 = [[0]*r2 for _ in range(n2*m2)]
    V2 = [[0]*r2 for _ in range(m2*p2)]
    W2 = [[0]*r2 for _ in range(p2*n2)]
    
    # Term 0
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    # Term 1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1
    # Term 2
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1
    # Term 3
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    # Term 4
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = 1; W2[2][4] = 1
    # Term 5
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    # Term 6
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = 1
    
    # Combine
    rank = r1 * r2
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1):
                for i2 in range(n2):
                    i = i1 * n2 + i2
                    for j1 in range(m1):
                        for j2 in range(m2):
                            j = j1 * m2 + j2
                            u_idx = i * m + j
                            u_val = U1[i1*m1+j1][c1] * U2[i2*m2+j2][c2]
                            if u_val != 0:
                                U[u_idx][c] = u_val
                            
                            for k1 in range(p1):
                                for k2 in range(p2):
                                    k = k1 * p2 + k2
                                    v_idx = j * p + k
                                    v_val = V1[j1*p1+k1][c1] * V2[j2*p2+k2][c2]
                                    if v_val != 0:
                                        V[v_idx][c] = v_val
                                    
                                    w_idx = k * n + i
                                    w_val = W1[k1*n1+i1][c1] * W2[k2*n2+i2][c2]
                                    if w_val != 0:
                                        W[w_idx][c] = w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
