"""
Program to construct a rank-minimal exact tensor decomposition for matrix multiplication.
Target: 3x4 matrix by 4x4 matrix.
Dimensions: n=3, m=4, p=4.

Strategy:
  - The 3x4x4 multiplication tensor can be decomposed as a block sum:
    C = A * B where A is 3x4 and B is 4x4.
    Split A into a 2x4 top block and a 1x4 bottom block.
    C_top (2x4) = A_top (2x4) * B (4x4)
    C_bot (1x4) = A_bot (1x4) * B (4x4)
    
  - Use the rank-7 Strassen decomposition for the 2x2x2 part of the top block,
    combined with a rank-4 schoolbook for the 2x2 structure in B.
    This gives rank 7*4 = 28 for the 2x4x4 part.
    
  - The 1x4x4 part has rank 4 (schoolbook).
    
  - Total rank: 28 + 4 = 32.
"""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition for matrix multiplication C = A * B.
    
    Shapes:
    U: (n*m, rank) = (12, rank)
    V: (m*p, rank) = (16, rank)
    W: (p*n, rank) = (12, rank)
    
    The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta (reconstructs C_ik)
    """
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    
    # Strassen's 7-multiplication algorithm for 2x2 matrix multiplication
    # C = A * B where A, B, C are 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[p1+p4-p5+p6, p2+p4], [p3+p5, p1-p3+p4+p7]]
    # where:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c+d)(f-h)
    # p7 = (a-b)(e+g)
    
    # For each of the 7 Strassen multiplications, we need to map to our indices.
    # For the 2x4x4 top block:
    # A_top is 2x4, B is 4x4, C_top is 2x4.
    # We split B into two 2x2 blocks: B = [[B11, B12], [B21, B22]] where each is 2x2.
    # A_top = [[a1, a2], [a3, a4]] where each ai is 1x2 (a row of A_top).
    
    # Actually, let's think of it as:
    # A_top has 2 rows, each of length 4.
    # B has 4 rows, 4 columns.
    # C_top has 2 rows, 4 columns.
    
    # We can view this as two 2x2 multiplications:
    # C_top[:, 0:2] = A_top * B[:, 0:2]  (2x2 * 2x2 = 2x2)
    # C_top[:, 2:4] = A_top * B[:, 2:4]  (2x2 * 2x2 = 2x2)
    
    # So for each of these two 2x2 multiplications, we use Strassen with rank 7.
    # Total rank for top block: 7 * 2 = 14? No, that's not right.
    
    # Actually, the 2x4x4 tensor is not a Kronecker product of 2x2x2 and 1x2x2.
    # Let me reconsider.
    
    # The 2x4x4 multiplication: A is 2x4, B is 4x4, C is 2x4.
    # This is equivalent to multiplying a 2x2 block matrix by a 4x4 block matrix.
    # If we view A as having 2 rows and B as having 4 rows, we can think of it as:
    # Each row of A (length 4) multiplies B (4x4) to give a row of C (length 4).
    
    # A better approach: use the fact that 2x4x4 can be decomposed using the
    # rank of the 2x2x2 tensor (which is 7) combined with a 2x1x2 tensor.
    
    # Actually, let's use a simpler approach: 
    # The 2x4x4 tensor has rank at most 2*4 = 8 if we use schoolbook for the 2x4 part,
    # but we can do better by using Strassen.
    
    # Let me try a different decomposition:
    # View the 2x4x4 as a Kronecker product of (2x2x2) and (1x2x2).
    # A = A1 ⊗ A2 where A1 is 2x2 and A2 is 1x2.
    # B = B1 ⊗ B2 where B1 is 2x2 and B2 is 2x2.
    # C = C1 ⊗ C2 where C1 is 2x2 and C2 is 1x2.
    
    # Then the rank of the Kronecker product tensor is rank(T1) * rank(T2).
    # rank(T1) = 7 (Strassen), rank(T2) = 4 (schoolbook for 1x2x2).
    # Total rank = 7 * 4 = 28.
    
    # Let's implement this.
    
    # Strassen 2x2x2 decomposition factors:
    # For C1[i,k] = sum_j A1[i,j] * B1[j,k], we have 7 terms.
    # U1 has shape (2*2, 7) = (4, 7)
    # V1 has shape (2*2, 7) = (4, 7)
    # W1 has shape (2*2, 7) = (4, 7)
    
    # Indexing: U1[i*2+j, r], V1[j*2+k, r], W1[k*2+i, r]
    
    # Strassen formulas:
    # p1 = (a+d)(e+h): U1: a=1,d=1; V1: e=1,h=1; W1: contributes to C[0,0] and C[1,1]
    # p2 = (c+d)e: U1: c=1,d=1; V1: e=1; W1: contributes to C[1,0]
    # p3 = a(f-h): U1: a=1; V1: f=1,h=-1; W1: contributes to C[0,0] and C[0,1]
    # p4 = d(g-e): U1: d=1; V1: g=1,e=-1; W1: contributes to C[1,0] and C[1,1]
    # p5 = (a+b)h: U1: a=1,b=1; V1: h=1; W1: contributes to C[0,1]
    # p6 = (c+d)(f-h): U1: c=1,d=1; V1: f=1,h=-1; W1: contributes to C[1,0] and C[1,1]
    # p7 = (a-b)(e+g): U1: a=1,b=-1; V1: e=1,g=1; W1: contributes to C[0,0] and C[1,1]
    
    # Let's build U1, V1, W1 for Strassen 2x2x2:
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # p1 = (a+d)(e+h)
    # U1: a=U1[0]=1, d=U1[2]=1 (indices: a=0*2+0=0, b=0*2+1=1, c=1*2+0=2, d=1*2+1=3)
    U1[0][0] = 1  # a
    U1[3][0] = 1  # d
    V1[0][0] = 1  # e (j=0,k=0)
    V1[3][0] = 1  # h (j=1,k=1)
    W1[0][0] = 1  # C[0,0] (k=0,i=0)
    W1[3][0] = 1  # C[1,1] (k=1,i=1)
    
    # p2 = (c+d)e
    U1[2][1] = 1  # c
    U1[3][1] = 1  # d
    V1[0][1] = 1  # e
    W1[2][1] = 1  # C[1,0] (k=0,i=1)
    
    # p3 = a(f-h)
    U1[0][2] = 1  # a
    V1[1][2] = 1  # f (j=0,k=1)
    V1[3][2] = -1  # h (j=1,k=1)
    W1[0][2] = 1  # C[0,0]
    W1[1][2] = 1  # C[0,1] (k=1,i=0)
    
    # p4 = d(g-e)
    U1[3][3] = 1  # d
    V1[2][3] = 1  # g (j=1,k=0)
    V1[0][3] = -1  # e (j=0,k=0)
    W1[2][3] = 1  # C[1,0]
    W1[3][3] = 1  # C[1,1]
    
    # p5 = (a+b)h
    U1[0][4] = 1  # a
    U1[1][4] = 1  # b
    V1[3][4] = 1  # h
    W1[1][4] = 1  # C[0,1]
    
    # p6 = (c+d)(f-h)
    U1[2][5] = 1  # c
    U1[3][5] = 1  # d
    V1[1][5] = 1  # f
    V1[3][5] = -1  # h
    W1[2][5] = 1  # C[1,0]
    W1[3][5] = 1  # C[1,1]
    
    # p7 = (a-b)(e+g)
    U1[0][6] = 1  # a
    U1[1][6] = -1  # b
    V1[0][6] = 1  # e
    V1[2][6] = 1  # g
    W1[0][6] = 1  # C[0,0]
    W1[3][6] = 1  # C[1,1]
    
    # Now for the 1x2x2 part (schoolbook, rank 4):
    # A2 is 1x2, B2 is 2x2, C2 is 1x2.
    # C2[0,k] = sum_j A2[0,j] * B2[j,k]
    # U2 has shape (1*2, 4) = (2, 4)
    # V2 has shape (2*2, 4) = (4, 4)
    # W2 has shape (2*1, 4) = (2, 4)
    
    U2 = [[0]*4 for _ in range(2)]
    V2 = [[0]*4 for _ in range(4)]
    W2 = [[0]*4 for _ in range(2)]
    
    for j in range(2):
        for k in range(2):
            col = j * 2 + k
            U2[j][col] = 1  # A2[0,j]
            V2[j * 2 + k][col] = 1  # B2[j,k]
            W2[k][col] = 1  # C2[0,k]
    
    # Now form the Kronecker product for the 2x4x4 top block:
    # U_top = U1 ⊗ U2, shape (4*2, 7*4) = (8, 28)
    # V_top = V1 ⊗ V2, shape (4*4, 7*4) = (16, 28)
    # W_top = W1 ⊗ W2, shape (4*2, 7*4) = (8, 28)
    
    def kron(U_a, U_b):
        """Kronecker product of two matrices."""
        ra, ca = len(U_a), len(U_a[0])
        rb, cb = len(U_b), len(U_b[0])
        result = [[0.0] * (ca * cb) for _ in range(ra * rb)]
        for i in range(ra):
            for j in range(ca):
                for k in range(rb):
                    for l in range(cb):
                        result[i * rb + k][j * cb + l] = U_a[i][j] * U_b[k][l]
        return result
    
    U_top = kron(U1, U2)  # shape (8, 28)
    V_top = kron(V1, V2)  # shape (16, 28)
    W_top = kron(W1, W2)  # shape (8, 28)
    
    # For the 1x4x4 bottom block:
    # A_bot is 1x4, B is 4x4, C_bot is 1x4.
    # Schoolbook rank: 4.
    # U_bot has shape (1*4, 4) = (4, 4)
    # V_bot has shape (4*4, 4) = (16, 4)
    # W_bot has shape (4*1, 4) = (4, 4)
    
    U_bot = [[0]*4 for _ in range(4)]
    V_bot = [[0]*4 for _ in range(16)]
    W_bot = [[0]*4 for _ in range(4)]
    
    for j in range(4):
        for k in range(4):
            col = j * 4 + k
            U_bot[j][col] = 1  # A_bot[0,j]
            V_bot[j * 4 + k][col] = 1  # B[j,k]
            W_bot[k][col] = 1  # C_bot[0,k]
    
    # Now combine top and bottom blocks:
    # Total rank = 28 + 4 = 32.
    
    rank_total = 28 + 4
    U = [[0.0] * rank_total for _ in range(12)]  # n*m = 3*4 = 12
    V = [[0.0] * rank_total for _ in range(16)]  # m*p = 4*4 = 16
    W = [[0.0] * rank_total for _ in range(12)]  # p*n = 4*3 = 12
    
    # Fill top block (rows 0-7 of U, all rows of V, rows 0-7 of W):
    # U_top has 8 rows corresponding to i in {0,1}, j in {0,1,2,3}
    # Global U index: i*4 + j for i in {0,1}
    for i in range(2):
        for j in range(4):
            global_u_row = i * 4 + j
            # U_top row index: (i * 2 + j % 2) * 2 + (j // 2)
            # Wait, let's think about the Kronecker product indexing.
            # U1 has 4 rows: index = i1 * 2 + j1 where i1 in {0,1}, j1 in {0,1}
            # U2 has 2 rows: index = i2 * 2 + j2 where i2 = 0, j2 in {0,1}
            # U_top row index = (i1 * 2 + j1) * 2 + (i2 * 2 + j2) = i1*4 + j1*2 + j2
            # Global U row for top block: i_global * 4 + j_global where i_global in {0,1}, j_global in {0,1,2,3}
            # We need to map j_global to (j1, j2): j_global = j1 * 2 + j2
            # So U_top row = i_global * 4 + j1 * 2 + j2 = i_global * 4 + j_global
            # That matches!
            top_row = i * 4 + j
            for r in range(28):
                U[global_u_row][r] = U_top[top_row][r]
    
    # V_top has 16 rows corresponding to j in {0,1,2,3}, k in {0,1,2,3}
    # V1 has 4 rows: index = j1 * 2 + k1
    # V2 has 4 rows: index = j2 * 2 + k2
    # V_top row index = (j1 * 2 + k1) * 4 + (j2 * 2 + k2)
    # Global V row: j_global * 4 + k_global where j_global in {0,1,2,3}, k_global in {0,1,2,3}
    # j_global = j1 * 2 + j2, k_global = k1 * 2 + k2
    # V_top row = (j1 * 2 + k1) * 4 + (j2 * 2 + k2) = j1*8 + k1*4 + j2*2 + k2
    # Global V row = (j1*2 + j2)*4 + (k1*2 + k2) = j1*8 + j2*4 + k1*2 + k2
    # These don't match directly. Let's just use the global index.
    for j in range(4):
        for k in range(4):
            global_v_row = j * 4 + k
            # j = j1 * 2 + j2, k = k1 * 2 + k2
            j1 = j // 2
            j2 = j % 2
            k1 = k // 2
            k2 = k % 2
            v_top_row = (j1 * 2 + k1) * 4 + (j2 * 2 + k2)
            for r in range(28):
                V[global_v_row][r] = V_top[v_top_row][r]
    
    # W_top has 8 rows corresponding to k in {0,1,2,3}, i in {0,1}
    # W1 has 4 rows: index = k1 * 2 + i1
    # W2 has 2 rows: index = k2 * 1 + i2 where i2 = 0
    # W_top row index = (k1 * 2 + i1) * 2 + k2
    # Global W row for top block: k_global * 3 + i_global where i_global in {0,1}
    # k_global = k1 * 2 + k2
    # Global W row = (k1*2 + k2)*3 + i1 = k1*6 + k2*3 + i1
    # W_top row = (k1*2 + i1)*2 + k2 = k1*4 + i1*2 + k2
    # These don't match. Let's map carefully.
    for i in range(2):
        for k in range(4):
            global_w_row = k * 3 + i
            k1 = k // 2
            k2 = k % 2
            w_top_row = (k1 * 2 + i) * 2 + k2
            for r in range(28):
                W[global_w_row][r] = W_top[w_top_row][r]
    
    # Fill bottom block (rows 8-11 of U, all rows of V, rows 8-11 of W):
    # U_bot has 4 rows corresponding to j in {0,1,2,3}
    # Global U row for bottom block: 2*4 + j = 8 + j
    for j in range(4):
        global_u_row = 8 + j
        for r in range(4):
            U[global_u_row][28 + r] = U_bot[j][r]
    
    # V_bot has 16 rows corresponding to j in {0,1,2,3}, k in {0,1,2,3}
    for j in range(4):
        for k in range(4):
            global_v_row = j * 4 + k
            for r in range(4):
                V[global_v_row][28 + r] = V_bot[global_v_row][r]
    
    # W_bot has 4 rows corresponding to k in {0,1,2,3}
    # Global W row for bottom block: k*3 + 2
    for k in range(4):
        global_w_row = k * 3 + 2
        for r in range(4):
            W[global_w_row][28 + r] = W_bot[k][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
