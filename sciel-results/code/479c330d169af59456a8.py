"""
Matrix multiplication 4x4 x 4x9 tensor decomposition.

Strategy:
  - Base: schoolbook rank 144 (n*m*p = 4*4*9).
  - Try to use known Strassen-like decompositions to reduce rank.
  - The 4x4x4 tensor has a known rank-18 decomposition (from combining
    Strassen 2x2x2 and other blocks). Using Kronecker structure,
    M_{4,4,9} can be viewed as M_{4,4,1} ⊗ M_{1,1,9} in a loose sense,
    but more directly: split the 4x9 into two 4x4 and one 4x1 block.
    
  - A safer approach: use the known rank-18 for 4x4x4 and adapt.
  - Actually, for M_{4,4,9}, we can think of it as 9 copies of M_{4,4,1},
    but that's not right either.

  Let's use the direct approach: the tensor T_{i,j,k} for i in 1..4 (rows of A),
  j in 1..4 (cols of A, rows of B), k in 1..9 (cols of B) has T_{i,j,k}=1.
  
  Schoolbook: rank 144.
  
  To get rank < 144, we need a non-trivial decomposition.
  
  One known fact: R(M_{2,2,2}) = 7 (Strassen).
  For M_{4,4,4}, we can partition into 2x2 blocks and apply Strassen, giving
  rank 7^2 = 49? No, that's not right. The Kronecker product of two 2x2x2
  tensors gives a 4x4x4 tensor with rank 7*7 = 49. But the actual rank of
  M_{4,4,4} is known to be at most 49, and in fact it's believed to be around
  49 or slightly less.
  
  However, for M_{4,4,9}, we can write it as:
  A (4x4) * B (4x9).
  
  Split B into B1 (4x4), B2 (4x4), B3 (4x1).
  Then C = A*B1 + A*B2 + A*B3.
  
  Using Strassen for A*B1 and A*B2 (each rank 7 for the 2x2x2 sub-blocks,
  but for 4x4x4 it's more complex).
  
  Actually, let's use a simpler approach: use the rank-18 decomposition for
  4x4x4 if we can find it, and then handle the 9 columns.
  
  For the 4x4x4 case, a known rank-18 decomposition exists (over certain fields).
  But implementing this correctly is complex.
  
  Alternative: use the fact that M_{4,4,9} has a tensor rank at most
  R(M_{4,4,4}) + R(M_{4,4,4}) + R(M_{4,4,1}).
  R(M_{4,4,1}) = 16 (schoolbook, since it's just A * column vector).
  R(M_{4,4,4}) <= 49 (from Kronecker of two Strassen).
  Total <= 49 + 49 + 16 = 114.
  
  But this is an upper bound via sum, not necessarily achievable in a single
  tensor decomposition of the combined problem. Actually, since the outputs
  are for different columns, the decompositions are independent and the total
  rank is the sum. So rank <= 114 is valid.
  
  Let's implement this: use Strassen-like rank-49 decomposition for 4x4x4,
  and rank-16 for 4x4x1.
  
  For 4x4x4, partition into 2x2 blocks:
  A = [[A11, A12], [A21, A22]]
  B = [[B11, B12], [B21, B22]]
  C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
             [A21*B11 + A22*B21, A21*B12 + A22*B22]]
  
  Using Strassen for each 2x2x2 sub-problem, but the 4x4x4 requires
  combining 2x2x2 decompositions. The rank of the Kronecker product of two
  rank-7 tensors is 7*7 = 49.
  
  So we can construct a rank-49 decomposition for 4x4x4 using the Kronecker
  product of two Strassen decompositions.
  
  Similarly, for 4x4x1, we can view it as 4x2x1 and 4x2x1, or just use
  schoolbook rank 16.
  
  Total rank = 49 + 49 + 16 = 114.
  
  Let's implement this.
"""

import time

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds, with safety margin
    
    n, m, p = 4, 4, 9
    
    # We'll build the decomposition by combining:
    # - Two 4x4x4 Strassen-Kronecker decompositions (rank 49 each)
    # - One 4x4x1 schoolbook decomposition (rank 16)
    
    # Helper: create Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        # Indices: i,j in 0..1 for A, j,k in 0..1 for B, k,i in 0..1 for C
        # C = A * B, so C[i,k] = sum_j A[i,j]*B[j,k]
        
        # Strassen's 7 products:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B11)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M4
        # C10 = M2 + M7
        # C11 = M1 - M2 + M3 - M6
        
        rank = 7
        U = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
        V = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
        W = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
        
        # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        
        # M1: (A00+A11)*(B00+B11)
        # A00: i=0,j=0; A11: i=1,j=1
        # B00: j=0,k=0; B11: j=1,k=1
        # C00: k=0,i=0; C11: k=1,i=1
        U[0][0] += 1  # A00
        U[3][0] += 1  # A11
        V[0][0] += 1  # B00
        V[3][0] += 1  # B11
        W[0][0] += 1  # C00
        W[3][0] -= 1  # C11
        
        # M2: (A10+A11)*B00
        # A10: i=1,j=0; A11: i=1,j=1
        # B00: j=0,k=0
        # C10: k=0,i=1
        U[2][1] += 1  # A10
        U[3][1] += 1  # A11
        V[0][1] += 1  # B00
        W[2][1] += 1  # C10
        
        # M3: A00*(B10-B11)
        # A00: i=0,j=0
        # B10: j=1,k=0; B11: j=1,k=1
        # C01: k=1,i=0
        U[0][2] += 1  # A00
        V[2][2] += 1  # B10
        V[3][2] -= 1  # B11
        W[1][2] += 1  # C01
        
        # M4: A11*(B01-B11)
        # A11: i=1,j=1
        # B01: j=0,k=1; B11: j=1,k=1
        # C01: k=1,i=0; C11: k=1,i=1
        U[3][3] += 1  # A11
        V[1][3] += 1  # B01
        V[3][3] -= 1  # B11
        W[1][3] += 1  # C01
        W[3][3] -= 1  # C11
        
        # M5: (A00+A01)*B11
        # A00: i=0,j=0; A01: i=0,j=1
        # B11: j=1,k=1
        # C00: k=0,i=0; C11: k=1,i=1
        U[0][4] += 1  # A00
        U[1][4] += 1  # A01
        V[3][4] += 1  # B11
        W[0][4] += 1  # C00
        W[3][4] -= 1  # C11
        
        # M6: (A10-A00)*(B00+B10)
        # A10: i=1,j=0; A00: i=0,j=0
        # B00: j=0,k=0; B10: j=1,k=0
        # C10: k=0,i=1; C11: k=1,i=1
        U[2][5] += 1  # A10
        U[0][5] -= 1  # A00
        V[0][5] += 1  # B00
        V[2][5] += 1  # B10
        W[2][5] -= 1  # C10
        W[3][5] -= 1  # C11
        
        # M7: (A01-A11)*(B01+B11)
        # A01: i=0,j=1; A11: i=1,j=1
        # B01: j=0,k=1; B11: j=1,k=1
        # C10: k=0,i=1; C00: k=0,i=0; C11: k=1,i=1
        U[1][6] += 1  # A01
        U[3][6] -= 1  # A11
        V[1][6] += 1  # B01
        V[3][6] += 1  # B11
        W[0][6] -= 1  # C00
        W[2][6] += 1  # C10
        
        return U, V, W
    
    # Helper: create schoolbook 4x4x1 decomposition (rank 16)
    def schoolbook_4x4x1():
        n, m, p = 4, 4, 1
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    # Helper: create Kronecker product of two tensor decompositions
    def kron_decomp(U1, V1, W1, U2, V2, W2):
        """
        U1: (n1*m1, r1), V1: (m1*p1, r1), W1: (p1*n1, r1)
        U2: (n2*m2, r2), V2: (m2*p2, r2), W2: (p2*n2, r2)
        Returns U, V, W for (n1*n2, m1*m2, p1*p2) with rank r1*r2
        """
        n1, m1, p1 = U1[0].__len__() and None or None  # Need to infer dimensions
        # Actually, let's infer from the shapes
        r1 = len(U1[0])
        r2 = len(U2[0])
        rank = r1 * r2
        
        # U1 has shape (n1*m1, r1), so n1*m1 = len(U1)
        nm1 = len(U1)
        mp1 = len(V1)
        pn1 = len(W1)
        
        nm2 = len(U2)
        mp2 = len(V2)
        pn2 = len(W2)
        
        U = [[0]*rank for _ in range(nm1 * nm2)]
        V = [[0]*rank for _ in range(mp1 * mp2)]
        W = [[0]*rank for _ in range(pn1 * pn2)]
        
        for c1 in range(r1):
            for c2 in range(r2):
                c = c1 * r2 + c2
                for i in range(nm1):
                    val_u1 = U1[i][c1]
                    if val_u1 != 0:
                        for i2 in range(nm2):
                            val_u2 = U2[i2][c2]
                            if val_u2 != 0:
                                U[i * nm2 + i2][c] += val_u1 * val_u2
                for i in range(mp1):
                    val_v1 = V1[i][c1]
                    if val_v1 != 0:
                        for i2 in range(mp2):
                            val_v2 = V2[i2][c2]
                            if val_v2 != 0:
                                V[i * mp2 + i2][c] += val_v1 * val_v2
                for i in range(pn1):
                    val_w1 = W1[i][c1]
                    if val_w1 != 0:
                        for i2 in range(pn2):
                            val_w2 = W2[i2][c2]
                            if val_w2 != 0:
                                W[i * pn2 + i2][c] += val_w1 * val_w2
        
        return U, V, W
    
    # Step 1: Get Strassen 2x2x2 decomposition
    U_s, V_s, W_s = strassen_2x2x2()
    
    # Step 2: Create 4x4x4 decomposition via Kronecker product
    U_444, V_444, W_444 = kron_decomp(U_s, V_s, W_s, U_s, V_s, W_s)
    
    # Step 3: Create 4x4x1 schoolbook decomposition
    U_441, V_441, W_441 = schoolbook_4x4x1()
    
    # Step 4: Combine the three decompositions
    # Total rank = 49 + 49 + 16 = 114
    rank_total = 49 + 49 + 16
    
    U_total = [[0]*rank_total for _ in range(16)]  # 4*4
    V_total = [[0]*rank_total for _ in range(36)]  # 4*9
    W_total = [[0]*rank_total for _ in range(36)]  # 9*4
    
    # Offset for first 4x4x4 (columns 0-3 of B)
    offset1 = 0
    for i in range(16):
        for c in range(49):
            U_total[i][offset1 + c] = U_444[i][c]
            V_total[i * 9 + (c // 49) * 4 + (c % 4) if False else i % 4 * 4 + (i // 4) * 0][offset1 + c] = 0  # This is getting complicated
            # Let me redo this more carefully
    
    # Actually, let me rebuild this more carefully.
    # For the first 4x4x4 block (B columns 0-3):
    # A is 4x4, B is 4x4 (first 4 columns of the 4x9 matrix)
    # U: 4x4 = 16 rows, V: 4x4 = 16 rows, W: 4x4 = 16 rows
    
    # For the second 4x4x4 block (B columns 4-7):
    # Same structure, but V and W indices shifted
    
    # For the 4x4x1 block (B column 8):
    # U: 4x4 = 16 rows, V: 4x1 = 4 rows, W: 1x4 = 4 rows
    
    # Let me rebuild from scratch with proper indexing.
    
    U_total = [[0]*rank_total for _ in range(16)]
    V_total = [[0]*rank_total for _ in range(36)]
    W_total = [[0]*rank_total for _ in range(36)]
    
    # Block 1: 4x4x4 for B columns 0-3
    for i in range(16):
        for c in range(49):
            U_total[i][c] = U_444[i][c]
    
    # V for block 1: B rows 0-3, cols 0-3
    for i in range(16):
        j = i // 4  # B row (0-3)
        k = i % 4   # B col (0-3)
        V_total[j * 9 + k][c] = V_444[i][c]
    
    # W for block 1: C rows 0-3, cols 0-3
    for i in range(16):
        k = i // 4  # C col (0-3)
        i_c = i % 4 # C row (0-3)
        W_total[k * 4 + i_c][c] = W_444[i][c]
    
    # Block 2: 4x4x4 for B columns 4-7
    offset2 = 49
    for i in range(16):
        for c in range(49):
            U_total[i][offset2 + c] = U_444[i][c]
    
    for i in range(16):
        j = i // 4  # B row (0-3)
        k = i % 4   # B col (0-3) within this block
        V_total[j * 9 + (k + 4)][offset2 + c] = V_444[i][c]
    
    for i in range(16):
        k = i // 4  # C col (0-3) within this block
        i_c = i % 4 # C row (0-3)
        W_total[(k + 4) * 4 + i_c][offset2 + c] = W_444[i][c]
    
    # Block 3: 4x4x1 for B column 8
    offset3 = 98
    for i in range(16):
        for c in range(16):
            U_total[i][offset3 + c] = U_441[i][c]
    
    for i in range(4):
        j = i  # B row (0-3)
        V_total[j * 9 + 8][offset3 + c] = V_441[i][c]
    
    for i in range(4):
        k = 8  # C col (8)
        i_c = i # C row (0-3)
        W_total[k * 4 + i_c][offset3 + c] = W_441[i][c]
    
    # Check time
    if time.time() - start_time > deadline - 5:
        # Fall back to schoolbook if running out of time
        pass  # The above should be fast enough
    
    return {"U": U_total, "V": V_total, "W": W_total}
