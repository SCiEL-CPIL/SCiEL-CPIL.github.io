import time

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[:,c] V[:,c]^T W[:,c]^T == tensor."""
    rank = len(U[0])
    # Tensor T[i, j, k] = 1
    # Reconstructed: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check a few random entries or all (small sizes)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strategy: 
    # The naive rank is n*m*p = 3*4*4 = 48.
    # We can try to find a lower rank.
    # Known bounds: 
    # For 2x2x2, Strassen is 7.
    # For 3x3x3, best known is 23.
    # For 3x4x4, we can decompose into 2x4x4 and 1x4x4.
    # 2x4x4 can be decomposed into four 2x2x2 multiplications?
    # Let A be 2x4, B be 4x4.
    # A = [A11 A12; A21 A22] where blocks are 2x2.
    # B = [B11 B12; B21 B22] where blocks are 2x2.
    # C = A B.
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    # This is 8 separate 2x2x2 multiplications? No, it's sums.
    # Each term Aij Bjk is a 2x2x2 multiplication.
    # There are 4 such multiplications for each block of C?
    # Actually, C is 2x4.
    # C[:, 0:2] = A1 B11 + A2 B21 (where A1, A2 are 2x2, B11, B21 are 2x2).
    # This requires computing A1 B11, A2 B21. (2 multiplications)
    # C[:, 2:4] = A1 B12 + A2 B22. (2 multiplications)
    # Total 4 multiplications of size 2x2x2 for the 2x4x4 block.
    # Rank for 2x4x4 <= 4 * 7 = 28.
    
    # Now the 1x4x4 block (row 2 of A).
    # A2 (1x4) * B (4x4).
    # A2 = [A21 A22] (1x2 each).
    # C2 = A21 B11 + A22 B21 + A21 B12 ... no.
    # C2 (1x4) = A2 B.
    # C2[0:2] = A21 B11 + A22 B21. (Two 1x2x2 multiplications)
    # C2[2:4] = A21 B12 + A22 B22. (Two 1x2x2 multiplications)
    # Rank of 1x2x2 is 4 (since it's just dot products, essentially 4 scalars).
    # Actually, 1x2x2 tensor has rank 4?
    # T[i,j,k] for i in 1, j in 2, k in 2.
    # T[0,j,k] = 1.
    # This is rank 1? No.
    # T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1.
    # It's the outer product of [1], [1,1], [1,1]. Rank 1.
    # Wait, the tensor for matrix multiplication M_{n x p} is T[i,j,k] where (A_i . B_j) -> C_{i,k}.
    # For 1x2x2:
    # A has 1 row, 2 cols. B has 2 rows, 2 cols. C has 1 row, 2 cols.
    # C_0,0 = A_0,0 B_0,0 + A_0,1 B_1,0
    # C_0,1 = A_0,0 B_0,1 + A_0,1 B_1,1
    # Tensor T[i, j, k] where i is output row (0), j is input col of A (0,1), k is output col (0,1).
    # Wait, standard indexing: U[i*m+j], V[j*p+k], W[k*n+i].
    # n=1, m=2, p=2.
    # U[0*2+j] = U[j]. V[j*2+k]. W[k*1+0] = W[k].
    # Product sum_c U[j,c] V[j,k,c] ... wait.
    # The tensor is T_{i,j,k} = 1 if we multiply A_{i,j} and B_{j,k}? No.
    # The multiplication tensor T satisfies C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}. It is 1 for all valid i,j,k.
    # So for 1x2x2, T is 1x2x2 all ones.
    # Rank of all-ones tensor of size I x J x K is max(I,J,K)? No.
    # Rank of all-ones tensor is 1 if we allow real numbers?
    # T = u v^T w^T where u=[1], v=[1,1], w=[1,1].
    # Yes, rank 1.
    # So the 1x4x4 block:
    # C2[0:2] = A21 B11 + A22 B21.
    # A21 is 1x2, B11 is 2x2. This is 1x2x2 multiplication. Rank 1.
    # A22 is 1x2, B21 is 2x2. This is 1x2x2 multiplication. Rank 1.
    # So C2[0:2] requires rank 2?
    # Let's check.
    # C2[0] = A21[0] B11[0,0] + A21[1] B11[1,0] + A22[0] B21[0,0] + A22[1] B21[1,0]
    # This is a sum of 4 terms.
    # Can we do it in rank 2?
    # The tensor for this part is not a single standard multiplication tensor, it's a sum.
    # However, we can just use the schoolbook rank for the 1x4x4 part which is 1*4*4=16.
    # Or we can be smarter.
    # The 1x4x4 multiplication is equivalent to 4 independent 1x2x2 multiplications?
    # A (1x4) * B (4x4).
    # Split A into two 1x2 vectors a1, a2.
    # Split B into four 2x2 blocks B11, B12, B21, B22.
    # C = a1 [B11 B12] + a2 [B21 B22].
    # C[0:2] = a1 B11 + a2 B21.
    # C[2:4] = a1 B12 + a2 B22.
    # a1 B11 is 1x2x2 mult (rank 1).
    # a2 B21 is 1x2x2 mult (rank 1).
    # So C[0:2] is sum of two rank-1 tensors?
    # No, they are separate multiplications.
    # Total rank for 1x4x4 using this decomposition:
    # 4 multiplications of type 1x2x2. Each rank 1. Total rank 4.
    
    # So total rank estimate:
    # Top 2x4x4: 4 * 7 = 28.
    # Bottom 1x4x4: 4 * 1 = 4.
    # Total rank = 32.
    
    # Let's verify if rank 32 is possible.
    # We need to construct U, V, W.
    
    # We will build the decomposition by concatenating the factor matrices for the top and bottom parts.
    
    # Part 1: Top 2x4x4
    # A_top (2x4), B (4x4), C_top (2x4).
    # Indices:
    # i in 0..1 (row of A_top)
    # j in 0..3 (col of A_top / row of B)
    # k in 0..3 (col of C_top)
    
    # We split A_top into A1 (rows 0-1, cols 0-1), A2 (rows 0-1, cols 2-3).
    # Wait, A_top is 2x4.
    # A1 = A_top[0:2, 0:2]
    # A2 = A_top[0:2, 2:4]
    # B1 = B[0:2, :] -> B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B2 = B[2:4, :] -> B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # C_top[0:2, 0:2] = A1 B11 + A2 B21
    # C_top[0:2, 2:4] = A1 B12 + A2 B22
    # C_top is 2x4.
    # Let's check dimensions.
    # A1 (2x2) * B11 (2x2) -> 2x2.
    # A2 (2x2) * B21 (2x2) -> 2x2.
    # Sum is 2x2. Correct.
    
    # We need 4 Strassen decompositions.
    # S1 for A1 B11
    # S2 for A2 B21
    # S3 for A1 B12
    # S4 for A2 B22
    
    # Each Strassen decomp has rank 7.
    # Total rank 28.
    
    # Part 2: Bottom 1x4x4
    # A_bot (1x4), B (4x4), C_bot (1x4).
    # A_bot[0:2] = a1, A_bot[2:4] = a2.
    # C_bot[0:2] = a1 B11 + a2 B21.
    # C_bot[2:4] = a1 B12 + a2 B22.
    
    # R1 for a1 B11 (1x2x2, rank 1)
    # R2 for a2 B21 (1x2x2, rank 1)
    # R3 for a1 B12 (1x2x2, rank 1)
    # R4 for a2 B22 (1x2x2, rank 1)
    
    # Total rank 4.
    
    # Grand total rank 32.
    
    # Let's construct the matrices.
    
    # Helper to create Strassen factors for 2x2x2
    # Returns U (4x7), V (4x7), W (4x7)
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    def strassen_222():
        # i in 0..1, j in 0..1, k in 0..1
        # U index: i*2+j
        # V index: j*2+k
        # W index: k*2+i
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1 # W[k*2+i]
        # W00=1, W01=1, W10=1, W11=1.
        # W index: k*2+i.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        # All 1.
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1 # C10, C11
        # W[k*2+i]. i=1.
        # k=0 -> 1. k=1 -> 3.
        # C10 is i=1, k=0 -> W[1]. C11 is i=1, k=1 -> W[3].
        # So W[1]=1, W[3]=1.
        
        # M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[0][2] = 1; W[2][2] = 1 # C01, C00?
        # C = A B.
        # C01 = A00 B01 + A01 B11.
        # C00 = A00 B00 + A01 B10.
        # M3 contributes to C01 and C11?
        # Standard Strassen:
        # C00 = M1 - M4 + M5
        # C01 = M3 + M4
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M6
        # Let's re-verify W indices for M3.
        # M3 = A00 (B10 - B11).
        # A00 is U[0].
        # B10 is V[2] (j=1, k=0). B11 is V[3] (j=1, k=1).
        # So V[2]=1, V[3]=-1.
        # Result is 2x2 matrix where all entries are -B11 + B10? No.
        # The product is scalar * matrix.
        # The tensor factor W must distribute this scalar to the correct C entries.
        # M3 is added to C01 and C11.
        # C01 index: i=0, k=1 -> W[1*2+0] = W[2].
        # C11 index: i=1, k=1 -> W[1*2+1] = W[3].
        # So W[2]=1, W[3]=1.
        
        # M4 = (A00+A01)B11
        U[0][3] = 1; U[1][3] = 1
        V[3][3] = 1
        # M4 subtracted from C00, added to C01, C11.
        # C00: i=0, k=0 -> W[0]. W[0] = -1.
        # C01: i=0, k=1 -> W[2]. W[2] = 1.
        # C11: i=1, k=1 -> W[3]. W[3] = 1.
        
        # M5 = (A01-A00)(B00+B10)
        U[1][4] = 1; U[0][4] = -1
        V[0][4] = 1; V[2][4] = 1
        # M5 added to C00, C10.
        # C00: W[0] = 1.
        # C10: i=1, k=0 -> W[1]. W[1] = 1.
        
        # M6 = (A10-A00)(B01+B11)
        U[2][5] = 1; U[0][5] = -1
        V[1][5] = 1; V[3][5] = 1
        # M6 added to C10, C11.
        # C10: W[1] = 1.
        # C11: W[3] = 1.
        
        # M7 = (A11-A10)(B01+B10)
        U[3][6] = 1; U[2][6] = -1
        V[1][6] = 1; V[2][6] = 1
        # M7 added to C00, C11.
        # C00: W[0] = 1.
        # C11: W[3] = 1.
        
        # Sum up W contributions
        # W[0]: M1(1) - M4(1) + M5(1) + M7(1) = 2?
        # Wait, Strassen formulas:
        # C00 = M1 - M4 + M5
        # C01 = M3 + M4
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M6
        # Where are the M7s?
        # Standard Strassen:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B21-B22)
        # P4 = A22(B12-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        # C11 = P1 - P4 + P5 - P7
        # C12 = P3 + P4
        # C21 = P2 - P6
        # C22 = P1 - P2 + P3 + P6
        # My M mapping was slightly off.
        
        # Let's use a known correct 2x2x2 rank 7 decomposition.
        # I will hardcode the 7 columns.
        
        U_s = [
            [1, 0, 1, 1, 1, -1, 0], # A00
            [0, 0, 0, 1, -1, 1, -1],# A01
            [0, 1, 0, 0, 0, -1, -1],# A10
            [1, 1, 0, 0, 0, 0, 1]   # A11
        ]
        V_s = [
            [1, 1, 0, 0, 0, 1, 0], # B00
            [0, 0, 1, -1, 0, 1, 1],# B01
            [1, 0, -1, 0, 1, 0, -1],# B10
            [1, 0, 0, 0, 1, 0, 0]  # B11
        ]
        W_s = [
            [1, 0, 0, -1, 0, 0, -1], # C00 (W[0])
            [0, 1, 1, 0, 1, 0, 0],   # C01 (W[1])
            [0, -1, 0, 0, -1, 1, 0], # C10 (W[2])
            [1, 0, 1, -1, 0, -1, 0]  # C11 (W[3])
        ]
        
        return U_s, V_s, W_s

    U_s, V_s, W_s = strassen_222()
    
    # Part 1: Top 2x4x4
    # Rank 28.
    # We need to map the 4x4x4 local indices to global 12x16x12 indices.
    # Global U size: 12 (3*4). Top 2 rows of A are i=0,1.
    # Global V size: 16 (4*4).
    # Global W size: 12 (4*3). Top 2 rows of C are i=0,1.
    
    # S1: A1 B11
    # A1 is A_top[0:2, 0:2]. Global U indices: i in 0,1, j in 0,1 -> i*4+j.
    # B11 is B[0:2, 0:2]. Global V indices: j in 0,1, k in 0,1 -> j*4+k.
    # C1 is C_top[0:2, 0:2]. Global W indices: k in 0,1, i in 0,1 -> k*3+i.
    
    U_top = [[0]*28 for _ in range(8)] # 2*4
    V_top = [[0]*28 for _ in range(16)] # 4*4
    W_top = [[0]*28 for _ in range(8)] # 2*4 -> but W is p*n = 4*3=12.
    # Wait, W is (p*n, rank). p=4, n=3. Size 12.
    # Top part uses i=0,1. So W rows k*3+i for k in 0..3, i in 0,1.
    
    # Let's just build full size matrices and fill in.
    U = [[0]*32 for _ in range(12)]
    V = [[0]*32 for _ in range(16)]
    W = [[0]*32 for _ in range(12)]
    
    # Strassen 1: A1 B11
    # Local i' in 0,1 (maps to global i in 0,1)
    # Local j' in 0,1 (maps to global j in 0,1)
    # Local k' in 0,1 (maps to global k in 0,1)
    for c in range(7):
        for i_l in range(2):
            for j_l in range(2):
                val_u = U_s[i_l*2+j_l][c]
                if val_u != 0:
                    i_g = i_l
                    j_g = j_l
                    U[i_g*4 + j_g][c] += val_u
        for j_l in range(2):
            for k_l in range(2):
                val_v = V_s[j_l*2+k_l][c]
                if val_v != 0:
                    j_g = j_l
                    k_g = k_l
                    V[j_g*4 + k_g][c] += val_v
        for k_l in range(2):
            for i_l in range(2):
                val_w = W_s[k_l*2+i_l][c]
                if val_w != 0:
                    k_g = k_l
                    i_g = i_l
                    W[k_g*3 + i_g][c] += val_w
    
    # Strassen 2: A2 B21
    # A2 is A_top[0:2, 2:4]. j_g in 2,3.
    # B21 is B[2:4, 0:2]. j_g in 2,3 (rows of B), k_g in 0,1 (cols of B).
    # C part is C_top[0:2, 0:2]. k_g in 0,1.
    for c in range(7, 14):
        cc = c - 7
        for i_l in range(2):
            for j_l in range(2):
                val_u = U_s[i_l*2+j_l][cc]
                if val_u != 0:
                    i_g = i_l
                    j_g = j_l + 2
                    U[i_g*4 + j_g][c] += val_u
        for j_l in range(2):
            for k_l in range(2):
                val_v = V_s[j_l*2+k_l][cc]
                if val_v != 0:
                    j_g = j_l + 2
                    k_g = k_l
                    V[j_g*4 + k_g][c] += val_v
        for k_l in range(2):
            for i_l in range(2):
                val_w = W_s[k_l*2+i_l][cc]
                if val_w != 0:
                    k_g = k_l
                    i_g = i_l
                    W[k_g*3 + i_g][c] += val_w
    
    # Strassen 3: A1 B12
    # A1 is A_top[0:2, 0:2]. j_g in 0,1.
    # B12 is B[0:2, 2:4]. j_g in 0,1, k_g in 2,3.
    # C part is C_top[0:2, 2:4]. k_g in 2,3.
    for c in range(14, 21):
        cc = c - 14
        for i_l in range(2):
            for j_l in range(2):
                val_u = U_s[i_l*2+j_l][cc]
                if val_u != 0:
                    i_g = i_l
                    j_g = j_l
                    U[i_g*4 + j_g][c] += val_u
        for j_l in range(2):
            for k_l in range(2):
                val_v = V_s[j_l*2+k_l][cc]
                if val_v != 0:
                    j_g = j_l
                    k_g = k_l + 2
                    V[j_g*4 + k_g][c] += val_v
        for k_l in range(2):
            for i_l in range(2):
                val_w = W_s[k_l*2+i_l][cc]
                if val_w != 0:
                    k_g = k_l + 2
                    i_g = i_l
                    W[k_g*3 + i_g][c] += val_w
    
    # Strassen 4: A2 B22
    # A2 is A_top[0:2, 2:4]. j_g in 2,3.
    # B22 is B[2:4, 2:4]. j_g in 2,3, k_g in 2,3.
    # C part is C_top[0:2, 2:4]. k_g in 2,3.
    for c in range(21, 28):
        cc = c - 21
        for i_l in range(2):
            for j_l in range(2):
                val_u = U_s[i_l*2+j_l][cc]
                if val_u != 0:
                    i_g = i_l
                    j_g = j_l + 2
                    U[i_g*4 + j_g][c] += val_u
        for j_l in range(2):
            for k_l in range(2):
                val_v = V_s[j_l*2+k_l][cc]
                if val_v != 0:
                    j_g = j_l + 2
                    k_g = k_l + 2
                    V[j_g*4 + k_g][c] += val_v
        for k_l in range(2):
            for i_l in range(2):
                val_w = W_s[k_l*2+i_l][cc]
                if val_w != 0:
                    k_g = k_l + 2
                    i_g = i_l
                    W[k_g*3 + i_g][c] += val_w
    
    # Part 2: Bottom 1x4x4
    # Rank 4.
    # R1: a1 B11. a1 is A_bot[0:2] (i=2, j in 0,1). B11 is B[0:2, 0:2].
    # C part is C_bot[0:2] (i=2, k in 0,1).
    # 1x2x2 mult.
    # U[2*4 + j, c] = 1 for j=0,1.
    # V[j*4 + k, c] = 1 for j=0,1, k=0,1.
    # W[k*3 + 2, c] = 1 for k=0,1.
    
    # R1: a1 B11
    c = 28
    for j in range(2):
        U[2*4 + j][c] = 1
    for j in range(2):
        for k in range(2):
            V[j*4 + k][c] = 1
    for k in range(2):
        W[k*3 + 2][c] = 1
        
    # R2: a2 B21
    # a2 is A_bot[2:4] (i=2, j in 2,3).
    # B21 is B[2:4, 0:2] (j in 2,3, k in 0,1).
    # C part is C_bot[0:2] (i=2, k in 0,1).
    c = 29
    for j in range(2, 4):
        U[2*4 + j][c] = 1
    for j in range(2, 4):
        for k in range(2):
            V[j*4 + k][c] = 1
    for k in range(2):
        W[k*3 + 2][c] = 1
        
    # R3: a1 B12
    # a1 is A_bot[0:2] (j in 0,1).
    # B12 is B[0:2, 2:4] (j in 0,1, k in 2,3).
    # C part is C_bot[2:4] (i=2, k in 2,3).
    c = 30
    for j in range(2):
        U[2*4 + j][c] = 1
    for j in range(2):
        for k in range(2, 4):
            V[j*4 + k][c] = 1
    for k in range(2, 4):
        W[k*3 + 2][c] = 1
        
    # R4: a2 B22
    # a2 is A_bot[2:4] (j in 2,3).
    # B22 is B[2:4, 2:4] (j in 2,3, k in 2,3).
    # C part is C_bot[2:4] (i=2, k in 2,3).
    c = 31
    for j in range(2, 4):
        U[2*4 + j][c] = 1
    for j in range(2, 4):
        for k in range(2, 4):
            V[j*4 + k][c] = 1
    for k in range(2, 4):
        W[k*3 + 2][c] = 1
        
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
