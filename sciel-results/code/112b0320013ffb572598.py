"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find a low-rank tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Uses recursive Strassen decomposition.
    1. Decompose 4x4x4 using tensor product of Strassen 2x2x2 (rank 7^2 = 49).
    2. Extend to 4x4x7 by adding schoolbook terms for the extra 3 columns (rank 4*4*3 = 48? No, 
       actually we can just use the 4x4x4 decomposition for the first 4 columns and 
       a separate decomposition for the last 3 columns? 
       Wait, the tensor is T[i,j,k] for i in 0..3, j in 0..3, k in 0..6.
       We can split k into k1 (0..3) and k2 (4..6).
       T[i,j,k1] is handled by the 4x4x4 Strassen decomp (rank 49).
       T[i,j,k2] is handled by a 4x4x3 decomp.
       For 4x4x3, we can use Strassen on 2x2 blocks? 
       4x4x3: A is 4x4, B is 4x3, C is 4x3.
       We can pad B to 4x4 and C to 4x4, use rank 49, but that's inefficient.
       Better: Use Strassen 2x2x2 for the 2x2 blocks of A and B.
       A = [[A11, A12], [A21, A22]] (2x2 blocks)
       B = [[B11, B12, B13], [B21, B22, B23]] (2x2, 2x1, 2x1? No, B is 4x3.
            B11 is 2x2, B12 is 2x1, B13 is 2x1? No, B is 4 rows, 3 cols.
            B = [[B11(2x2), B12(2x1)], [B21(2x2), B22(2x1)]]? No, 3 cols means 2+1.
            So B11 is 2x2, B12 is 2x1.
            C = [[C11(2x2), C12(2x1)], [C21(2x2), C22(2x1)]].
            C11 = A11*B11 + A21*B21. This is 2x2 * 2x2 + 2x2 * 2x2.
            Using Strassen for each product? No, Strassen is for 2x2 * 2x2.
            We can compute C11 using Strassen if we treat the sum as two separate multiplications?
            No, Strassen computes (A11*B11 + A21*B21) is not directly Strassen.
            Strassen computes C = A*B where A,B are 2x2.
            Here we have C_block = A_top * B_top + A_bottom * B_bottom.
            This is exactly the 2x2 * 2x2 multiplication if we view 
            A' = [[A_top], [A_bottom]] (4x2) and B' = [[B_top, B_bottom]] (2x4)? No.
            
            Let's just use the tensor product approach for the whole thing.
            We can construct the 4x4x7 tensor as the Kronecker product of:
            - A 2x2x2 Strassen decomposition (rank 7) for the "block" structure of A and C (2x2 blocks)
            - And a "vector" structure for the inner dimension.
            
            Actually, the simplest robust method:
            1. Get Strassen 2x2x2 (U2, V2, W2) with rank 7.
            2. Compute 4x4x4 decomposition via tensor product:
               U444[i*m+j, c1*7+c2] = U2[i2*m2+j2, c1] * U2[i1*m1+j1, c2]
               This gives rank 49 for 4x4x4.
            3. For the 4x4x7 case, we need to handle k=4,5,6.
               We can just add 12 schoolbook terms for the last 3 columns?
               Rank = 49 + 4*4*3 = 49 + 48 = 97.
               Schoolbook for 4x4x7 is 112. So 97 is an improvement.
               
               Can we do better for the 4x4x3 part?
               4x4x3 can be decomposed using Strassen 2x2x2 for the 2x2 blocks?
               Let's try to construct 4x4x3 using Strassen 2x2x2 on the 2x2 blocks of A and B.
               A is 4x4 -> 2x2 blocks.
               B is 4x3 -> 2x2 block (cols 0-1) and 2x1 block (col 2).
               C is 4x3 -> 2x2 block (cols 0-1) and 2x1 block (col 2).
               
               C11 (2x2) = A11*B11 + A21*B21.
               This is a sum of two 2x2 products.
               We can use Strassen to compute A11*B11 (rank 7) and A21*B21 (rank 7).
               So C11 takes rank 14.
               Similarly C21 (2x2) = A12*B11 + A22*B21. Rank 14.
               
               C12 (2x1) = A11*B12 + A21*B22.
               B12 is 2x1, B22 is 2x1.
               This is a "vector" multiplication.
               A11 (2x2) * B12 (2x1) -> 2x1.
               This can be done with rank 2? (Schoolbook: 2*2*1 = 4 terms).
               Can we do better?
               For 2x2 * 2x1, the tensor is rank 4? 
               Let's just use schoolbook for the 2x1 parts.
               Rank for C12 = 4 (A11*B12) + 4 (A21*B22) = 8?
               Or we can combine?
               
               Total rank for 4x4x3:
               C11: 14
               C21: 14
               C12: 8
               C22: 8
               Total: 44.
               
               So total rank for 4x4x7 = 49 (for 4x4x4) + 44 (for 4x4x3) = 93.
               
               Is there a better way?
               We can use the 4x4x4 Strassen decomposition and just extend it.
               But 93 is already better than 112.
               
               Let's implement the tensor product for 4x4x4 and schoolbook for the rest?
               49 + 48 = 97.
               The Strassen-based 4x4x3 (rank 44) is better.
               
               Let's implement the 4x4x4 Strassen tensor product and the 4x4x3 Strassen-based decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # 1. Define Strassen 2x2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i] for n=m=p=2
    # U[i*2+j], V[j*2+k], W[k*2+i]
    
    def get_strassen_2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        # A00=0, A11=3. B00=0, B11=3.
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        W[0][0]=1; W[3][0]=1
        
        # M2 = (A10+A11)B00 -> C10
        # A10=2, A11=3. B00=0.
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[2][1]=1
        
        # M3 = A00(B10-B11) -> C01
        # A00=0. B10=2, B11=3.
        U[0][2]=1
        V[2][2]=1; V[3][2]=-1
        W[1][2]=1
        
        # M4 = A11(B01-B00) -> C00, C10
        # A11=3. B01=1, B00=0.
        U[3][3]=1
        V[1][3]=1; V[0][3]=-1
        W[0][3]=1; W[2][3]=1
        
        # M5 = (A00+A01)B11 -> C00, C01
        # A00=0, A01=1. B11=3.
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        W[0][4]=-1; W[1][4]=1
        
        # M6 = (A10-A00)(B00+B01) -> C11
        # A10=2, A00=0. B00=0, B01=1.
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1
        
        # M7 = (A01-A11)(B10+B11) -> C00
        # A01=1, A11=3. B10=2, B11=3.
        U[1][6]=1; U[3][6]=-1
        V[2][6]=1; V[3][6]=1
        W[0][6]=1
        
        return U, V, W

    U2, V2, W2 = get_strassen_2x2()
    
    # 2. Construct 4x4x4 decomposition via tensor product
    # n=m=p=4. Split into 2x2 blocks.
    # U4[i*4+j, c1*7+c2] = U2[i2*2+j2, c1] * U2[i1*2+j1, c2]
    # where i = 2*i2 + i1, j = 2*j2 + j1.
    # V4[j*4+k, c1*7+c2] = V2[j2*2+k2, c1] * V2[j1*2+k1, c2]
    # W4[k*4+i, c1*7+c2] = W2[k2*2+i2, c1] * W2[k1*2+i1, c2]
    
    rank_444 = 49
    U444 = [[0]*rank_444 for _ in range(16)]
    V444 = [[0]*rank_444 for _ in range(16)]
    W444 = [[0]*rank_444 for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1*7 + c2
            for i2 in range(2):
                for j2 in range(2):
                    u2_val = U2[i2*2+j2][c1]
                    if u2_val == 0: continue
                    for i1 in range(2):
                        for j1 in range(2):
                            u1_val = U2[i1*2+j1][c2]
                            if u1_val == 0: continue
                            u_val = u2_val * u1_val
                            i = i2*2 + i1
                            j = j2*2 + j1
                            U444[i*4+j][c] = u_val
                            
                            # V part
                            # V2[j2*2+k2][c1] * V2[j1*2+k1][c2]
                            for k2 in range(2):
                                v2_val = V2[j2*2+k2][c1]
                                if v2_val == 0: continue
                                for k1 in range(2):
                                    v1_val = V2[j1*2+k1][c2]
                                    if v1_val == 0: continue
                                    v_val = v2_val * v1_val
                                    k = k2*2 + k1
                                    V444[j*4+k][c] = v_val
                                    
                                    # W part
                                    # W2[k2*2+i2][c1] * W2[k1*2+i1][c2]
                                    w2_val = W2[k2*2+i2][c1]
                                    if w2_val == 0: continue
                                    w1_val = W2[k1*2+i1][c2]
                                    if w1_val == 0: continue
                                    w_val = w2_val * w1_val
                                    W444[k*4+i][c] = w_val
                                    
    # 3. Construct 4x4x3 decomposition
    # n=4, m=4, p=3.
    # We can use Strassen 2x2x2 for the 2x2 blocks of A and B (first 2 cols of B).
    # B is 4x3. B = [B_left(4x2), B_right(4x1)].
    # C is 4x3. C = [C_left(4x2), C_right(4x1)].
    # C_left = A * B_left. This is 4x4 * 4x2.
    # We can decompose 4x4 * 4x2 using Strassen 2x2x2 on blocks?
    # A = [[A11, A12], [A21, A22]] (2x2 blocks)
    # B_left = [[B11, B12], [B21, B22]] (2x2 blocks)
    # C_left = [[C11, C12], [C21, C22]] (2x2 blocks)
    # C11 = A11*B11 + A21*B21. (2x2 * 2x2 + 2x2 * 2x2)
    # We can use Strassen for A11*B11 (rank 7) and A21*B21 (rank 7).
    # So C11 takes 14 rank.
    # Similarly C12, C21, C22 take 14 rank each.
    # Total for C_left: 4 * 14 = 56.
    
    # C_right = A * B_right. (4x4 * 4x1)
    # B_right = [B13 (2x1), B23 (2x1)].
    # C_right = [C13 (2x1), C23 (2x1)].
    # C13 = A11*B13 + A21*B23. (2x2 * 2x1 + 2x2 * 2x1)
    # Schoolbook for 2x2 * 2x1 is rank 4.
    # So C13 takes 4 + 4 = 8 rank.
    # C23 takes 8 rank.
    # Total for C_right: 16.
    
    # Total rank for 4x4x3: 56 + 16 = 72.
    # Wait, 4x4x3 schoolbook is 48. 72 is worse!
    # My Strassen extension for 4x4x3 is inefficient because Strassen is for square matrices.
    # For 4x4 * 4x3, schoolbook is 48.
    # So we should just use schoolbook for the 4x4x3 part.
    
    rank_443 = 48
    U443 = [[0]*rank_443 for _ in range(16)]
    V443 = [[0]*rank_443 for _ in range(12)] # 4*3
    W443 = [[0]*rank_443 for _ in range(12)] # 3*4
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U443[i*4+j][col] = 1
                V443[j*4+k][col] = 1
                W443[k*4+i][col] = 1
                col += 1
                
    # 4. Combine 4x4x4 and 4x4x3
    # Total rank = 49 + 48 = 97.
    total_rank = rank_444 + rank_443
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)] # 4*7
    W = [[0]*total_rank for _ in range(28)] # 7*4
    
    # Copy 4x4x4 part (k=0..3)
    for r in range(rank_444):
        for i in range(16):
            U[i][r] = U444[i][r]
        for j in range(16):
            V[j][r] = V444[j][r]
        for k in range(16):
            W[k][r] = W444[k][r]
            
    # Copy 4x4x3 part (k=4..6)
    # V index for 4x4x3: j*4+k where k is 0..2.
    # In full V, k is 0..6. So k_full = k + 4.
    # W index for 4x4x3: k*4+i where k is 0..2.
    # In full W, k is 0..6. So k_full = k + 4.
    
    for r in range(rank_443):
        for i in range(16):
            U[i][rank_444 + r] = U443[i][r]
        for j in range(16):
            for k in range(3):
                V[j*4 + k + 4][rank_444 + r] = V443[j*4 + k][r]
        for k in range(3):
            for i in range(4):
                W[(k+4)*4 + i][rank_444 + r] = W443[k*4 + i][r]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
