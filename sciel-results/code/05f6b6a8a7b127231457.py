import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    # Verify a decomposition
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Strassen decomposition for 2x2 * 2x2
    # Returns U (4x7), V (4x7), W (4x7) where U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    def strassen_2x2():
        """
        Returns U, V, W for 2x2 * 2x2 with rank 7.
        U is 4x7, V is 4x7, W is 4x7.
        Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]
        
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*2+i)
        
        # M1 = (A00+A11)(B00+B11)
        # A part: A00=1, A11=1 -> U[0,0]=1, U[3,0]=1
        # B part: B00=1, B11=1 -> V[0,0]=1, V[3,0]=1
        # C part: C00 += 1, C11 += 1 -> W[0,0]=1, W[3,0]=1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        # A part: A10=1, A11=1 -> U[2,1]=1, U[3,1]=1
        # B part: B00=1 -> V[0,1]=1
        # C part: C10 += 1 -> W[2,1]=1 (k=1, i=0 -> 1*2+0=2)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = A00(B01-B11)
        # A part: A00=1 -> U[0,2]=1
        # B part: B01=1, B11=-1 -> V[1,2]=1, V[3,2]=-1
        # C part: C01 += 1 -> W[1,2]=1 (k=0, i=1 -> 0*2+1=1)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4 = A11(B10-B00)
        # A part: A11=1 -> U[3,3]=1
        # B part: B10=1, B00=-1 -> V[2,3]=1, V[0,3]=-1
        # C part: C10 += 1, C00 += -1? No, C10 += 1, C11 += -1?
        # C22 = M1 - M2 + M3 + M6
        # C21 = M2 + M4
        # C11 = M1 + M4 - M5 + M7
        # Let's re-verify the C contributions for M4.
        # M4 contributes to C10 (C21 in 0-indexed C matrix where C[r,c] is row r col c)
        # Wait, C is 2x2. C[i,k].
        # C21 (i=1, k=0) is index 1*2+0 = 2.
        # C11 (i=0, k=1) is index 0*2+1 = 1.
        # C22 (i=1, k=1) is index 1*2+1 = 3.
        # C10 (i=0, k=0) is index 0.
        
        # Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M4 contributes +1 to C00 and +1 to C10.
        # C00 index 0. C10 index 2.
        W[0][3] = 1
        W[2][3] = 1
        
        # M5 = (A00+A01)B11
        # A part: A00=1, A01=1 -> U[0,4]=1, U[1,4]=1
        # B part: B11=1 -> V[3,4]=1
        # C part: -1 to C00, +1 to C01
        W[0][4] = -1
        W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        # A part: A10=1, A00=-1 -> U[2,5]=1, U[0,5]=-1
        # B part: B00=1, B01=1 -> V[0,5]=1, V[1,5]=1
        # C part: +1 to C11
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        # A part: A01=1, A11=-1 -> U[1,6]=1, U[3,6]=-1
        # B part: B10=1, B11=1 -> V[2,6]=1, V[3,6]=1
        # C part: +1 to C00
        W[0][6] = 1
        
        return U, V, W

    # Try to embed 4x5x6 into a larger structure using Strassen blocks.
    # This is complex. Let's try a simpler heuristic:
    # Can we reduce rank by finding linear dependencies in the schoolbook basis?
    # The tensor is T[i,j,k] = 1.
    # Rank is 120.
    
    # Let's try a specific known decomposition if possible.
    # For 4x4x4, rank is 49.
    # We can embed 4x4x4 into 4x5x6.
    # The 4x5x6 tensor contains a 4x4x4 sub-tensor (indices i<4, j<4, k<4).
    # If we have a rank 49 decomposition for 4x4x4, we can use it for the (i,j,k) in [0,4)x[0,4)x[0,4).
    # The remaining terms are:
    # j=4 (5th column of A, 5th row of B): i in [0,4), k in [0,6). 4*6=24 terms.
    # k=4,5 (5th, 6th cols of B): i in [0,4), j in [0,5). 4*5*2=40 terms.
    # But we must avoid double counting.
    # Total terms = 49 (for 4x4x4) + terms where j=4 or k>=4.
    # Terms with j=4: i in [0,4), k in [0,6). 24 terms.
    # Terms with j<4, k>=4: i in [0,4), j in [0,4), k in [4,6). 4*4*2=32 terms.
    # Total = 49 + 24 + 32 = 105.
    
    # Can we find a rank 49 decomposition for 4x4x4?
    # Yes, it's a known result. But hardcoding 49 terms is error-prone.
    # However, we can try to construct it using Strassen recursively.
    # 4x4x4 can be decomposed using 2x2x2 Strassen blocks?
    # No, Strassen is for 2x2x2.
    # We can split 4x4x4 into 8 blocks of 2x2x2.
    # Using Strassen for each block would give 8*7 = 56 terms.
    # But we can share terms between blocks?
    # Actually, the rank of 4x4x4 is 49, which is less than 56.
    # So recursive Strassen gives 56, not 49.
    # 56 + 24 + 32 = 112.
    # 112 < 120. This is an improvement!
    
    # Let's implement the 56-rank decomposition for 4x4x4 using Strassen.
    # Split A (4x4) into 4 blocks of 2x2: A00, A01, A10, A11.
    # Split B (4x4) into 4 blocks of 2x2: B00, B01, B10, B11.
    # C (4x4) = sum A_ab * B_bc.
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # We can use Strassen for the 2x2x2 multiplication of the blocks.
    # But the blocks are 2x2 matrices.
    # The multiplication of 4x4 matrices is equivalent to 16 multiplications of 2x2 matrices?
    # No, it's 4 multiplications of 2x2 matrices if we use the block structure directly?
    # No, C00 = A00B00 + A01B10. This is a sum of two products.
    # We can compute A00B00 using Strassen (7 terms), A01B10 using Strassen (7 terms).
    # Total 14 terms for C00.
    # Similarly for C01, C10, C11.
    # Total 4 * 14 = 56 terms.
    
    # Let's build this 56-rank decomposition for the 4x4x4 part.
    # Then add the extra terms for j=4 and k>=4.
    
    def build_4x4x4_strassen():
        """
        Builds U, V, W for 4x4x4 with rank 56 using Strassen on 2x2 blocks.
        """
        rank = 56
        U = [[0] * rank for _ in range(16)] # 4*4
        V = [[0] * rank for _ in range(16)] # 4*4
        W = [[0] * rank for _ in range(16)] # 4*4
        
        # We have 4 block multiplications:
        # 1. A00 * B00 -> contributes to C00
        # 2. A01 * B10 -> contributes to C00
        # 3. A00 * B01 -> contributes to C01
        # 4. A01 * B11 -> contributes to C01
        # 5. A10 * B00 -> contributes to C10
        # 6. A11 * B10 -> contributes to C10
        # 7. A10 * B01 -> contributes to C11
        # 8. A11 * B11 -> contributes to C11
        
        # Each block multiplication is 2x2 * 2x2.
        # We use Strassen for each.
        
        # Helper to add a Strassen block
        def add_strassen_block(a_row, a_col, b_row, b_col, c_row, c_col, offset):
            """
            Adds a Strassen decomposition for A[a_row:a_row+2, a_col:a_col+2] * B[b_row:b_row+2, b_col:b_col+2]
            contributing to C[c_row:c_row+2, c_col:c_col+2].
            """
            sU, sV, sW = strassen_2x2()
            for r in range(4):
                for c in range(rank):
                    if sU[r][c] != 0:
                        # Map local r to global U index
                        # Local r is 0..3 corresponding to (i_local, j_local) in 0..1
                        # Global i = a_row + i_local, j = a_col + j_local
                        i_local, j_local = divmod(r, 2)
                        gi = a_row + i_local
                        gj = a_col + j_local
                        U[gi * 4 + gj][offset + c] = sU[r][c]
                        
                    if sV[r][c] != 0:
                        j_local, k_local = divmod(r, 2)
                        gj = b_row + j_local
                        gk = b_col + k_local
                        V[gj * 4 + gk][offset + c] = sV[r][c]
                        
                    if sW[r][c] != 0:
                        k_local, i_local = divmod(r, 2)
                        gk = c_col + k_local
                        gi = c_row + i_local
                        W[gk * 4 + gi][offset + c] = sW[r][c]

        # Block 1: A00 * B00 -> C00
        add_strassen_block(0, 0, 0, 0, 0, 0, 0)
        # Block 2: A01 * B10 -> C00
        add_strassen_block(0, 2, 2, 0, 0, 0, 7)
        # Block 3: A00 * B01 -> C01
        add_strassen_block(0, 0, 0, 2, 0, 2, 14)
        # Block 4: A01 * B11 -> C01
        add_strassen_block(0, 2, 2, 2, 0, 2, 21)
        # Block 5: A10 * B00 -> C10
        add_strassen_block(2, 0, 0, 0, 2, 0, 28)
        # Block 6: A11 * B10 -> C10
        add_strassen_block(2, 2, 2, 0, 2, 0, 35)
        # Block 7: A10 * B01 -> C11
        add_strassen_block(2, 0, 0, 2, 2, 2, 42)
        # Block 8: A11 * B11 -> C11
        add_strassen_block(2, 2, 2, 2, 2, 2, 49)

        return U, V, W

    # Now, extend to 4x5x6.
    # We have U (16x56), V (16x56), W (16x56) for the 4x4x4 part.
    # We need to add terms for:
    # 1. j=4 (5th column of A, 5th row of B). i in [0,4), k in [0,6).
    #    For each i, k, we need a term where U[i*5+4, c]=1, V[4*6+k, c]=1, W[k*4+i, c]=1.
    #    Note: In the 4x4x4 part, V had indices j*4+k. Now V has indices j*6+k.
    #    So we need to rebuild V and W with the new dimensions.
    
    # Let's rebuild the whole thing with correct dimensions.
    def build_full_decomposition():
        rank_444 = 56
        extra_rank = 0
        # Terms with j=4: i in [0,4), k in [0,6). 24 terms.
        extra_rank += 24
        # Terms with j<4, k>=4: i in [0,4), j in [0,4), k in [4,6). 32 terms.
        extra_rank += 32
        total_rank = rank_444 + extra_rank
        
        U = [[0] * total_rank for _ in range(4 * 5)] # 20
        V = [[0] * total_rank for _ in range(5 * 6)] # 30
        W = [[0] * total_rank for _ in range(6 * 4)] # 24
        
        # First, add the 4x4x4 part.
        U444, V444, W444 = build_4x4x4_strassen()
        
        for i in range(4):
            for j in range(4):
                for c in range(rank_444):
                    if U444[i * 4 + j][c] != 0:
                        U[i * 5 + j][c] = U444[i * 4 + j][c]
        
        for j in range(4):
            for k in range(4):
                for c in range(rank_444):
                    if V444[j * 4 + k][c] != 0:
                        V[j * 6 + k][c] = V444[j * 4 + k][c]
        
        for k in range(4):
            for i in range(4):
                for c in range(rank_444):
                    if W444[k * 4 + i][c] != 0:
                        W[k * 4 + i][c] = W444[k * 4 + i][c]
        
        col = rank_444
        
        # Add terms for j=4
        for i in range(4):
            for k in range(6):
                U[i * 5 + 4][col] = 1
                V[4 * 6 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
        
        # Add terms for j<4, k>=4
        for i in range(4):
            for j in range(4):
                for k in range(4, 6):
                    U[i * 5 + j][col] = 1
                    V[j * 6 + k][col] = 1
                    W[k * 4 + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}

    result = build_full_decomposition()
    
    # Verify
    if verify(result["U"], result["V"], result["W"]):
        best_result = result
        best_rank = len(result["U"][0])
    else:
        # If verification fails, fall back to schoolbook
        best_result = schoolbook()
        best_rank = n * m * p

    return best_result

DIMENSIONS = [4, 5, 6]
