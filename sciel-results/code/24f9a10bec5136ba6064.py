import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition for 4x4x4, 4x4x1, 4x1x4, 4x1x1
    # and combine them.
    
    # Helper to get schoolbook decomposition
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    # Helper to get Strassen decomposition for 2x2x2
    def strassen_222():
        # Returns U, V, W for 2x2x2 with rank 7
        # Indices: U (2*2=4), V (2*2=4), W (2*2=4)
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1; W[2][0] = 1; W[1][0] = -1 # C11 += M1, C22 += M1
        
        # M2 = (A21+A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1 # C21 += M2, C22 -= M2
        
        # M3 = A11(B12-B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1 # C12 += M3, C22 += M3
        
        # M4 = A22(B21-B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1 # C11 += M4, C21 += M4
        
        # M5 = (A11+A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[3][4] = 1 # C11 -= M5, C12 += M5
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1 # C22 += M6
        
        # M7 = (A12-A22)(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[2][6] = -1 # C11 += M7, C21 -= M7
        
        return U, V, W

    # Helper to get Strassen decomposition for 4x4x4
    def strassen_444():
        # Compose 4x4x4 from 2x2x2 blocks
        # 4x4 matrix is 2x2 blocks of 2x2 matrices
        # We need to map the 4x4 indices to the 2x2 block indices
        # U (4*4=16), V (4*4=16), W (4*4=16)
        # Rank 49
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        U2, V2, W2 = strassen_222()
        
        # The 4x4 multiplication is a 2x2 block multiplication
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        # Each Ci,j is a 2x2 matrix
        # Ci,j = sum_k Ak,l Bl,m ...
        
        # We iterate over the 7 Strassen products for the 2x2 block structure
        # And for each, we iterate over the 7 Strassen products for the 2x2 sub-blocks
        # Total 49 terms
        
        # Mapping of 4x4 index (i,j) to block (bi, bj) and local (li, lj)
        # i = 2*bi + li, j = 2*bj + lj
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                # U2, V2, W2 are for the 2x2 block level
                # U2[bi*2+bj][r1] is the coefficient for A block (bi,bj)
                # V2[bj*2+bk][r1] is the coefficient for B block (bj,bk)
                # W2[bk*2+bi][r1] is the coefficient for C block (bk,bi)
                
                for bi in range(2):
                    for bj in range(2):
                        coeff_u = U2[bi*2+bj][r1]
                        if coeff_u != 0:
                            # For each local index in the 2x2 block
                            for li in range(2):
                                for lj in range(2):
                                    # U index: (2*bi+li)*4 + (2*bj+lj)
                                    u_idx = (2*bi+li)*4 + (2*bj+lj)
                                    # V2 index for block (bj, bk)
                                    # We need to sum over bk? No, r1 is fixed.
                                    # The structure is:
                                    # Term = (Sum_bij A_bij U2_bij) * (Sum_bjk B_bjk V2_bjk) * ...
                                    # This is not quite right. The Strassen algorithm is linear.
                                    # The 4x4 Strassen is a tensor product of 2x2 Strassen.
                                    # T_444 = T_222 \otimes T_222
                                    # So U_444[i,j] = U_222[bi,bj] * U_222[li,lj]
                                    
                                    # Let's check the tensor product structure.
                                    # A_4x4 is a matrix of 2x2 blocks.
                                    # The multiplication is bilinear.
                                    # The rank-1 tensors for 4x4 are products of rank-1 tensors for 2x2?
                                    # Yes, if we view the 4x4 multiplication as a 2x2 multiplication of 2x2 matrices.
                                    # Each 2x2 multiplication uses rank 7.
                                    # So we have 7 "outer" products and 7 "inner" products.
                                    
                                    # U_444[i,j]^{(r1,r2)} = U_222[bi,bj]^{(r1)} * U_222[li,lj]^{(r2)}
                                    # V_444[j,k]^{(r1,r2)} = V_222[bj,bk]^{(r1)} * V_222[lj,lk]^{(r2)}
                                    # W_444[k,i]^{(r1,r2)} = W_222[bk,bi]^{(r1)} * W_222[lk,li]^{(r2)}
                                    
                                    # We need to sum over bk for V and W?
                                    # No, the indices are coupled.
                                    # For a fixed r1, r2, and bi, bj, lk, li, bk is determined by the tensor structure?
                                    # No, bk is an index in V and W.
                                    # We must sum over bk in the decomposition?
                                    # The decomposition is a sum of rank-1 tensors.
                                    # Each rank-1 tensor is defined by U, V, W vectors.
                                    # For the tensor product, the indices are:
                                    # i = 2*bi + li, j = 2*bj + lj, k = 2*bk + lk.
                                    # So for each r1, r2, bi, bj, bk, li, lj, lk, we have a term?
                                    # No, the rank is 7*7=49.
                                    # The indices i,j,k range over 0..3.
                                    # The term is non-zero only if the local indices match the Strassen pattern.
                                    
                                    # Let's iterate over all i,j,k and compute the coefficients.
                                    # This is easier than trying to map the loops.
                                    pass
                # This manual mapping is error-prone.
                # Let's use a simpler approach:
                # Generate the 4x4x4 tensor and decompose it? No, too slow.
                # Let's stick to the 85-rank decomposition but use a simpler 4x4x4 decomposition if Strassen is too complex to code correctly in short time.
                # Actually, schoolbook for 4x4x4 is rank 64.
                # 64 + 16 + 16 + 4 = 100. No improvement.
                # We NEED Strassen for 4x4x4 to get below 100.
                
        # Let's try to implement the tensor product logic correctly.
        # U_444[i,j] = sum_{bi,bj,li,lj} delta(i, 2bi+li) delta(j, 2bj+lj) U_222[bi,bj] U_222[li,lj]
        # This is not a sum, it's a product for a specific rank-1 term.
        # The rank-1 term (r1, r2) has:
        # U[i,j] = U_222[bi,bj]^{(r1)} U_222[li,lj]^{(r2)} if i=2bi+li, j=2bj+lj.
        # V[j,k] = V_222[bj,bk]^{(r1)} V_222[lj,lk]^{(r2)} if j=2bj+lj, k=2bk+lk.
        # W[k,i] = W_222[bk,bi]^{(r1)} W_222[lk,li]^{(r2)} if k=2bk+lk, i=2bi+li.
        
        # So for each r1, r2, we fill U, V, W for all i,j,k.
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i in range(4):
                    for j in range(4):
                        bi, li = divmod(i, 2)
                        bj, lj = divmod(j, 2)
                        u_val = U2[bi*2+bj][r1] * U2[li*2+lj][r2]
                        U[i*4+j][r] = u_val
                        
                        for k in range(4):
                            bk, lk = divmod(k, 2)
                            v_val = V2[bj*2+bk][r1] * V2[lj*2+lk][r2]
                            V[j*4+k][r] = v_val
                            
                            w_val = W2[bk*2+bi][r1] * W2[lk*2+li][r2]
                            W[k*4+i][r] = w_val
                            
        return U, V, W

    # Get the components
    U_444, V_444, W_444 = strassen_444()
    U_441, V_441, W_441 = schoolbook(4, 4, 1)
    U_414, V_414, W_414 = schoolbook(4, 1, 4)
    U_411, V_411, W_411 = schoolbook(4, 1, 1)
    
    # Combine
    # m=5, p=5
    # Split m into 4+1, p into 4+1
    # Block 1: j in 0..3, k in 0..3 (4x4x4)
    # Block 2: j in 0..3, k in 4   (4x4x1)
    # Block 3: j in 4   , k in 0..3 (4x1x4)
    # Block 4: j in 4   , k in 4   (4x1x1)
    
    rank_444 = len(U_444[0])
    rank_441 = len(U_441[0])
    rank_414 = len(U_414[0])
    rank_411 = len(U_411[0])
    
    total_rank = rank_444 + rank_441 + rank_414 + rank_411
    
    U = [[0] * total_rank for _ in range(4 * 5)]
    V = [[0] * total_rank for _ in range(5 * 5)]
    W = [[0] * total_rank for _ in range(5 * 4)]
    
    offset = 0
    # Add 4x4x4
    for i in range(4):
        for j in range(4):
            for r in range(rank_444):
                U[i*5+j][offset+r] = U_444[i*4+j][r]
    for j in range(4):
        for k in range(4):
            for r in range(rank_444):
                V[j*5+k][offset+r] = V_444[j*4+k][r]
    for k in range(4):
        for i in range(4):
            for r in range(rank_444):
                W[k*4+i][offset+r] = W_444[k*4+i][r]
    offset += rank_444
    
    # Add 4x4x1 (j in 0..3, k=4)
    for i in range(4):
        for j in range(4):
            for r in range(rank_441):
                U[i*5+j][offset+r] = U_441[i*4+j][r]
    for j in range(4):
        for r in range(rank_441):
            V[j*5+4][offset+r] = V_441[j*4+0][r] # p=1, so k=0 in local
    for i in range(4):
        for r in range(rank_441):
            W[4*4+i][offset+r] = W_441[0*4+i][r] # p=1, n=4, so k=0 in local
    offset += rank_441
    
    # Add 4x1x4 (j=4, k in 0..3)
    for i in range(4):
        for r in range(rank_414):
            U[i*5+4][offset+r] = U_414[i*1+0][r]
    for k in range(4):
        for r in range(rank_414):
            V[4*5+k][offset+r] = V_414[0*4+k][r]
    for k in range(4):
        for i in range(4):
            for r in range(rank_414):
                W[k*4+i][offset+r] = W_414[k*4+i][r]
    offset += rank_414
    
    # Add 4x1x1 (j=4, k=4)
    for i in range(4):
        for r in range(rank_411):
            U[i*5+4][offset+r] = U_411[i*1+0][r]
    for r in range(rank_411):
        V[4*5+4][offset+r] = V_411[0*1+0][r]
    for i in range(4):
        for r in range(rank_411):
            W[4*4+i][offset+r] = W_411[0*4+i][r]
            
    return {"U": U, "V": V, "W": W}
