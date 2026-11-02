import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part, Rank 16 for 4x4x1 part. Total 65.
    rank = 65
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: 4x4x4 multiplication using Strassen-Strassen (Rank 49)
    # We split A (4x4) and B (4x4) into 2x2 blocks.
    # Strassen on blocks uses 7 products.
    # Each product is 2x2x2, using Strassen (Rank 7).
    # Total 49.
    
    # Strassen 2x2x2 factors
    # U_2x2, V_2x2, W_2x2 for rank 7
    # Indices for 2x2: i in 0..1, j in 0..1, k in 0..1
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # Strassen factors for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to the global indices.
    # Block (bi, bj) where bi, bj in 0..1.
    # A block A_bi,bj has rows 2*bi .. 2*bi+1, cols 2*bj .. 2*bj+1.
    # Global i = 2*bi + i_local.
    # Global j = 2*bj + j_local.
    
    # Strassen 2x2 factors (local)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # r=0: M1
    U_2[0,0] = 1; U_2[3,0] = 1 # A11, A22 (indices 0, 3 in 2x2 flat? 0:00, 1:01, 2:10, 3:11)
    # Wait, flat index i*2+j.
    # A11: i=0,j=0 -> 0. A22: i=1,j=1 -> 3.
    V_2[0,0] = 1; V_2[3,0] = 1 # B11, B22
    W_2[0,0] = 1; W_2[3,0] = 1; W_2[1,0] = -1; W_2[2,0] = -1 # C11, C22, C12, C21?
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    
    # Let's use a known correct set of Strassen factors.
    # Ref: https://en.wikipedia.org/wiki/Strassen_algorithm
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U (A factors)
    # r=0: A11+A22
    U_2[0,0]=1; U_2[3,0]=1
    # r=1: A21+A22
    U_2[2,1]=1; U_2[3,1]=1
    # r=2: A11
    U_2[0,2]=1
    # r=3: A22
    U_2[3,3]=1
    # r=4: A11+A12
    U_2[0,4]=1; U_2[1,4]=1
    # r=5: A21-A11
    U_2[2,5]=1; U_2[0,5]=-1
    # r=6: A12-A22
    U_2[1,6]=1; U_2[3,6]=-1
    
    # V (B factors)
    # r=0: B11+B22
    V_2[0,0]=1; V_2[3,0]=1
    # r=1: B11
    V_2[0,1]=1
    # r=2: B12-B22
    V_2[1,2]=1; V_2[3,2]=-1
    # r=3: B21-B11
    V_2[2,3]=1; V_2[0,3]=-1
    # r=4: B22
    V_2[3,4]=1
    # r=5: B11+B12
    V_2[0,5]=1; V_2[1,5]=1
    # r=6: B21+B22
    V_2[2,6]=1; V_2[3,6]=1
    
    # W (C factors)
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    # W indices: k*2+i.
    # C11: i=0,k=0 -> 0.
    # C12: i=0,k=1 -> 2.
    # C21: i=1,k=0 -> 1.
    # C22: i=1,k=1 -> 3.
    
    W_2[0,0]=1  # C11 += M1
    W_2[2,0]=1  # C12 += M1? No.
    # Let's check C12. C12 = M3 + M5.
    # C11 = M1 - M4 + M6 - M7.
    # C21 = M2 + M4.
    # C22 = M1 - M3 + M5 + M7.
    
    # r=0 (M1): C11 += 1, C22 += 1
    W_2[0,0]=1; W_2[3,0]=1
    # r=1 (M2): C21 += 1
    W_2[1,1]=1
    # r=2 (M3): C12 += 1, C22 -= 1
    W_2[2,2]=1; W_2[3,2]=-1
    # r=3 (M4): C11 -= 1, C21 += 1
    W_2[0,3]=-1; W_2[1,3]=1
    # r=4 (M5): C12 += 1, C22 += 1
    W_2[2,4]=1; W_2[3,4]=1
    # r=5 (M6): C11 += 1
    W_2[0,5]=1
    # r=6 (M7): C11 -= 1, C22 += 1
    W_2[0,6]=-1; W_2[3,6]=1
    
    # Now combine 4 block multiplications using Strassen on blocks.
    # Block indices bi, bj in 0..1.
    # Global i = 2*bi + il.
    # Global j = 2*bj + jl.
    # Global k = 2*bk + kl.
    
    # Block Strassen factors (for the 2x2 block matrix multiplication)
    # U_block (4x4 blocks), V_block, W_block.
    # Rank 7.
    U_b = np.zeros((4, 7))
    V_b = np.zeros((4, 7))
    W_b = np.zeros((4, 7))
    
    # Same structure as U_2 but for blocks.
    # Block indices: 0:11, 1:12, 2:21, 3:22.
    
    U_b[0,0]=1; U_b[3,0]=1 # A11+A22
    U_b[2,1]=1; U_b[3,1]=1 # A21+A22
    U_b[0,2]=1 # A11
    U_b[3,3]=1 # A22
    U_b[0,4]=1; U_b[1,4]=1 # A11+A12
    U_b[2,5]=1; U_b[0,5]=-1 # A21-A11
    U_b[1,6]=1; U_b[3,6]=-1 # A12-A22
    
    V_b[0,0]=1; V_b[3,0]=1 # B11+B22
    V_b[0,1]=1 # B11
    V_b[1,2]=1; V_b[3,2]=-1 # B12-B22
    V_b[2,3]=1; V_b[0,3]=-1 # B21-B11
    V_b[3,4]=1 # B22
    V_b[0,5]=1; V_b[1,5]=1 # B11+B12
    V_b[2,6]=1; V_b[3,6]=1 # B21+B22
    
    W_b[0,0]=1; W_b[3,0]=1 # C11, C22
    W_b[1,1]=1 # C21
    W_b[2,2]=1; W_b[3,2]=-1 # C12, C22
    W_b[0,3]=-1; W_b[1,3]=1 # C11, C21
    W_b[2,4]=1; W_b[3,4]=1 # C12, C22
    W_b[0,5]=1 # C11
    W_b[0,6]=-1; W_b[3,6]=1 # C11, C22
    
    # Iterate over block Strassen terms (rb) and inner Strassen terms (ri)
    col = 0
    for rb in range(7):
        for ri in range(7):
            # For each block term rb, we have a product of blocks.
            # The block product is computed using inner Strassen (ri).
            # The global U, V, W are constructed by combining block indices and local indices.
            
            # U_global[i,j, col]
            # i = 2*bi + il, j = 2*bj + jl.
            # U_block[bi*bj? No, block index is flat.
            # Block index b = bi*2 + bj.
            # U_b[b, rb] gives the coefficient for block b in term rb.
            # For each non-zero block coefficient, we add the local U_2 pattern.
            
            # We need to iterate over all blocks b where U_b[b, rb] != 0.
            for b in range(4):
                coeff_u = U_b[b, rb]
                if coeff_u == 0: continue
                bi = b // 2
                bj = b % 2
                
                # For this block, we add U_2 pattern scaled by coeff_u * V_b? No.
                # The structure is:
                # U_global = sum_rb sum_ri U_b[bi,bj, rb] * U_2[il,jl, ri]
                # This is a tensor product of the block decomposition and the local decomposition.
                
                # Wait, the block decomposition is for the *matrix* of blocks.
                # The local decomposition is for the *elements* of the blocks.
                # So yes, it's a product.
                
                for il in range(2):
                    for jl in range(2):
                        u_local = U_2[il*2+jl, ri]
                        if u_local == 0: continue
                        i_global = 2*bi + il
                        j_global = 2*bj + jl
                        U[i_global * m + j_global, col] += coeff_u * u_local
            
            # Similarly for V
            for b in range(4):
                coeff_v = V_b[b, rb]
                if coeff_v == 0: continue
                # V block index is bk*2 + bl?
                # V_b index b corresponds to block B_b.
                # In block multiplication, B is split into B11, B12, B21, B22.
                # Index b: 0:11, 1:12, 2:21, 3:22.
                # So b = bk*2 + bl?
                # B11: bk=0, bl=0.
                # B12: bk=0, bl=1.
                # B21: bk=1, bl=0.
                # B22: bk=1, bl=1.
                # So b = bk*2 + bl.
                bk = b // 2
                bl = b % 2
                
                for jl in range(2):
                    for kl in range(2):
                        v_local = V_2[jl*2+kl, ri]
                        if v_local == 0: continue
                        j_global = 2*bk + jl # Wait, j is the column of A, row of B.
                        # In block multiplication, A is (bi,bj), B is (bk,bl).
                        # Product is C(bi,bl).
                        # The j index in V is the column of B? No, V is (m*p).
                        # V index is j*p + k.
                        # j is the shared index.
                        # In block B(bk,bl), the rows are 2*bk.., cols are 2*bl..
                        # So j_global = 2*bk + jl.
                        # k_global = 2*bl + kl.
                        j_global = 2*bk + jl
                        k_global = 2*bl + kl
                        V[j_global * p + k_global, col] += coeff_v * v_local
            
            # Similarly for W
            for b in range(4):
                coeff_w = W_b[b, rb]
                if coeff_w == 0: continue
                # W block index is C(bi,bl).
                # b = bi*2 + bl.
                bi = b // 2
                bl = b % 2
                
                for kl in range(2):
                    for il in range(2):
                        w_local = W_2[kl*2+il, ri]
                        if w_local == 0: continue
                        k_global = 2*bl + kl
                        i_global = 2*bi + il
                        W[k_global * n + i_global, col] += coeff_w * w_local
            
            col += 1
            
    # Part 2: 4x4x1 multiplication (Schoolbook, Rank 16)
    # B is 4x1.
    # i in 0..3, j in 0..3, k=0.
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 0, col] = 1
            W[0 * n + i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
