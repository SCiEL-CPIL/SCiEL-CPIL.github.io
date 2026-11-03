import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # We decompose p=9 into 4 + 4 + 1
    # Part 1: 4x4x4 multiplication (Strassen rank 49)
    # Part 2: 4x4x4 multiplication (Strassen rank 49)
    # Part 3: 4x4x1 multiplication (Schoolbook rank 16)
    
    # Step 1: Construct Strassen 2x2x2 decomposition
    # Indices: i,j,k in {0,1}
    # U shape (2*2, 7) = (4, 7)
    # V shape (2*2, 7) = (4, 7)
    # W shape (2*2, 7) = (4, 7)
    
    U_222 = np.zeros((4, 7), dtype=int)
    V_222 = np.zeros((4, 7), dtype=int)
    W_222 = np.zeros((4, 7), dtype=int)
    
    # Strassen formulas for 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the tensor decomposition T_ijk = sum_r u_ir v_jr w_kr
    # where C_ik = sum_j A_ij B_jk.
    # Note: My standard Strassen derivation above was for C_ik = sum_j A_ij B_jk.
    # The indices in the problem are:
    # U[i*m+j], V[j*p+k], W[k*n+i]
    # C[i,k] = sum_j U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Let's define the rank-1 terms for 2x2x2 (n=2, m=2, p=2)
    # r=0..6
    
    # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=-1? No, C00+=M1, C11-=M1?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to construct U, V, W such that sum_r U[:,r] V[:,r] W[:,r] = T
    # T[i,j,k] = 1 if k is the output col, i is output row, j is inner index?
    # No, T[i,j,k] corresponds to coefficient of A_i,j and B_j,k in C_i,k.
    # So T[i,j,k] = 1.
    # Decomposition: T = sum u_r (x) v_r (y) w_r (z)
    # u_r[i,j], v_r[j,k], w_r[k,i]
    
    # Let's fill the 7 terms.
    # Term 0 (M1): (A00+A11)(B00+B11)
    # u0: A00=1, A11=1 -> U[0,0]=1, U[3,0]=1 (indices 0*2+0, 1*2+1)
    # v0: B00=1, B11=1 -> V[0,0]=1, V[3,0]=1
    # w0: C00+=1, C11+=1? 
    # C00 gets M1. C11 gets M1.
    # So w0[0,0]=1 (k=0,i=0 -> index 0*2+0=0), w0[1,1]=1 (k=1,i=1 -> index 1*2+1=3)
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # Term 1 (M2): (A10+A11) B00
    # u1: A10=1, A11=1 -> U[2,1]=1 (1*2+0), U[3,1]=1 (1*2+1)
    # v1: B00=1 -> V[0,1]=1
    # w1: C10+=1 -> w1[0,1]=1 (k=0,i=1 -> index 0*2+1=1)
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[1, 1] = 1
    
    # Term 2 (M3): A00 (B01-B11)
    # u2: A00=1 -> U[0,2]=1
    # v2: B01=1, B11=-1 -> V[1,2]=1 (0*2+1), V[3,2]=-1 (1*2+1)
    # w2: C01+=1 -> w2[1,0]=1 (k=1,i=0 -> index 1*2+0=2)
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[2, 2] = 1
    
    # Term 3 (M4): A11 (B10-B00)
    # u3: A11=1 -> U[3,3]=1
    # v3: B10=1, B00=-1 -> V[2,3]=1 (1*2+0), V[0,3]=-1 (0*2+0)
    # w3: C00+=1, C10+=1? 
    # C00+=M4, C10+=M4.
    # w3: C00 (k=0,i=0) idx 0, C10 (k=0,i=1) idx 1
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[0, 3] = 1; W_222[1, 3] = 1
    
    # Term 4 (M5): (A00+A01) B11
    # u4: A00=1, A01=1 -> U[0,4]=1, U[1,4]=1 (0*2+1)
    # v4: B11=1 -> V[3,4]=1
    # w4: C00-=1, C01+=1
    # w4: C00 (idx 0) = -1, C01 (idx 2) = 1
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = -1; W_222[2, 4] = 1
    
    # Term 5 (M6): (A10-A00) (B00+B01)
    # u5: A10=1, A00=-1 -> U[2,5]=1, U[0,5]=-1
    # v5: B00=1, B01=1 -> V[0,5]=1, V[1,5]=1
    # w5: C11+=1 -> w5[3,5]=1 (k=1,i=1 idx 3)
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[3, 5] = 1
    
    # Term 6 (M7): (A01-A11) (B10+B11)
    # u6: A01=1, A11=-1 -> U[1,6]=1, U[3,6]=-1
    # v6: B10=1, B11=1 -> V[2,6]=1, V[3,6]=1
    # w6: C00+=1 -> w6[0,6]=1
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1
    
    # Step 2: Lift 2x2x2 to 4x4x4 using Kronecker product
    # U_444 = np.kron(U_222, U_222) ?
    # Indices: 
    # U_222[i2, j2, r2]
    # U_444[(i1*2+i2)*2 + (j1*2+j2), r1*7+r2]
    # This corresponds to A[i1*2+i2, j1*2+j2] = sum U_222[i2, j2, r2] ...
    # Yes, Kronecker product works for tensor decomposition of block matrices.
    
    U_444_1 = np.kron(U_222, U_222)
    V_444_1 = np.kron(V_222, V_222)
    W_444_1 = np.kron(W_222, W_222)
    
    # We need two copies for the two 4x4x4 parts (cols 0-3 and 4-7 of B)
    # The second part is identical in structure, just shifted in the V and W indices?
    # No, the decomposition is local to the 4x4x4 block.
    # The "shift" is handled by the fact that we are adding these to the global U, V, W.
    # V index for part 1 (cols 0-3): j*4 + k.
    # V index for part 2 (cols 4-7): j*4 + (k+4).
    # W index for part 1 (cols 0-3): k*4 + i.
    # W index for part 2 (cols 4-7): (k+4)*4 + i.
    
    # However, U is the same for both parts (since A is the same).
    # So we can just repeat the U columns? No, the rank is additive.
    # We need distinct rank indices for part 1 and part 2.
    
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_444 + rank_441
    
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((36, total_rank), dtype=int) # m*p = 4*9
    W = np.zeros((36, total_rank), dtype=int) # p*n = 9*4
    
    # Part 1: Cols 0-3 of B
    # U[:, 0:49] = U_444_1
    U[:, 0:49] = U_444_1
    
    # V: m*p = 4*9. Indices j*9+k.
    # For part 1, k is 0..3.
    # V_444_1 has shape (16, 49). Rows are j*2+k2.
    # We need to map row index r_v (0..15) to V index j*9+k.
    # r_v = j*2 + k.
    # V[r_v*? + ?]
    # Actually, V_444_1 row index is j*2+k (where j is 0..3, k is 0..3? No, j is 0..3 in 4x4)
    # In 4x4x4, V index is j*4+k.
    # But V_444_1 was constructed from 2x2x2 Kronecker.
    # V_222 row index is j2*2+k2.
    # V_444_1 row index is (j1*2+j2)*2 + (k1*2+k2) = j*4+k.
    # So V_444_1[i, r] corresponds to V_444[i, r] for 4x4x4.
    # We need to place this into V (4*9) for k=0..3.
    # V_444 index is j*4+k.
    # Global V index is j*9+k.
    # So for each row idx_v in V_444_1 (0..15):
    # j = idx_v // 4
    # k = idx_v % 4
    # global_idx = j * 9 + k
    for idx_v in range(16):
        j = idx_v // 4
        k = idx_v % 4
        global_idx = j * 9 + k
        V[global_idx, 0:49] = V_444_1[idx_v, :]
        
    # W: p*n = 9*4. Indices k*4+i.
    # W_444_1 has shape (16, 49). Row index is k*4+i.
    # For part 1, k is 0..3.
    # Global W index is k*4+i. (Since k is 0..3, it's the same)
    W[0:16, 0:49] = W_444_1
    
    # Part 2: Cols 4-7 of B
    # U is same.
    U[:, 49:98] = U_444_1
    
    # V: k is 4..7.
    for idx_v in range(16):
        j = idx_v // 4
        k = idx_v % 4
        global_k = k + 4
        global_idx = j * 9 + global_k
        V[global_idx, 49:98] = V_444_1[idx_v, :]
        
    # W: k is 4..7.
    # Global W index is (k+4)*4 + i = k*4 + 16 + i.
    # W_444_1 row index is k*4+i.
    for idx_w in range(16):
        k = idx_w // 4
        i = idx_w % 4
        global_idx = (k + 4) * 4 + i
        W[global_idx, 49:98] = W_444_1[idx_w, :]
        
    # Part 3: Cols 8 of B (4x4x1)
    # Schoolbook rank 16.
    # U: 16x16 identity-like.
    # V: 4x16 (j*1+0 -> j).
    # W: 4x16 (0*4+i -> i).
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            # U index i*4+j
            U[i * 4 + j, 98 + r] = 1
            # V index j*1+8 = j+8
            V[j + 8, 98 + r] = 1
            # W index 8*4+i = 32+i
            W[32 + i, 98 + r] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
