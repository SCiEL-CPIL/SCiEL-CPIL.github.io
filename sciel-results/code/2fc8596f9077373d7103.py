import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Strassen factors for 2x2 x 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    S_U = np.array([
        [1,0,0,1],   # M1: a+d
        [0,0,1,1],   # M2: c+d
        [1,0,0,0],   # M3: a
        [0,0,0,1],   # M4: d
        [1,1,0,0],   # M5: a+b
        [-1,0,1,0],  # M6: c-a
        [0,1,0,-1],  # M7: b-d
    ], dtype=float)  # (7, 4)
    
    S_V = np.array([
        [1,0,0,1],   # M1: e+h
        [1,0,0,0],   # M2: e
        [0,1,0,-1],  # M3: f-h
        [0,0,1,-1],  # M4: g-e
        [0,0,0,1],   # M5: h
        [1,1,0,0],   # M6: e+f
        [0,0,1,1],   # M7: g+h
    ], dtype=float)  # (7, 4)
    
    S_W = np.array([
        [1,0,0,1],   # M1: C00+=1, C11+=1
        [0,0,1,-1],  # M2: C10+=1, C11-=1
        [0,1,0,1],   # M3: C01+=1, C11+=1
        [1,0,1,0],   # M4: C00+=1, C10+=1
        [-1,1,0,0],  # M5: C00-=1, C01+=1
        [0,0,0,1],   # M6: C11+=1
        [1,0,0,0],   # M7: C00+=1
    ], dtype=float)  # (7, 4)
    
    # Schoolbook for 1x1 x 1x4
    # A is 1x1, B is 1x4, C is 1x4
    # C[k] = A[0] * B[k]
    SB_U = np.array([[1,1,1,1]], dtype=float)  # (1, 4)
    SB_V = np.eye(4, dtype=float)  # (4, 4)
    SB_W = np.eye(4, dtype=float)  # (4, 4)
    
    # For each block (bh=0,1), construct 2x2 x 2x8 decomposition
    # Using Kronecker product of Strassen (rank 7) and schoolbook 1x1x1x4 (rank 4)
    # Total rank per block: 7*4 = 28
    # Total rank: 2*28 = 56
    
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    for bh in range(2):
        # 2x2 x 2x8 block
        # A: 2x2 (4 entries), B: 2x8 (16 entries), C: 2x8 (16 entries)
        
        # Kronecker product: U = S_U^T ⊗ SB_U^T, etc.
        # S_U is (7,4), we need (4, 7) for U
        # SB_U is (1,4), we need (4,1) for... 
        
        # Let me think in terms of the tensor factors.
        # For the 2x2x2x2 tensor: U1[i, r], V1[j, r], W1[k, r] where i,j,k in 0..3, r in 0..6
        # For the 1x1x1x4 tensor: U2[i2, r2], V2[j2, r2], W2[k2, r2] where i2 in 0, j2,k2 in 0..3, r2 in 0..3
        
        # Kronecker: U[i, i2, r, r2] = U1[i, r] * U2[i2, r2]
        # Since i2=0 always: U[i, r, r2] = U1[i, r] * 1
        
        # In the full 2x4x4x8 tensor:
        # A_idx = i*4 + j where i in 0..1, j in 0..3
        # B_idx = j*8 + k where j in 0..3, k in 0..7
        # C_idx = k*2 + i where k in 0..7, i in 0..1
        
        # For block bh, j = 2*bh + jl where jl in 0..1
        # A_idx = i*4 + 2*bh + jl
        # B_idx = (2*bh+jl)*8 + k
        # C_idx = k*2 + i
        
        # In the 2x2x2x8 subproblem:
        # A_sub[i, jl] = A[i, 2*bh+jl]
        # B_sub[jl, k] = B[2*bh+jl, k]
        # C_sub[i, k] = C[i, k]
        
        # The 2x2x2x8 tensor: T[A_sub_idx, B_sub_idx, C_sub_idx]
        # A_sub_idx = i*2 + jl (i in 0..1, jl in 0..1)
        # B_sub_idx = jl*8 + k (jl in 0..1, k in 0..7)
        # C_sub_idx = i*8 + k (i in 0..1, k in 0..7)
        
        # Using Kronecker: 2x2x2x8 = (2x2x2x2) ⊗ (1x1x1x4)
        # A: (2x2) ⊗ (1x1) = 2x2
        # B: (2x2) ⊗ (1x4) = 2x8
        # C: (2x2) ⊗ (1x4) = 2x8
        
        # For the Kronecker product factors:
        # U_kron[A_idx, r1, r2] = U1[A_idx1, r1] * U2[A_idx2, r2]
        # where A_idx = A_idx1 * 1 + A_idx2 (since 1x1 has only 1 entry)
        # A_idx1 in 0..3 (for 2x2), A_idx2 = 0 (for 1x1)
        
        # So U_kron[A_idx1, r1, r2] = S_U[r1, A_idx1] * SB_U[r2, 0] = S_U[r1, A_idx1]
        
        # V_kron[B_idx, r1, r2] = V1[B_idx1, r1] * V2[B_idx2, r2]
        # B_idx = B_idx1 * 4 + B_idx2 (since 1x4 has 4 entries)
        # B_idx1 in 0..3 (for 2x2), B_idx2 in 0..3 (for 1x4)
        # V_kron[B_idx1*4+B_idx2, r1, r2] = S_V[r1, B_idx1] * SB_V[r2, B_idx2]
        
        # W_kron[C_idx, r1, r2] = W1[C_idx1, r1] * W2[C_idx2, r2]
        # C_idx = C_idx1 * 4 + C_idx2
        # C_idx1 in 0..3 (for 2x2), C_idx2 in 0..3 (for 1x4)
        # W_kron[C_idx1*4+C_idx2, r1, r2] = S_W[r1, C_idx1] * SB_W[r2, C_idx2]
        
        block_rank = 28
        U_block = np.zeros((4, block_rank))
        V_block = np.zeros((16, block_rank))
        W_block = np.zeros((16, block_rank))
        
        for r1 in range(7):
            for r2 in range(4):
                col = r1 * 4 + r2
                # U
                for i in range(4):
                    U_block[i, col] = S_U[r1, i] * SB_U[0, r2]
                # V
                for b1 in range(4):
                    for b2 in range(4):
                        V_block[b1*4+b2, col] = S_V[r1, b1] * SB_V[r2, b2]
                # W
                for c1 in range(4):
                    for c2 in range(4):
                        W_block[c1*4+c2, col] = S_W[r1, c1] * SB_W[r2, c2]
        
        # Map to full tensor indices
        for i in range(2):
            for jl in range(2):
                a_idx = i * 4 + (2*bh + jl)
                a_sub_idx = i * 2 + jl
                U_full_row = U_block[a_sub_idx, :]
                # Add to U_list at row a_idx
                if len(U_list) == 0:
                    U_list.append(np.zeros((8, block_rank)))
                U_list[bh][a_idx, :] = U_full_row
        
        for jl in range(2):
            for k in range(8):
                b_idx = (2*bh + jl) * 8 + k
                b_sub_idx = jl * 8 + k
                V_full_row = V_block[b_sub_idx, :]
                if len(V_list) == 0:
                    V_list.append(np.zeros((32, block_rank)))
                V_list[bh][b_idx, :] = V_full_row
        
        for i in range(2):
            for k in range(8):
                c_idx = k * 2 + i
                c_sub_idx = i * 8 + k
                W_full_row = W_block[c_sub_idx, :]
                if len(W_list) == 0:
                    W_list.append(np.zeros((16, block_rank)))
                W_list[bh][c_idx, :] = W_full_row
    
    # Combine blocks
    U_full = np.hstack([U_list[0], U_list[1]])  # (8, 56)
    V_full = np.hstack([V_list[0], V_list[1]])  # (32, 56)
    W_full = np.hstack([W_list[0], W_list[1]])  # (16, 56)
    
    # Convert to lists
    U = U_full.tolist()
    V = V_full.tolist()
    W = W_full.tolist()
    
    return {"U": U, "V": V, "W": W}
