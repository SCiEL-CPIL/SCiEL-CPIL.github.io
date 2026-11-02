import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5
    rank = 0
    
    # 1. Construct Strassen decomposition for 2x2x2
    # U, V, W are lists of 4-element lists (vectors for the 4 entries of the 2x2 matrix)
    # Indexing: i*2 + j for A, j*2 + k for B, k*2 + i for C
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (index k*2+i)
    
    sU = []
    sV = []
    sW = []
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    sU.append([1, 0, 0, 1])
    sV.append([1, 0, 0, 1])
    sW.append([1, 0, 0, 1]) # C00, C11
    
    # M2 = (A10+A11)B00 -> C10, -C11
    sU.append([0, 0, 1, 1])
    sV.append([1, 0, 0, 0])
    sW.append([0, 0, 1, -1]) # C10, C11
    
    # M3 = A00(B01-B11) -> C01, C11
    sU.append([1, 0, 0, 0])
    sV.append([0, 1, 0, -1])
    sW.append([0, 1, 0, 1]) # C01, C11
    
    # M4 = A11(B10-B00) -> C00, C10
    sU.append([0, 0, 0, 1])
    sV.append([-1, 0, 1, 0])
    sW.append([1, 0, 1, 0]) # C00, C10
    
    # M5 = (A00+A01)B11 -> -C00, C01
    sU.append([1, 1, 0, 0])
    sV.append([0, 0, 0, 1])
    sW.append([-1, 1, 0, 0]) # C00, C01
    
    # M6 = (A10-A00)(B00+B01) -> C11
    sU.append([-1, 0, 1, 0])
    sV.append([1, 1, 0, 0])
    sW.append([0, 0, 0, 1]) # C11
    
    # M7 = (A01-A11)(B10+B11) -> C00
    sU.append([0, 1, 0, -1])
    sV.append([0, 0, 1, 1])
    sW.append([1, 0, 0, 0]) # C00
    
    # 2. Construct Block Strassen coefficients for 4x4x4
    # Blocks A00, A01, A10, A11 (indices 0,1,2,3)
    # Blocks B00, B01, B10, B11 (indices 0,1,2,3)
    # Blocks C00, C01, C10, C11 (indices 0,1,2,3)
    
    block_coeffs = []
    # Each entry: (u_dict, v_dict, w_dict)
    # u_dict: {block_idx: coeff}
    
    # M1
    block_coeffs.append((
        {0: 1, 3: 1}, 
        {0: 1, 3: 1}, 
        {0: 1, 3: 1}
    ))
    # M2
    block_coeffs.append((
        {2: 1, 3: 1}, 
        {0: 1}, 
        {2: 1, 3: -1}
    ))
    # M3
    block_coeffs.append((
        {0: 1}, 
        {1: 1, 3: -1}, 
        {1: 1, 3: 1}
    ))
    # M4
    block_coeffs.append((
        {3: 1}, 
        {2: 1, 0: -1}, 
        {0: 1, 2: 1}
    ))
    # M5
    block_coeffs.append((
        {0: 1, 1: 1}, 
        {3: 1}, 
        {0: -1, 1: 1}
    ))
    # M6
    block_coeffs.append((
        {2: 1, 0: -1}, 
        {0: 1, 1: 1}, 
        {3: 1}
    ))
    # M7
    block_coeffs.append((
        {1: 1, 3: -1}, 
        {2: 1, 3: 1}, 
        {0: 1}
    ))
    
    # Initialize U, V, W for the 4x4x4 part
    # U size: 16 x 49
    # V size: 16 x 49
    # W size: 16 x 49
    rank_444 = 49
    U_444 = [[0]*rank_444 for _ in range(16)]
    V_444 = [[0]*rank_444 for _ in range(16)]
    W_444 = [[0]*rank_444 for _ in range(16)]
    
    for r in range(7):
        u_block, v_block, w_block = block_coeffs[r]
        for s in range(7):
            col = r * 7 + s
            u_vec = sU[s]
            v_vec = sV[s]
            w_vec = sW[s]
            
            # Expand u_vec to U_444
            for block_idx, coeff in u_block.items():
                if coeff == 0: continue
                # Block block_idx corresponds to rows block_idx*4 to block_idx*4+3 in U
                # Wait, A is 4x4. Block (a,b) where a is row block, b is col block.
                # Index in U is i*4 + j.
                # Block (a,b) has i in a*2..a*2+1, j in b*2..b*2+1.
                # So rows are a*4 + b*2 + di + dj?
                # Let's map carefully.
                # A_{ij} with i=a*2+di, j=b*2+dj.
                # U index = i*4 + j = (a*2+di)*4 + (b*2+dj) = a*8 + b*2 + di*4 + dj.
                
                for di in range(2):
                    for dj in range(2):
                        row_u = (block_idx // 2) * 8 + (block_idx % 2) * 2 + di * 4 + dj
                        U_444[row_u][col] += coeff * u_vec[di * 2 + dj]
                        
            # Expand v_vec to V_444
            # B_{jk} with j is row, k is col.
            # Block (j_block, k_block).
            # V index = j*4 + k? No, V shape is m*p = 4*4 = 16?
            # Wait, m=4, p=4 for the 4x4x4 part.
            # V index is j*p + k = j*4 + k.
            # Block (jb, kb). j=jb*2+di, k=kb*2+dj.
            # V index = (jb*2+di)*4 + (kb*2+dj) = jb*8 + kb*2 + di*4 + dj.
            
            for block_idx, coeff in v_block.items():
                if coeff == 0: continue
                for di in range(2):
                    for dj in range(2):
                        row_v = (block_idx // 2) * 8 + (block_idx % 2) * 2 + di * 4 + dj
                        V_444[row_v][col] += coeff * v_vec[di * 2 + dj]
                        
            # Expand w_vec to W_444
            # C_{ik}. W index = k*n + i = k*4 + i.
            # Block (kb, ib). i=ib*2+di, k=kb*2+dj.
            # W index = (kb*2+dj)*4 + (ib*2+di) = kb*8 + ib*2 + dj*4 + di.
            
            for block_idx, coeff in w_block.items():
                if coeff == 0: continue
                for di in range(2):
                    for dj in range(2):
                        row_w = (block_idx % 2) * 8 + (block_idx // 2) * 2 + dj * 4 + di
                        W_444[row_w][col] += coeff * w_vec[di * 2 + dj]
                        
    # 3. Construct 4x4x1 part (Matrix-Vector)
    # A (4x4) * b (4x1) -> c (4x1)
    # Rank 16.
    rank_441 = 16
    U_441 = [[0]*rank_441 for _ in range(16)]
    V_441 = [[0]*rank_441 for _ in range(20)] # m*p = 4*5
    W_441 = [[0]*rank_441 for _ in range(20)] # p*n = 5*4
    
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            # U: 1 at i*4+j
            U_441[i * 4 + j][col] = 1
            # V: 1 at j*5 + 4 (5th column of B)
            V_441[j * 5 + 4][col] = 1
            # W: 1 at 4*4 + i (5th row of C? No, k=4, i=i)
            W_441[4 * 4 + i][col] = 1
            
    # 4. Combine
    total_rank = rank_444 + rank_441
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(20)]
    W = [[0]*total_rank for _ in range(20)]
    
    # Copy 444 part
    for i in range(16):
        U[i] = U_444[i] + [0]*rank_441
        V[i] = V_444[i] + [0]*rank_441 # V_444 is 16x49, but V needs 20 rows.
        # Wait, V_444 was constructed for p=4.
        # The 444 part only uses the first 4 columns of B.
        # So V_444 rows 0-15 correspond to B[0:4, 0:4].
        # The 5th column (index 4) of B is not touched in 444 part.
        # So we need to extend V_444 to 20 rows, with zeros for row 16-19.
        # Actually, V_444 is 16x49. The global V is 20xTotalRank.
        # Rows 0-15 of global V get V_444.
        # Rows 16-19 of global V get 0.
        
        W[i] = W_444[i] + [0]*rank_441 # W_444 is 16x49. Global W is 20xTotalRank.
        # Rows 0-15 of global W get W_444.
        # Rows 16-19 of global W get 0.
        
    # Copy 441 part
    for i in range(16):
        for c in range(rank_441):
            U[i][rank_444 + c] = U_441[i][c]
            
    for i in range(20):
        for c in range(rank_441):
            V[i][rank_444 + c] = V_441[i][c]
            
    for i in range(20):
        for c in range(rank_441):
            W[i][rank_444 + c] = W_441[i][c]
            
    # Verification
    # Random A (4x4), B (4x5)
    A = np.random.randint(-2, 3, (4, 4))
    B = np.random.randint(-2, 3, (4, 5))
    C_true = A @ B
    
    C_calc = np.zeros((4, 5))
    
    for r in range(total_rank):
        u_vec = np.array(U[:, r])
        v_vec = np.array(V[:, r])
        w_vec = np.array(W[:, r])
        
        # u_vec shape 16 -> A (4x4)
        A_u = u_vec.reshape(4, 4).T @ A # Sum U[i,j,r] A[i,j]
        # Wait, U[i*4+j, r] * A[i,j].
        # u_vec is 16. reshape(4,4) gives rows i, cols j.
        # Element-wise product with A, then sum.
        A_u = np.sum(u_vec.reshape(4, 4) * A)
        
        # v_vec shape 20 -> B (4x5)
        # V[j*5+k, r] * B[j,k]
        B_v = np.sum(v_vec.reshape(4, 5) * B)
        
        # w_vec shape 20 -> C (4x5)
        # W[k*4+i, r] * C[k,i]
        # We want to compute C_calc.
        # C_calc[i,k] += w_vec[k*4+i] * A_u * B_v
        
        w_mat = w_vec.reshape(5, 4)
        # w_mat[k, i]
        C_calc += A_u * B_v * w_mat.T # w_mat.T is 4x5, [i, k]
        
    if not np.allclose(C_calc, C_true, atol=1e-9):
        print("Verification Failed")
        print("C_true:", C_true)
        print("C_calc:", C_calc)
        # Fallback to schoolbook if failed
        rank_school = 80
        U_s = [[0]*rank_school for _ in range(16)]
        V_s = [[0]*rank_school for _ in range(20)]
        W_s = [[0]*rank_school for _ in range(20)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U_s[i*4+j][col] = 1
                    V_s[j*5+k][col] = 1
                    W_s[k*4+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
