import numpy as np
import time

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    # Target: Rank 30 using three blocks of 2x2x3 (rank 10 each)
    # Fallback: Rank 32 using four blocks of 2x2x2 (rank 7 each) and one 2x2x1 (rank 4)
    
    # First, attempt to construct the rank-10 decomposition for 2x2x3
    # A = [[a, b], [c, d]]
    # B = [[x, y, z], [u, v, w]]
    # C = A B
    
    # We define 10 bilinear forms M_i
    # U_block (4x10), V_block (6x10), W_block (6x10)
    # Indices for A: 0:a, 1:b, 2:c, 3:d
    # Indices for B: 0:x, 1:y, 2:z, 3:u, 4:v, 5:w
    # Indices for C: 0:C11, 1:C12, 2:C13, 3:C21, 4:C22, 5:C23
    
    U_block = np.zeros((4, 10))
    V_block = np.zeros((6, 10))
    W_block = np.zeros((6, 10))
    
    # Proposed rank-10 decomposition
    # M0 = a*x
    # M1 = b*u
    # M2 = c*x
    # M3 = d*u
    # M4 = (a+b)*(y+v)
    # M5 = (c+d)*(y+v)
    # M6 = (a+b)*(z+w)
    # M7 = (c+d)*(z+w)
    # M8 = (a-c)*(x+u)
    # M9 = (b-d)*(x+u)
    
    # Fill U_block (coeffs for a,b,c,d)
    U_block[0, 0] = 1  # a in M0
    U_block[1, 1] = 1  # b in M1
    U_block[2, 2] = 1  # c in M2
    U_block[3, 3] = 1  # d in M3
    
    U_block[0, 4] = 1  # a in M4
    U_block[1, 4] = 1  # b in M4
    U_block[2, 5] = 1  # c in M5
    U_block[3, 5] = 1  # d in M5
    
    U_block[0, 6] = 1  # a in M6
    U_block[1, 6] = 1  # b in M6
    U_block[2, 7] = 1  # c in M7
    U_block[3, 7] = 1  # d in M7
    
    U_block[0, 8] = 1  # a in M8
    U_block[2, 8] = -1 # c in M8
    U_block[1, 9] = 1  # b in M9
    U_block[3, 9] = -1 # d in M9
    
    # Fill V_block (coeffs for x,y,z,u,v,w)
    V_block[0, 0] = 1  # x in M0
    V_block[3, 1] = 1  # u in M1
    V_block[0, 2] = 1  # x in M2
    V_block[3, 3] = 1  # u in M3
    
    V_block[1, 4] = 1  # y in M4
    V_block[4, 4] = 1  # v in M4
    V_block[1, 5] = 1  # y in M5
    V_block[4, 5] = 1  # v in M5
    
    V_block[2, 6] = 1  # z in M6
    V_block[5, 6] = 1  # w in M6
    V_block[2, 7] = 1  # z in M7
    V_block[5, 7] = 1  # w in M7
    
    V_block[0, 8] = 1  # x in M8
    V_block[3, 8] = 1  # u in M8
    V_block[0, 9] = 1  # x in M9
    V_block[3, 9] = 1  # u in M9
    
    # Now we need to find W_block such that the tensor equation holds.
    # The tensor T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # We know T for the multiplication.
    # Let's solve for W_block using least squares or direct inversion if possible.
    # Actually, we can just check if the proposed W works, or solve for it.
    
    # Let's construct the full tensor T for 2x2x3
    # T[A_idx, B_idx, C_idx]
    # A_idx: 0(a),1(b),2(c),3(d)
    # B_idx: 0(x),1(y),2(z),3(u),4(v),5(w)
    # C_idx: 0(C11),1(C12),2(C13),3(C21),4(C22),5(C23)
    
    T = np.zeros((4, 6, 6))
    # C11 = ax + bu
    T[0, 0, 0] = 1 # a*x -> C11
    T[1, 3, 0] = 1 # b*u -> C11
    # C12 = ay + bv
    T[0, 1, 1] = 1 # a*y -> C12
    T[1, 4, 1] = 1 # b*v -> C12
    # C13 = az + bw
    T[0, 2, 2] = 1 # a*z -> C13
    T[1, 5, 2] = 1 # b*w -> C13
    # C21 = cx + du
    T[2, 0, 3] = 1 # c*x -> C21
    T[3, 3, 3] = 1 # d*u -> C21
    # C22 = cy + dv
    T[2, 1, 4] = 1 # c*y -> C22
    T[3, 4, 4] = 1 # d*v -> C22
    # C23 = cz + dw
    T[2, 2, 5] = 1 # c*z -> C23
    T[3, 5, 5] = 1 # d*w -> C23
    
    # We have U_block (4x10) and V_block (6x10).
    # We want W_block (6x10) such that for all i,j,k:
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # This is a linear system for W_block.
    # Let's flatten it.
    # For each k (output index), we have a matrix equation:
    # T[:,:,k] = U_block @ diag(V_block.T) @ W_block[:,k]? No.
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Let M_k = T[:,:,k] (4x6 matrix)
    # M_k = U_block * V_block.T * W_block[:,k]? No.
    # (U * V.T) is (4x10) * (10x6) = (4x6). Let P = U_block @ V_block.T.
    # Then T[i,j,k] is not simply P[i,j] * W[k,r] because the sum is over r.
    # Actually, T[i,j,k] = sum_r (U[i,r] V[j,r]) W[k,r].
    # This means for a fixed k, the matrix T[:,:,k] is NOT a simple product.
    # Wait, the decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # So T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
    
    # We can solve for W_block using least squares for each k?
    # No, W_block is shared.
    # We can set up a large linear system.
    # Variables: W[k,r] for k in 0..5, r in 0..9. Total 60 variables.
    # Equations: T[i,j,k] for all i,j,k. Total 4*6*6 = 144 equations.
    # Coefficients for variable W[k', r] in equation (i,j,k):
    # If k' == k, coeff is U[i,r] * V[j,r].
    # If k' != k, coeff is 0.
    
    # So for each k, we have a system:
    # sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k]
    # This is a linear system for the vector w_k = W[k, :] (length 10).
    # The matrix A_k (144x10) has rows corresponding to (i,j) pairs.
    # A_k[(i,j), r] = U[i,r] * V[j,r].
    # RHS_k[(i,j)] = T[i,j,k].
    
    # We can solve for each k independently!
    # This is much simpler.
    
    W_block = np.zeros((6, 10))
    valid = True
    for k in range(6):
        # Construct A_k (36x10) since i in 0..3, j in 0..5 -> 24 pairs? No, 4*6=24.
        # Wait, T is 4x6x6. i=4, j=6. Pairs = 24.
        A_k = np.zeros((24, 10))
        rhs_k = np.zeros(24)
        idx = 0
        for i in range(4):
            for j in range(6):
                for r in range(10):
                    A_k[idx, r] = U_block[i, r] * V_block[j, r]
                rhs_k[idx] = T[i, j, k]
                idx += 1
        
        # Solve A_k w_k = rhs_k
        # Use least squares if not square/consistent
        try:
            w_k, residuals, rank, sv = np.linalg.lstsq(A_k, rhs_k, rcond=None)
            # Check if the solution is exact (or close)
            if np.allclose(A_k @ w_k, rhs_k, atol=1e-9):
                W_block[k, :] = w_k
            else:
                # If not exact, the proposed U,V might not support a rank-10 decomposition
                # Or the decomposition is not valid for this set of U,V.
                valid = False
                break
        except:
            valid = False
            break
    
    if valid:
        # We have a valid rank-10 decomposition for 2x2x3.
        # Now construct for 2x2x9.
        # B is 2x9. Split into 3 blocks of 2x3.
        # B1 = B[:, 0:3], B2 = B[:, 3:6], B3 = B[:, 6:9]
        # C1 = A B1, C2 = A B2, C3 = A B3
        # C = [C1, C2, C3]
        
        # Total rank = 30.
        rank = 30
        U = np.zeros((4, rank))
        V = np.zeros((18, rank)) # m*p = 2*9 = 18
        W = np.zeros((18, rank)) # p*n = 9*2 = 18
        
        # Map for W indices:
        # C is 2x9.
        # W indices are k*n + i.
        # k is 0..8 (columns of B/C), i is 0..1 (rows of A/C).
        # For block 1 (cols 0,1,2):
        # C11 (k=0, i=0) -> W[0]
        # C12 (k=1, i=0) -> W[2]
        # C13 (k=2, i=0) -> W[4]
        # C21 (k=0, i=1) -> W[1]
        # C22 (k=1, i=1) -> W[3]
        # C23 (k=2, i=1) -> W[5]
        
        # For block 2 (cols 3,4,5):
        # C14 (k=3, i=0) -> W[6]
        # ...
        
        # For block 3 (cols 6,7,8):
        # C17 (k=6, i=0) -> W[12]
        # ...
        
        # V indices:
        # B is 2x9.
        # V indices are j*p + k.
        # j is 0..1 (rows of B), k is 0..8 (cols of B).
        # Block 1 (cols 0,1,2):
        # B11 (j=0, k=0) -> V[0]
        # B12 (j=0, k=1) -> V[1]
        # B13 (j=0, k=2) -> V[2]
        # B21 (j=1, k=0) -> V[9]
        # B22 (j=1, k=1) -> V[10]
        # B23 (j=1, k=2) -> V[11]
        
        # Block 2 (cols 3,4,5):
        # B14 (j=0, k=3) -> V[3]
        # ...
        
        # Block 3 (cols 6,7,8):
        # B17 (j=0, k=6) -> V[6]
        # ...
        
        # U is the same for all blocks, just repeated 3 times.
        
        for block_idx in range(3):
            offset = block_idx * 10
            
            # U
            U[:, offset:offset+10] = U_block
            
            # V
            # V_block rows: 0:x, 1:y, 2:z, 3:u, 4:v, 5:w
            # Correspond to B indices for this block.
            # Block 0: cols 0,1,2.
            # V_block[0] (x) -> B[0,0] -> V[0*9+0] = V[0]
            # V_block[1] (y) -> B[0,1] -> V[1]
            # V_block[2] (z) -> B[0,2] -> V[2]
            # V_block[3] (u) -> B[1,0] -> V[9]
            # V_block[4] (v) -> B[1,1] -> V[10]
            # V_block[5] (w) -> B[1,2] -> V[11]
            
            v_indices = [block_idx*3 + 0, block_idx*3 + 1, block_idx*3 + 2, 
                         9 + block_idx*3 + 0, 9 + block_idx*3 + 1, 9 + block_idx*3 + 2]
            
            for r in range(10):
                for b_row in range(6):
                    V[v_indices[b_row], offset+r] = V_block[b_row, r]
            
            # W
            # W_block rows: 0:C11, 1:C12, 2:C13, 3:C21, 4:C22, 5:C23
            # Correspond to C indices for this block.
            # Block 0: cols 0,1,2.
            # W_block[0] (C11) -> C[0,0] -> W[0*2+0] = W[0]
            # W_block[1] (C12) -> C[0,1] -> W[1*2+0] = W[2]
            # W_block[2] (C13) -> C[0,2] -> W[2*2+0] = W[4]
            # W_block[3] (C21) -> C[1,0] -> W[0*2+1] = W[1]
            # W_block[4] (C22) -> C[1,1] -> W[1*2+1] = W[3]
            # W_block[5] (C23) -> C[1,2] -> W[2*2+1] = W[5]
            
            w_indices = [block_idx*3 + 0, (block_idx*3 + 1) + 2*0, (block_idx*3 + 2) + 2*0,
                         block_idx*3 + 0 + 2*1, (block_idx*3 + 1) + 2*1, (block_idx*3 + 2) + 2*1]
            # Simplify:
            # C[0, col] index = col * 2 + 0
            # C[1, col] index = col * 2 + 1
            
            w_indices = [
                (block_idx*3 + 0)*2 + 0, # C11
                (block_idx*3 + 1)*2 + 0, # C12
                (block_idx*3 + 2)*2 + 0, # C13
                (block_idx*3 + 0)*2 + 1, # C21
                (block_idx*3 + 1)*2 + 1, # C22
                (block_idx*3 + 2)*2 + 1  # C23
            ]
            
            for r in range(10):
                for c_row in range(6):
                    W[w_indices[c_row], offset+r] = W_block[c_row, r]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to Rank 32 (Strassen blocks)
        # 2x2x2 (rank 7) x 4 blocks? No, 9 cols.
        # 2x2x2 (rank 7) for first 2, next 2, next 2, next 2.
        # Last col 2x2x1 (rank 4).
        # Total 7*4 + 4 = 32.
        
        rank = 32
        U = np.zeros((4, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        # Strassen 2x2x2
        # A=[[a,b],[c,d]], B=[[x,y],[u,v]], C=[[ax+bu, ay+bv],[cx+du, cy+dv]]
        # 7 products:
        # M1 = (a+d)(x+v)
        # M2 = (c+d)x
        # M3 = a(y-v)
        # M4 = d(u-x)
        # M5 = (a+b)v
        # M6 = (c+d)(y-v)
        # M7 = (a-b)(x+u)
        
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7)) # B is 2x2, so 4 entries
        W_s = np.zeros((4, 7)) # C is 2x2, so 4 entries
        
        # Fill U_s, V_s, W_s for Strassen
        # M1
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        # M2
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        # M3
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        # M4
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        # M5
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        # M6
        U_s[2, 5] = 1; U_s[3, 5] = 1
        V_s[1, 5] = 1; V_s[3, 5] = -1
        # M7
        U_s[0, 6] = 1; U_s[1, 6] = -1
        V_s[0, 6] = 1; V_s[2, 6] = 1
        
        # W_s (reconstruction)
        # C11 = M1 - M5 + M7 + M4 ? No.
        # C11 = ax+bu
        # C12 = ay+bv
        # C21 = cx+du
        # C22 = cy+dv
        
        # M1 = ax+av+dx+dv
        # M2 = cx+dx
        # M3 = ay-av
        # M4 = du-dx
        # M5 = av+bv
        # M6 = cy-dv
        # M7 = ax+au-bx-bu
        
        # C11 = ax+bu
        # C12 = ay+bv
        # C21 = cx+du
        # C22 = cy+dv
        
        # Let's solve for W_s using lstsq as well, to be safe.
        T_s = np.zeros((4, 4, 4))
        T_s[0,0,0] = 1; T_s[1,2,0] = 1 # C11 = ax+bu
        T_s[0,1,1] = 1; T_s[1,3,1] = 1 # C12 = ay+bv
        T_s[2,0,2] = 1; T_s[3,2,2] = 1 # C21 = cx+du
        T_s[2,1,3] = 1; T_s[3,3,3] = 1 # C22 = cy+dv
        
        W_s = np.zeros((4, 7))
        for k in range(4):
            A_k = np.zeros((16, 7))
            rhs_k = np.zeros(16)
            idx = 0
            for i in range(4):
                for j in range(4):
                    for r in range(7):
                        A_k[idx, r] = U_s[i, r] * V_s[j, r]
                    rhs_k[idx] = T_s[i, j, k]
                    idx += 1
            w_k, _, _, _ = np.linalg.lstsq(A_k, rhs_k, rcond=None)
            W_s[k, :] = w_k
        
        # 2x2x1 (rank 4)
        # A=[[a,b],[c,d]], B=[[x],[u]], C=[[ax+bu],[cx+du]]
        # M1=ax, M2=bu, M3=cx, M4=du
        U_1 = np.zeros((4, 4))
        V_1 = np.zeros((2, 4))
        W_1 = np.zeros((2, 4))
        U_1[0,0]=1; U_1[1,1]=1; U_1[2,2]=1; U_1[3,3]=1
        V_1[0,0]=1; V_1[1,1]=1; V_1[0,2]=1; V_1[1,3]=1
        W_1[0,0]=1; W_1[0,1]=1; W_1[1,2]=1; W_1[1,3]=1
        
        # Construct full 2x2x9
        # 4 blocks of 2x2x2, 1 block of 2x2x1
        col_offset = 0
        rank_offset = 0
        for _ in range(4):
            U[:, rank_offset:rank_offset+7] = U_s
            # V indices for 2x2 block at col_offset
            # B[0, col_offset] -> V[col_offset]
            # B[0, col_offset+1] -> V[col_offset+1]
            # B[1, col_offset] -> V[9+col_offset]
            # B[1, col_offset+1] -> V[9+col_offset+1]
            v_indices = [col_offset, col_offset+1, 9+col_offset, 9+col_offset+1]
            for r in range(7):
                for b_row in range(4):
                    V[v_indices[b_row], rank_offset+r] = V_s[b_row, r]
            
            # W indices for 2x2 block at col_offset
            # C[0, col_offset] -> W[col_offset*2]
            # C[0, col_offset+1] -> W[(col_offset+1)*2]
            # C[1, col_offset] -> W[col_offset*2+1]
            # C[1, col_offset+1] -> W[(col_offset+1)*2+1]
            w_indices = [col_offset*2, (col_offset+1)*2, col_offset*2+1, (col_offset+1)*2+1]
            for r in range(7):
                for c_row in range(4):
                    W[w_indices[c_row], rank_offset+r] = W_s[c_row, r]
            
            col_offset += 2
            rank_offset += 7
        
        # Last block 2x2x1 at col_offset=8
        U[:, rank_offset:rank_offset+4] = U_1
        v_indices = [8, 17] # B[0,8], B[1,8]
        for r in range(4):
            for b_row in range(2):
                V[v_indices[b_row], rank_offset+r] = V_1[b_row, r]
        
        w_indices = [8*2, 8*2+1] # C[0,8], C[1,8]
        for r in range(4):
            for c_row in range(2):
                W[w_indices[c_row], rank_offset+r] = W_1[c_row, r]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

