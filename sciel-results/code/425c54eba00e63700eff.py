import numpy as np

DIMENSIONS = [4, 4, 7]

def get_strassen_2x2x2():
    # Returns lists of U, V, W vectors for 2x2x2
    # U, V, W are length 4
    us = []
    vs = []
    ws = []
    
    # M1 = (a+d)(e+h)
    u = [1, 0, 0, 1]
    v = [1, 1, 0, 0]
    w = [1, 0, 0, 1] # C00, C01, C10, C11
    us.append(u); vs.append(v); ws.append(w)
    
    # M2 = (c+d)e
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, 0] # C10
    us.append(u); vs.append(v); ws.append(w)
    
    # M3 = a(f-h)
    u = [1, 0, 0, 0]
    v = [0, 1, 0, -1]
    w = [0, 1, 0, 1] # C01, C11
    us.append(u); vs.append(v); ws.append(w)
    
    # M4 = d(g-e)
    u = [0, 0, 0, 1]
    v = [0, 0, 1, -1]
    w = [1, 0, 0, 1] # C00, C11
    us.append(u); vs.append(v); ws.append(w)
    
    # M5 = (a+b)h
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    w = [0, 1, 0, 0] # C01? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M5 contributes to C01.
    w = [0, 1, 0, 0]
    us.append(u); vs.append(v); ws.append(w)
    
    # M6 = (c-a)(e+f)
    u = [0, 0, 1, -1] # c - a? No, (c-a) is -a+c.
    # Let's check signs.
    # C11 has +M6.
    # C00 has -M5.
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # M6:
    u = [0, 0, 1, -1] # c - a
    v = [1, 1, 0, 0] # e + f
    w = [0, 0, 0, 1] # C11
    us.append(u); vs.append(v); ws.append(w)
    
    # M7:
    u = [0, 1, 0, -1] # b - d
    v = [0, 0, 1, 1] # g + h
    w = [1, 0, 0, 0] # C00
    us.append(u); vs.append(v); ws.append(w)
    
    return us, vs, ws

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: 4x4x4 (cols 0-3)
    # Use Kronecker product of 2x2x2 Strassen
    us2, vs2, ws2 = get_strassen_2x2x2()
    
    # 4x4x4 decomposition
    # A is 4x4. B is 4x4. C is 4x4.
    # U size 16, V size 16, W size 16 (local)
    # Global V size 28, W size 28.
    
    for r1 in range(7):
        for r2 in range(7):
            u_local = []
            for i in range(4):
                for j in range(4):
                    # A index i,j
                    # u2 is for 2x2 A.
                    # A[i,j] corresponds to u2[i//2, j//2] * u2[i%2, j%2]?
                    # No, Kronecker product of vectors.
                    # u = u_r1 \otimes u_r2
                    # u_r1 is length 4. u_r2 is length 4.
                    # u is length 16.
                    pass
            
            # Generate Kronecker product
            u_vec = []
            for x in us2[r1]:
                for y in us2[r2]:
                    u_vec.append(x * y)
            
            v_vec_local = []
            for x in vs2[r1]:
                for y in vs2[r2]:
                    v_vec_local.append(x * y)
            
            w_vec_local = []
            for x in ws2[r1]:
                for y in ws2[r2]:
                    w_vec_local.append(x * y)
            
            # Map to global
            # U: 16 entries. i*4+j.
            # Local u_vec is already in row-major order for 4x4?
            # us2[r1] is length 4. Index 0:00, 1:01, 2:10, 3:11.
            # Kronecker: (00,00), (00,01), (00,10), (00,11), (01,00)...
            # This matches i*4+j if we treat i as major?
            # i=0, j=0 -> 0. i=0, j=1 -> 1.
            # u_vec[0] = u2[0]*u2[0] (0,0)
            # u_vec[1] = u2[0]*u2[1] (0,1)
            # u_vec[2] = u2[0]*u2[2] (0,2) -- Wait.
            # us2 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
            # Kronecker of (0,0) and (0,0) is (0,0,0,0).
            # The order of Kronecker product $u \otimes v$ is $u_0 v, u_1 v, \dots$.
            # So $u_{r1} \otimes u_{r2}$.
            # Element $k$ corresponds to $i = k // 2, j = k \% 2$? No.
            # If $u_{r1}$ is index $i_1$, $u_{r2}$ is index $i_2$.
            # The combined index is $i_1 \cdot 2 + i_2$.
            # This corresponds to $i = i_1 \cdot 2 + i_2$ in a 4x4 matrix if we split i into $i_1, i_2$?
            # No, A is 4x4. Split into 2x2 blocks.
            # $A_{i,j}$. $i = i_1 \cdot 2 + i_2$. $j = j_1 \cdot 2 + j_2$.
            # $u$ vector index is $i \cdot 4 + j$.
            # $u_{r1}$ is for $A_{i_1, j_1}$? No, $u_{r1}$ is for the block row/col?
            # In $M_4 = M_2 \otimes M_2$, the "scalar" is the $2 \times 2$ matrix.
            # So $U_{4} = U_{2} \otimes U_{2}$.
            # $U_{2}$ has indices for $2 \times 2$ matrix.
            # So $U_{4}$ index $k$ corresponds to $i = k // 4$? No.
            # Let's just iterate.
            
            U_global = [0] * 16
            for idx, val in enumerate(u_vec):
                # idx corresponds to (i1, j1) in u_r1 and (i2, j2) in u_r2?
                # No, u_vec is flattened Kronecker.
                # u_vec[k] where k = i1*4 + j1? No.
                # u_r1 is length 4. u_r2 is length 4.
                # u_vec is length 16.
                # u_vec[i1*4 + i2] = u_r1[i1] * u_r2[i2].
                # i1 is row/col index in 2x2 block.
                # i2 is row/col index in 2x2 sub-block.
                # So global i = i1*2 + i2.
                # Global j = j1*2 + j2.
                # But u_vec mixes i and j?
                # No, u_r1 is a vector of length 4 representing the 2x2 matrix A1.
                # The indices of u_r1 are 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
                # So u_vec index k corresponds to (i1, j1) from u_r1 and (i2, j2) from u_r2?
                # No, Kronecker product of two vectors $u, v$ is $u_0 v, u_1 v, \dots$.
                # So $u_{vec}[i1 \cdot 4 + i2] = u_{r1}[i1] \cdot u_{r2}[i2]$.
                # Here $i1$ is the index in the first 2x2 matrix (block index).
                # $i2$ is the index in the second 2x2 matrix (sub-block index).
                # So $i1$ corresponds to $(I1, J1)$ and $i2$ corresponds to $(I2, J2)$.
                # Wait, $u_{r1}$ is a single vector. It encodes both row and col.
                # So $i1$ is a flat index 0..3 for the 2x2 matrix.
                # Let $i1 = I1 \cdot 2 + J1$.
                # Let $i2 = I2 \cdot 2 + J2$.
                # Then global $I = I1 \cdot 2 + I2$.
                # Global $J = J1 \cdot 2 + J2$.
                # Global U index = $I \cdot 4 + J$.
                
                I1 = idx // 4
                I2 = idx % 4
                # I1 is flat index in u_r1.
                r1_i = I1 // 2
                r1_j = I1 % 2
                r2_i = I2 // 2
                r2_j = I2 % 2
                
                I = r1_i * 2 + r2_i
                J = r1_j * 2 + r2_j
                
                U_global[I * 4 + J] = val
            
            V_global = [0] * 28
            # V_local is for 4x4 B.
            # Same mapping for j, k.
            # j = J1*2 + J2. k = K1*2 + K2.
            # Global V index = j * 7 + k.
            # But B in Part 1 is 4x4 (cols 0-3).
            # So k is 0..3.
            for idx, val in enumerate(v_vec_local):
                I1 = idx // 4
                I2 = idx % 4
                r1_j = I1 // 2
                r1_k = I1 % 2
                r2_j = I2 // 2
                r2_k = I2 % 2
                
                J = r1_j * 2 + r2_j
                K = r1_k * 2 + r2_k
                
                V_global[J * 7 + K] = val
            
            W_global = [0] * 28
            # W_local is for 4x4 C.
            # k, i.
            for idx, val in enumerate(w_vec_local):
                I1 = idx // 4
                I2 = idx % 4
                r1_k = I1 // 2
                r1_i = I1 % 2
                r2_k = I2 // 2
                r2_i = I2 % 2
                
                K = r1_k * 2 + r2_k
                I = r1_i * 2 + r2_i
                
                W_global[K * 4 + I] = val
            
            U_list.append(U_global)
            V_list.append(V_global)
            W_list.append(W_global)

    # Part 2: 4x4x2 (cols 4-5)
    # 4 independent 2x2x2 Strassen
    for block_a in range(4):
        # block_a: 0: A11, 1: A12, 2: A21, 3: A22
        for block_b in range(2):
            # block_b: 0: B_top, 1: B_bot
            # Only valid pairs:
            # A11 * B_top (block_a=0, block_b=0)
            # A12 * B_bot (block_a=1, block_b=1)
            # A21 * B_top (block_a=2, block_b=0)
            # A22 * B_bot (block_a=3, block_b=1)
            if (block_a == 0 and block_b == 0) or \
               (block_a == 1 and block_b == 1) or \
               (block_a == 2 and block_b == 0) or \
               (block_a == 3 and block_b == 1):
                
                for r in range(7):
                    u_vec = us2[r]
                    v_vec = vs2[r]
                    w_vec = ws2[r]
                    
                    U_global = [0] * 16
                    V_global = [0] * 28
                    W_global = [0] * 28
                    
                    # Map A block
                    if block_a == 0: # A11
                        i_off = 0; j_off = 0
                    elif block_a == 1: # A12
                        i_off = 0; j_off = 2
                    elif block_a == 2: # A21
                        i_off = 2; j_off = 0
                    else: # A22
                        i_off = 2; j_off = 2
                        
                    for idx, val in enumerate(u_vec):
                        I = idx // 2
                        J = idx % 2
                        U_global[(i_off + I) * 4 + (j_off + J)] = val
                        
                    # Map B block
                    # B is 4x2. Cols 4,5.
                    # B_top: j=0,1. B_bot: j=2,3.
                    # k_local 0,1 -> k_global 4,5.
                    if block_b == 0: # B_top
                        j_off_b = 0
                    else: # B_bot
                        j_off_b = 2
                        
                    for idx, val in enumerate(v_vec):
                        J = idx // 2
                        K = idx % 2
                        V_global[(j_off_b + J) * 7 + (4 + K)] = val
                        
                    # Map C block
                    # C is 4x2. Cols 4,5.
                    # C_top: i=0,1. C_bot: i=2,3.
                    # A11, A12 contribute to C_top.
                    # A21, A22 contribute to C_bot.
                    if block_a in [0, 1]:
                        i_off_c = 0
                    else:
                        i_off_c = 2
                        
                    for idx, val in enumerate(w_vec):
                        K = idx // 2
                        I = idx % 2
                        W_global[(4 + K) * 4 + (i_off_c + I)] = val
                        
                    U_list.append(U_global)
                    V_list.append(V_global)
                    W_list.append(W_global)

    # Part 3: 4x4x1 (col 6)
    # Schoolbook
    for i in range(4):
        for j in range(4):
            U_global = [0] * 16
            V_global = [0] * 28
            W_global = [0] * 28
            
            U_global[i * 4 + j] = 1
            V_global[j * 7 + 6] = 1
            W_global[6 * 4 + i] = 1
            
            U_list.append(U_global)
            V_list.append(V_global)
            W_list.append(W_global)

    return {"U": U_list, "V": V_list, "W": W_list}
