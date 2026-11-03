def _construct_3x4x8():
    n, m, p = 3, 4, 8
    total_rank = 0
    blocks = []
    
    # i=0,1 part: 2x4x8
    # 4 blocks of 2x2x4
    for jp in range(2):
        for kp in range(2):
            blocks.append(('2x2x4', jp, kp, 14))
            total_rank += 14
            
    # i=2 part: 1x4x8
    # 8 blocks of 1x2x2
    for jp in range(2):
        for kp in range(4):
            blocks.append(('1x2x2', jp, kp, 4))
            total_rank += 4
            
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for btype, jp, kp, r in blocks:
        j0 = 2 * jp
        j1 = j0 + 1
        
        if btype == '2x2x4':
            k0 = 4 * kp
            # Two 2x2x2 blocks: k0, k0+1 and k0+2, k0+3
            
            # Block 1: k0, k0+1
            U_sub, V_sub, W_sub = _strassen_2x2x2()
            for t in range(7):
                # U indices: i=0,1; j=j0,j1
                # u_vec is length 4: [A_00, A_01, A_10, A_11] corresponding to (i=0,j=j0), (i=0,j=j1), (i=1,j=j0), (i=1,j=j1)
                u_indices = [0*m + j0, 0*m + j1, 1*m + j0, 1*m + j1]
                for idx, val in enumerate(U_sub[t]):
                    if val != 0:
                        U[u_indices[idx]][col_offset] += val
                        
                # V indices: j=j0,j1; k=k0,k0+1
                # v_vec is length 4: [B_00, B_01, B_10, B_11] corresponding to (j=j0,k=k0), (j=j0,k=k0+1), (j=j1,k=k0), (j=j1,k=k0+1)
                v_indices = [j0*p + k0, j0*p + (k0+1), j1*p + k0, j1*p + (k0+1)]
                for idx, val in enumerate(V_sub[t]):
                    if val != 0:
                        V[v_indices[idx]][col_offset] += val
                        
                # W indices: k=k0,k0+1; i=0,1
                # w_vec is length 4: [C_00, C_01, C_10, C_11] corresponding to (k=k0,i=0), (k=k0,i=1), (k=k0+1,i=0), (k=k0+1,i=1)
                # Wait, W index is k*n + i.
                # C_00 -> k0*n + 0
                # C_01 -> k0*n + 1
                # C_10 -> (k0+1)*n + 0
                # C_11 -> (k0+1)*n + 1
                w_indices = [k0*n + 0, k0*n + 1, (k0+1)*n + 0, (k0+1)*n + 1]
                for idx, val in enumerate(W_sub[t]):
                    if val != 0:
                        W[w_indices[idx]][col_offset] += val
                        
                col_offset += 1

            # Block 2: k0+2, k0+3
            U_sub, V_sub, W_sub = _strassen_2x2x2()
            for t in range(7):
                u_indices = [0*m + j0, 0*m + j1, 1*m + j0, 1*m + j1]
                for idx, val in enumerate(U_sub[t]):
                    if val != 0:
                        U[u_indices[idx]][col_offset] += val
                        
                v_indices = [j0*p + (k0+2), j0*p + (k0+3), j1*p + (k0+2), j1*p + (k0+3)]
                for idx, val in enumerate(V_sub[t]):
                    if val != 0:
                        V[v_indices[idx]][col_offset] += val
                        
                w_indices = [(k0+2)*n + 0, (k0+2)*n + 1, (k0+3)*n + 0, (k0+3)*n + 1]
                for idx, val in enumerate(W_sub[t]):
                    if val != 0:
                        W[w_indices[idx]][col_offset] += val
                        
                col_offset += 1

        elif btype == '1x2x2':
            k0 = 2 * kp
            u_indices = [2*m + j0, 2*m + j1]
            v_indices = [j0*p + k0, j0*p + (k0+1), j1*p + k0, j1*p + (k0+1)]
            w_indices = [k0*n + 2, (k0+1)*n + 2]
            
            # Term 1: A[2,j0] B[j0,k0] -> C[2,k0]
            U[u_indices[0]][col_offset] += 1
            V[v_indices[0]][col_offset] += 1
            W[w_indices[0]][col_offset] += 1
            col_offset += 1
            
            # Term 2: A[2,j0] B[j0,k1] -> C[2,k1]
            U[u_indices[0]][col_offset] += 1
            V[v_indices[1]][col_offset] += 1
            W[w_indices[1]][col_offset] += 1
            col_offset += 1
            
            # Term 3: A[2,j1] B[j1,k0] -> C[2,k0]
            U[u_indices[1]][col_offset] += 1
            V[v_indices[2]][col_offset] += 1
            W[w_indices[0]][col_offset] += 1
            col_offset += 1
            
            # Term 4: A[2,j1] B[j1,k1] -> C[2,k1]
            U[u_indices[1]][col_offset] += 1
            V[v_indices[3]][col_offset] += 1
            W[w_indices[1]][col_offset] += 1
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
