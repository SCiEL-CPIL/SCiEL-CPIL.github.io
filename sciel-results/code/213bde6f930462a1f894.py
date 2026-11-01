"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen's 7 factors for 2x2 matrix multiplication
    # Indices for 2x2 matrices: 0:00, 1:01, 2:10, 3:11
    # u: A, v: B, w: C
    # C_ki = sum_j A_ij B_jk
    # My previous derivation used C_ik? 
    # Seed code: W[k*n+i]. So C index is k, i.
    # Strassen formulas derived for C_ik?
    # Let's re-verify Strassen for C_ki = sum A_ij B_jk.
    # This is C^T if C is standard.
    # Standard Strassen: C = AB. C_ij = sum A_ik B_kj.
    # If we want C_ki = sum A_ij B_jk, let A' = A^T, B' = B^T.
    # Then (A'B')_kj = sum A'_ki B'_jk = sum A_ik B_jk.
    # This is not quite it.
    # Let's stick to the seed convention:
    # U[i*m+j], V[j*p+k], W[k*n+i].
    # This computes C[k,i] = sum_j A[i,j] B[j,k].
    
    # Let's re-derive Strassen for this specific orientation.
    # A is 2x2. B is 2x2. C is 2x2.
    # C_00 = A_00 B_00 + A_01 B_10
    # C_01 = A_00 B_01 + A_01 B_11
    # C_10 = A_10 B_00 + A_11 B_10
    # C_11 = A_10 B_01 + A_11 B_11
    
    # Standard Strassen computes C_std = A_std B_std.
    # C_std[i,j] = sum A_std[i,k] B_std[k,j].
    # If we set A_std = A, B_std = B, then C_std[i,j] = sum A[i,k] B[k,j].
    # Our C is defined by C[k,i] = sum A[i,j] B[j,k].
    # So C[k,i] = C_std[i,k].
    # So C = C_std^T.
    # C = (AB)^T = B^T A^T.
    # So we can compute B^T A^T using Strassen, or just compute AB and transpose W.
    # If we use standard Strassen factors for AB, the W factors will be for C_std.
    # We need W for C = C_std^T.
    # So we need to transpose the W vectors.
    # w_t[k, i] should be w_std_t[i, k].
    # Index in w vector: 2k+i.
    # Index in w_std vector: 2i+k.
    # So w_new[2k+i] = w_std[2i+k].
    
    # Let's use standard Strassen factors for C_std = AB.
    # C_std_00 = M_1 + M_4 - M_5 + M_7
    # C_std_01 = M_3 + M_5
    # C_std_10 = M_2 + M_4
    # C_std_11 = M_1 - M_2 + M_3 + M_6
    
    # u_t, v_t are same.
    # w_t needs to be transposed.
    
    # Standard Strassen w_t (for C_std):
    # w_1: C_00:1, C_11:1 -> [1, 0, 0, 1]
    # w_2: C_10:1, C_11:-1 -> [0, 0, 1, -1]
    # w_3: C_01:1, C_11:1 -> [0, 1, 0, 1]
    # w_4: C_00:1, C_10:1 -> [1, 0, 1, 0]
    # w_5: C_00:-1, C_01:1 -> [-1, 1, 0, 0]
    # w_6: C_11:1 -> [0, 0, 0, 1]
    # w_7: C_00:1 -> [1, 0, 0, 0]
    
    # Transposed w_t (for C = C_std^T):
    # w_t[k, i] = w_std_t[i, k].
    # Index 2k+i.
    # w_1: [1, 0, 0, 1] -> Transpose: [1, 0, 0, 1] (Symmetric)
    # w_2: [0, 0, 1, -1] -> w_std[0,0]=0, w_std[0,1]=0, w_std[1,0]=1, w_std[1,1]=-1.
    #    w_new[0,0]=w_std[0,0]=0
    #    w_new[0,1]=w_std[1,0]=1
    #    w_new[1,0]=w_std[0,1]=0
    #    w_new[1,1]=w_std[1,1]=-1
    #    w_new = [0, 1, 0, -1]
    # w_3: [0, 1, 0, 1] -> w_std[0,0]=0, w_std[0,1]=1, w_std[1,0]=0, w_std[1,1]=1.
    #    w_new[0,0]=0
    #    w_new[0,1]=0
    #    w_new[1,0]=1
    #    w_new[1,1]=1
    #    w_new = [0, 0, 1, 1]
    # w_4: [1, 0, 1, 0] -> w_std[0,0]=1, w_std[0,1]=0, w_std[1,0]=1, w_std[1,1]=0.
    #    w_new[0,0]=1
    #    w_new[0,1]=1
    #    w_new[1,0]=0
    #    w_new[1,1]=0
    #    w_new = [1, 1, 0, 0]
    # w_5: [-1, 1, 0, 0] -> w_std[0,0]=-1, w_std[0,1]=1, w_std[1,0]=0, w_std[1,1]=0.
    #    w_new[0,0]=-1
    #    w_new[0,1]=0
    #    w_new[1,0]=1
    #    w_new[1,1]=0
    #    w_new = [-1, 0, 1, 0]
    # w_6: [0, 0, 0, 1] -> w_std[0,0]=0, w_std[0,1]=0, w_std[1,0]=0, w_std[1,1]=1.
    #    w_new[0,0]=0
    #    w_new[0,1]=0
    #    w_new[1,0]=0
    #    w_new[1,1]=1
    #    w_new = [0, 0, 0, 1]
    # w_7: [1, 0, 0, 0] -> w_std[0,0]=1, others 0.
    #    w_new = [1, 0, 0, 0]
    
    # Let's verify w_2 again.
    # C_std_10 = M_2 + M_4. C_std_11 = -M_2 + ...
    # w_std_2: C_std_10:1, C_std_11:-1.
    # C_10 = C_std_01. C_11 = C_std_11.
    # So w_2 should have 1 at C_01 and -1 at C_11.
    # C_01 index: k=0, i=1 -> 2(0)+1 = 1.
    # C_11 index: k=1, i=1 -> 2(1)+1 = 3.
    # So w_2[1]=1, w_2[3]=-1.
    # w_2 = [0, 1, 0, -1]. Matches.
    
    u = [
        [1, 0, 0, 1], # 1
        [0, 0, 1, 1], # 2
        [1, 0, 0, 0], # 3
        [0, 0, 0, 1], # 4
        [1, 0, 1, 0], # 5
        [-1, 1, 0, 0],# 6
        [0, 0, 1, -1] # 7
    ]
    
    v = [
        [1, 0, 0, 1], # 1
        [1, 0, 0, 0], # 2
        [0, 0, 1, -1],# 3
        [-1, 1, 0, 0],# 4
        [0, 0, 0, 1], # 5
        [1, 0, 1, 0], # 6
        [0, 1, 0, 1]  # 7
    ]
    
    w = [
        [1, 0, 0, 1], # 1
        [0, 1, 0, -1],# 2
        [0, 0, 1, 1], # 3
        [1, 1, 0, 0], # 4
        [-1, 0, 1, 0],# 5
        [0, 0, 0, 1], # 6
        [1, 0, 0, 0]  # 7
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for t in range(7):
        for s in range(7):
            c = t * 7 + s
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            idx_u = 8*i1 + 4*i0 + 2*j1 + j0
                            U[idx_u][c] = u[t][2*i1 + j1] * u[s][2*i0 + j0]
                            
                            idx_v = 8*j1 + 4*j0 + 2*(1-j1) + (1-j0) # Wait, k indices?
                            # V index: j, k.
                            # j = 2j1 + j0.
                            # k = 2k1 + k0.
                            # idx_v = 8*j1 + 4*j0 + 2*k1 + k0.
                            # v_t[j1, k1] is at v[t][2*j1 + k1].
                            # v_s[j0, k0] is at v[s][2*j0 + k0].
                            # So we need to iterate k1, k0 too.
                            pass
                            
    # Let's rewrite the loops properly.
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for t in range(7):
        for s in range(7):
            c = t * 7 + s
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            for k1 in range(2):
                                for k0 in range(2):
                                    # U index: i, j
                                    idx_u = 8*i1 + 4*i0 + 2*j1 + j0
                                    U[idx_u][c] = u[t][2*i1 + j1] * u[s][2*i0 + j0]
                                    
                                    # V index: j, k
                                    idx_v = 8*j1 + 4*j0 + 2*k1 + k0
                                    V[idx_v][c] = v[t][2*j1 + k1] * v[s][2*j0 + k0]
                                    
                                    # W index: k, i
                                    idx_w = 8*k1 + 4*k0 + 2*i1 + i0
                                    W[idx_w][c] = w[t][2*k1 + i1] * w[s][2*k0 + i0]
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
