import time

DIMENSIONS = [3, 5, 6]

def get_strassen_222():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[2][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[2][6] = 1
    W[0][6] = 1; W[1][6] = 1
    return U, V, W

def get_simple_122():
    U = [[0]*2 for _ in range(2)]
    V = [[0]*2 for _ in range(4)]
    W = [[0]*2 for _ in range(2)]
    U[0][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1
    U[1][1] = 1
    V[2][1] = 1; V[3][1] = 1
    W[1][1] = 1
    return U, V, W

def get_simple_212():
    U = [[0]*2 for _ in range(2)]
    V = [[0]*2 for _ in range(2)]
    W = [[0]*2 for _ in range(4)]
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1
    W[0][0] = 1; W[2][0] = 1
    U[0][1] = 1; U[1][1] = 1
    V[1][1] = 1
    W[1][1] = 1; W[3][1] = 1
    return U, V, W

def get_simple_112():
    U = [[0]*1 for _ in range(1)]
    V = [[0]*1 for _ in range(2)]
    W = [[0]*1 for _ in range(2)]
    U[0][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[1][0] = 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    N, M, P = n, m, p
    
    I_starts = [0, 2]
    I_sizes = [2, 1]
    J_starts = [0, 2, 4]
    J_sizes = [2, 2, 1]
    K_starts = [0, 2, 4]
    K_sizes = [2, 2, 2]
    
    dec_222 = get_strassen_222()
    dec_122 = get_simple_122()
    dec_212 = get_simple_212()
    dec_112 = get_simple_112()
    
    rank = 0
    for b in range(3):
        for a in range(2):
            for c in range(3):
                na = I_sizes[a]
                mb = J_sizes[b]
                pc = K_sizes[c]
                if na == 2 and mb == 2 and pc == 2:
                    rank += 7
                elif na == 1 and mb == 2 and pc == 2:
                    rank += 2
                elif na == 2 and mb == 1 and pc == 2:
                    rank += 2
                elif na == 1 and mb == 1 and pc == 2:
                    rank += 1

    U = [[0]*rank for _ in range(N*M)]
    V = [[0]*rank for _ in range(M*P)]
    W = [[0]*rank for _ in range(P*N)]
    
    col = 0
    for b in range(3):
        js = J_starts[b]
        mb = J_sizes[b]
        for a in range(2):
            is_ = I_starts[a]
            na = I_sizes[a]
            for c in range(3):
                ks = K_starts[c]
                pc = K_sizes[c]
                
                if na == 2 and mb == 2 and pc == 2:
                    u_loc, v_loc, w_loc = dec_222
                    r_loc = 7
                elif na == 1 and mb == 2 and pc == 2:
                    u_loc, v_loc, w_loc = dec_122
                    r_loc = 2
                elif na == 2 and mb == 1 and pc == 2:
                    u_loc, v_loc, w_loc = dec_212
                    r_loc = 2
                elif na == 1 and mb == 1 and pc == 2:
                    u_loc, v_loc, w_loc = dec_112
                    r_loc = 1
                else:
                    continue

                for r in range(r_loc):
                    for i_l in range(na):
                        for j_l in range(mb):
                            u_val = u_loc[i_l * mb + j_l][r]
                            if u_val != 0:
                                i_g = is_ + i_l
                                j_g = js + j_l
                                u_idx = i_g * M + j_g
                                U[u_idx][col] += u_val
                                
                    for j_l in range(mb):
                        for k_l in range(pc):
                            v_val = v_loc[j_l * pc + k_l][r]
                            if v_val != 0:
                                j_g = js + j_l
                                k_g = ks + k_l
                                v_idx = j_g * P + k_g
                                V[v_idx][col] += v_val
                                
                    for k_l in range(pc):
                        for i_l in range(na):
                            w_val = w_loc[k_l * na + i_l][r]
                            if w_val != 0:
                                k_g = ks + k_l
                                i_g = is_ + i_l
                                w_idx = k_g * N + i_g
                                W[w_idx][col] += w_val
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
