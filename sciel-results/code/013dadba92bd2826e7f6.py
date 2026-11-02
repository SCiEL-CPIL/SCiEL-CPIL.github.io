import numpy as np

def construct():
    n, m, p = 6, 6, 6
    # Rank 63
    rank = 63
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # u_s[i_loc, j_loc], v_s[j_loc, k_loc], w_s[i_loc, k_loc]
    strassen = []
    
    # P1
    u = [[0,0],[0,0]]; u[0][0]=1; u[1][1]=1
    v = [[0,0],[0,0]]; v[0][0]=1; v[1][1]=1
    w = [[0,0],[0,0]]; w[0][0]=1; w[1][1]=1
    strassen.append((u,v,w))
    
    # P2
    u = [[0,0],[0,0]]; u[1][0]=1; u[1][1]=1
    v = [[0,0],[0,0]]; v[0][0]=1
    w = [[0,0],[0,0]]; w[1][0]=1
    strassen.append((u,v,w))
    
    # P3
    u = [[0,0],[0,0]]; u[0][0]=1
    v = [[0,0],[0,0]]; v[0][1]=1; v[1][1]=-1
    w = [[0,0],[0,0]]; w[0][1]=1; w[1][1]=1
    strassen.append((u,v,w))
    
    # P4
    u = [[0,0],[0,0]]; u[1][1]=1
    v = [[0,0],[0,0]]; v[1][0]=1; v[0][0]=-1
    w = [[0,0],[0,0]]; w[0][0]=1; w[1][0]=1
    strassen.append((u,v,w))
    
    # P5
    u = [[0,0],[0,0]]; u[0][0]=1; u[0][1]=1
    v = [[0,0],[0,0]]; v[1][1]=1
    w = [[0,0],[0,0]]; w[0][0]=-1; w[0][1]=1
    strassen.append((u,v,w))
    
    # P6
    u = [[0,0],[0,0]]; u[1][0]=1; u[0][0]=-1
    v = [[0,0],[0,0]]; v[0][0]=1; v[0][1]=1
    w = [[0,0],[0,0]]; w[1][1]=1
    strassen.append((u,v,w))
    
    # P7
    u = [[0,0],[0,0]]; u[0][1]=1; u[1][1]=-1
    v = [[0,0],[0,0]]; v[1][0]=1; v[1][1]=1
    w = [[0,0],[0,0]]; w[0][0]=1
    strassen.append((u,v,w))
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for s in range(7):
                    u, v, w = strassen[s]
                    for i_loc in range(2):
                        for j_loc in range(2):
                            if u[i_loc][j_loc] != 0:
                                i_glob = I * 2 + i_loc
                                j_glob = J * 2 + j_loc
                                U[i_glob * m + j_glob][col] = u[i_loc][j_loc]
                    for j_loc in range(2):
                        for k_loc in range(2):
                            if v[j_loc][k_loc] != 0:
                                j_glob = J * 2 + j_loc
                                k_glob = K * 2 + k_loc
                                V[j_glob * p + k_glob][col] = v[j_loc][k_loc]
                    for i_loc in range(2):
                        for k_loc in range(2):
                            if w[i_loc][k_loc] != 0:
                                i_glob = I * 2 + i_loc
                                k_glob = K * 2 + k_loc
                                W[k_glob * n + i_glob][col] = w[i_loc][k_loc]
                    col += 1
    return {"U": U, "V": V, "W": W}
