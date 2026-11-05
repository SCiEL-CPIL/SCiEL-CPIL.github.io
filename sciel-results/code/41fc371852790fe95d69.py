import time

DIMENSIONS = [4, 5, 5]
_DEADLINE = time.time() + 145  # safety margin before 150s limit

def _strassen_2x2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1
    
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[1][4] += 1; W[0][4] -= 1
    
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[2][5] += 1
    W[3][5] += 1
    
    U[1][6] += 1; U[3][6] -= 1
    V[1][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _schoolbook_block(nb, mb, pb):
    rank = nb * mb * pb
    U = [[0]*rank for _ in range(nb * mb)]
    V = [[0]*rank for _ in range(mb * pb)]
    W = [[0]*rank for _ in range(pb * nb)]
    col = 0
    for i in range(nb):
        for j in range(mb):
            for k in range(pb):
                U[i*mb+j][col] = 1
                V[j*pb+k][col] = 1
                W[k*nb+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    n_blocks = [(0, 2), (2, 4)]
    m_blocks = [(0, 2), (2, 4), (4, 5)]
    p_blocks = [(0, 2), (2, 4), (4, 5)]
    
    total_U = n * m
    total_V = m * p
    total_W = p * n
    
    cols_U = []
    cols_V = []
    cols_W = []
    
    for (ns, ne) in n_blocks:
        nsz = ne - ns
        for (ms, me) in m_blocks:
            msz = me - ms
            for (ps, pe) in p_blocks:
                psz = pe - ps
                
                if nsz == 2 and msz == 2 and psz == 2:
                    Ub, Vb, Wb = _strassen_2x2x2()
                else:
                    Ub, Vb, Wb = _schoolbook_block(nsz, msz, psz)
                
                for c in range(len(Ub[0])):
                    u_full = [0] * total_U
                    v_full = [0] * total_V
                    w_full = [0] * total_W
                    
                    for r in range(nsz * msz):
                        il = r // msz
                        jl = r % msz
                        ig = ns + il
                        jg = ms + jl
                        u_full[ig * m + jg] = Ub[r][c]
                    
                    for r in range(msz * psz):
                        jl = r // psz
                        kl = r % psz
                        jg = ms + jl
                        kg = ps + kl
                        v_full[jg * p + kg] = Vb[r][c]
                    
                    for r in range(psz * nsz):
                        kl = r // nsz
                        il = r % nsz
                        kg = ps + kl
                        ig = ns + il
                        w_full[kg * n + ig] = Wb[r][c]
                    
                    cols_U.append(u_full)
                    cols_V.append(v_full)
                    cols_W.append(w_full)
    
    rank = len(cols_U)
    U = [[cols_U[c][r] for c in range(rank)] for r in range(total_U)]
    V = [[cols_V[c][r] for c in range(rank)] for r in range(total_V)]
    W = [[cols_W[c][r] for c in range(rank)] for r in range(total_W)]
    
    return {"U": U, "V": V, "W": W}
