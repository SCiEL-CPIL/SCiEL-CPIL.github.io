import numpy as np
import time

_DEADLINE = time.time() + 140

def construct():
    n, m, p = 4, 4, 7
    
    def verify(U, V, W, rank):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
        return np.max(np.abs(recon - target)) < 1e-9
    
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def strassen_4x4x4():
        rank = 49
        st = [
            ({(0,0):1,(1,1):1},{(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1},{(0,0):1}),
            ({(0,0):1},{(0,1):1,(1,1):-1}),
            ({(1,1):1},{(1,0):1,(0,0):-1}),
            ({(0,0):1,(0,1):1},{(1,1):1}),
            ({(1,0):1,(0,0):-1},{(0,0):1,(0,1):1}),
            ({(0,1):1,(1,1):-1},{(1,0):1,(1,1):1})
        ]
        cm = [
            {(0,0):1,(1,1):1},
            {(1,0):1,(1,1):-1},
            {(0,1):1,(1,1):1},
            {(0,0):1,(1,0):1},
            {(0,0):-1,(0,1):1},
            {(1,1):1},
            {(0,0):1},
        ]
        
        block_products = [
            ({(0,0):1,(1,1):1},{(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1},{(0,0):1}),
            ({(0,0):1},{(0,1):1,(1,1):-1}),
            ({(1,1):1},{(1,0):1,(0,0):-1}),
            ({(0,0):1,(0,1):1},{(1,1):1}),
            ({(1,0):1,(0,0):-1},{(0,0):1,(0,1):1}),
            ({(0,1):1,(1,1):-1},{(1,0):1,(1,1):1})
        ]
        c_blocks = [
            {(0,0):1, (1,1):1},
            {(1,0):1},
            {(0,1):1, (1,1):-1},
            {(0,0):-1, (1,0):1},
            {(0,1):1, (1,1):1},
            {(0,0):1, (1,1):-1},
            {(0,0):1},
        ]
        
        U = [[0.0]*rank for _ in range(16)]
        V = [[0.0]*rank for _ in range(16)]
        W = [[0.0]*rank for _ in range(16)]
        
        for bp in range(7):
            a_def, b_def = block_products[bp]
            c_def = c_blocks[bp]
            for si in range(7):
                r = bp * 7 + si
                x_sub, y_sub = st[si]
                
                for (li,lj), cx in x_sub.items():
                    for (bi,bj), cb in a_def.items():
                        i = 2*bi + li
                        j = 2*bj + lj
                        U[i*4+j][r] += cx * cb
                
                for (li,lj), cy in y_sub.items():
                    for (bi,bj), cb in b_def.items():
                        i = 2*bi + li
                        j = 2*bj + lj
                        V[i*4+j][r] += cy * cb
                
                for (li,lj), cm_c in cm[si].items():
                    for (br,bc), cb_c in c_def.items():
                        i_final = 2*br + li
                        j_final = 2*bc + lj
                        W[j_final*4 + i_final][r] += cm_c * cb_c
        
        return U, V, W, rank
    
    def build_4x4x3():
        rank = 44
        st2 = [
            ({(0,0):1,(1,1):1},{(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1},{(0,0):1}),
            ({(0,0):1},{(0,1):1,(1,1):-1}),
            ({(1,1):1},{(1,0):1,(0,0):-1}),
            ({(0,0):1,(0,1):1},{(1,1):1}),
            ({(1,0):1,(0,0):-1},{(0,0):1,(0,1):1}),
            ({(0,1):1,(1,1):-1},{(1,0):1,(1,1):1})
        ]
        cm2 = [
            {(0,0):1,(1,1):1},
            {(1,0):1,(1,1):-1},
            {(0,1):1,(1,1):1},
            {(0,0):1,(1,0):1},
            {(0,0):-1,(0,1):1},
            {(1,1):1},
            {(0,0):1},
        ]
        
        U = [[0.0]*rank for _ in range(16)]
        V = [[0.0]*rank for _ in range(12)]
        W = [[0.0]*rank for _ in range(12)]
        
        blocks = [(0,0,0,0), (0,1,1,0), (1,0,0,1), (1,1,1,1)]
        current_rank = 0
        
        for (arb, acb, brb, crb) in blocks:
            for si in range(7):
                r = current_rank
                current_rank += 1
                x_sub, y_sub = st2[si]
                
                for (li,lj), cx in x_sub.items():
                    i = 2*arb + li
                    j = 2*acb + lj
                    U[i*4+j][r] += cx
                
                for (li,lj), cy in y_sub.items():
                    i = 2*brb + li
                    j = lj
                    V[i*3+j][r] += cy
                
                for (li,lj), cm_c in cm2[si].items():
                    i_final = 2*crb + li
                    k_final = lj
                    W[k_final*4 + i_final][r] += cm_c
            
            sb_terms = [((0,0),(0,0),0), ((0,1),(1,0),0), ((1,0),(0,0),1), ((1,1),(1,0),1)]
            for (a_pos, b_pos, c_row) in sb_terms:
                r = current_rank
                current_rank += 1
                ia = 2*arb + a_pos[0]
                ja = 2*acb + a_pos[1]
                U[ia*4+ja][r] = 1.0
                ib = 2*brb + b_pos[0]
                V[ib*3+2][r] = 1.0
                ic = 2*crb + c_row
                W[2*4 + ic][r] = 1.0
        
        return U, V, W, current_rank
    
    U_444, V_444, W_444, r_444 = strassen_4x4x4()
    U_443, V_443, W_443, r_443 = build_4x4x3()
    
    total_rank = r_444 + r_443
    
    U = [[0.0]*total_rank for _ in range(16)]
    V = [[0.0]*total_rank for _ in range(28)]
    W = [[0.0]*total_rank for _ in range(28)]
    
    for i in range(16):
        for r in range(r_444):
            U[i][r] = U_444[i][r]
            V[i][r] = V_444[i][r]
            W[i][r] = W_444[i][r]
    
    for i in range(16):
        for r in range(r_443):
            U[i][r_444+r] = U_443[i][r]
    for i in range(12):
        for r in range(r_443):
            row = i // 3
            col = i % 3
            V[row*7 + 4 + col][r_444+r] = V_443[i][r]
    for i in range(12):
        for r in range(r_443):
            k = i // 4
            row = i % 4
            W[(4+k)*4 + row][r_444+r] = W_443[i][r]
    
    if verify(U, V, W, total_rank):
        return {"U": U, "V": V, "W": W}
    
    U_sb, V_sb, W_sb, _ = schoolbook()
    return {"U": U_sb, "V": V_sb, "W": W_sb}
