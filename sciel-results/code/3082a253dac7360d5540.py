# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    Uses a block decomposition:
    C = A1*B1 + A2*B2
    where A1 is 2x2 (cols 0,1 of A), B1 is 2x3 (rows 0,1 of B).
    A2 is 2x2 (cols 2,3 of A), B2 is 2x3 (rows 2,3 of B).
    
    Rank of M_{2,2,3} is 11 (7 from Strassen for 2x2x2 + 4 from Schoolbook for 2x2x1).
    Total Rank = 11 + 11 = 22.
    """
    n, m, p = 2, 4, 6
    
    def get_m223_terms():
        terms = []
        # Strassen for M_{2,2,2} (using B cols 0,1)
        # U indices: i*2+l
        # V indices: l*3+k
        # W indices: k*2+i
        
        # M1
        u = [0]*4; v = [0]*6; w = [0]*6
        u[0]=1; u[3]=1
        v[0]=1; v[4]=1
        w[0]=1; w[3]=1
        terms.append((u,v,w))
        
        # M2
        u = [0]*4; v = [0]*6; w = [0]*6
        u[2]=1; u[3]=1
        v[0]=1
        w[1]=1; w[3]=-1
        terms.append((u,v,w))
        
        # M3
        u = [0]*4; v = [0]*6; w = [0]*6
        u[0]=1
        v[1]=1; v[4]=-1
        w[2]=1; w[3]=1
        terms.append((u,v,w))
        
        # M4
        u = [0]*4; v = [0]*6; w = [0]*6
        u[3]=1
        v[3]=1; v[0]=-1
        w[0]=1; w[1]=1
        terms.append((u,v,w))
        
        # M5
        u = [0]*4; v = [0]*6; w = [0]*6
        u[0]=1; u[1]=1
        v[4]=1
        w[0]=-1; w[2]=1
        terms.append((u,v,w))
        
        # M6
        u = [0]*4; v = [0]*6; w = [0]*6
        u[2]=1; u[0]=-1
        v[0]=1; v[1]=1
        w[3]=1
        terms.append((u,v,w))
        
        # M7
        u = [0]*4; v = [0]*6; w = [0]*6
        u[1]=1; u[3]=-1
        v[3]=1; v[4]=1
        w[0]=1
        terms.append((u,v,w))
        
        # Schoolbook for M_{2,2,1} (using B col 2)
        for l in range(2):
            for i in range(2):
                u = [0]*4; v = [0]*6; w = [0]*6
                u[i*2+l] = 1
                v[l*3+2] = 1
                w[2*2+i] = 1
                terms.append((u,v,w))
                
        return terms

    m223_terms = get_m223_terms()
    rank = 22
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    col = 0
    for block in range(2):
        a_col_off = block * 2
        b_row_off = block * 2
        c_col_off = block * 3
        
        for u, v, w in m223_terms:
            for i in range(2):
                for l in range(2):
                    local_idx = i*2 + l
                    global_idx = i*4 + (a_col_off + l)
                    U[global_idx][col] += u[local_idx]
                    
            for l in range(2):
                for k in range(3):
                    local_idx = l*3 + k
                    global_idx = (b_row_off + l)*6 + (c_col_off + k)
                    V[global_idx][col] += v[local_idx]
                    
            for k in range(3):
                for i in range(2):
                    local_idx = k*2 + i
                    global_idx = (c_col_off + k)*2 + i
                    W[global_idx][col] += w[local_idx]
                    
            col += 1
            
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END
