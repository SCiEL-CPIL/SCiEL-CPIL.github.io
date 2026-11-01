import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We will build the lists of coefficients for U, V, W
    # Each "term" is a column in the decomposition.
    # U_col is a vector of length n*m
    # V_col is a vector of length m*p
    # W_col is a vector of length p*n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen terms for 2x2 block
    # Inputs: A (2x2), B (2x2)
    # A indices: i*m+j -> 0,1,2,3 (0:00, 1:01, 2:10, 3:11)
    # B indices: j*p+k. For a block starting at k0, indices are 0*p+k0, 0*p+k0+1, 1*p+k0, 1*p+k0+1
    # W indices: k*n+i. For a block starting at k0, indices are k0*n+0, k0*n+1, (k0+1)*n+0, (k0+1)*n+1
    
    def strassen_block(k_start):
        # k_start is 0, 2, 4, 6
        # B cols are k_start, k_start+1
        # B rows are 0, 1
        # B indices in V:
        # b00: 0*p + k_start
        # b01: 0*p + k_start + 1
        # b10: 1*p + k_start
        # b11: 1*p + k_start + 1
        
        # W indices:
        # c00: k_start*n + 0
        # c01: k_start*n + 1
        # c10: (k_start+1)*n + 0
        # c11: (k_start+1)*n + 1
        
        # A indices in U:
        # a00: 0
        # a01: 1
        # a10: 2
        # a11: 3
        
        # M1 = (a00+a11)(b00+b11)
        u = [0]*4; v = [0]*(m*p); w = [0]*(p*n)
        u[0]=1; u[3]=1
        v[0*p+k_start]=1; v[1*p+k_start+1]=1
        w[k_start*n+0]=1; w[k_start*n+1]=1; w[(k_start+1)*n+0]=1; w[(k_start+1)*n+1]=1
        # C00 += M1, C01 += M1, C10 += M1, C11 += M1
        # Wait, Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So we add contributions to w.
        # For M1: w[C00]+=1, w[C01]+=1, w[C10]+=1, w[C11]+=1
        # Let's just define the 7 terms and their contributions.
        
        terms = []
        
        # M1 = (a00+a11)(b00+b11)
        u1 = [0]*4; u1[0]=1; u1[3]=1
        v1 = [0]*(m*p); v1[0*p+k_start]=1; v1[1*p+k_start+1]=1
        w1 = [0]*(p*n)
        w1[k_start*n+0] += 1
        w1[k_start*n+1] += 1
        w1[(k_start+1)*n+0] += 1
        w1[(k_start+1)*n+1] += 1
        terms.append((u1, v1, w1))
        
        # M2 = (a10+a11)b00
        u2 = [0]*4; u2[2]=1; u2[3]=1
        v2 = [0]*(m*p); v2[0*p+k_start]=1
        w2 = [0]*(p*n)
        w2[(k_start+1)*n+0] += 1
        w2[(k_start+1)*n+1] -= 1
        terms.append((u2, v2, w2))
        
        # M3 = a00(b01-b11)
        u3 = [0]*4; u3[0]=1
        v3 = [0]*(m*p); v3[0*p+k_start+1]=1; v3[1*p+k_start+1]=-1
        w3 = [0]*(p*n)
        w3[k_start*n+1] += 1
        w3[(k_start+1)*n+1] += 1
        terms.append((u3, v3, w3))
        
        # M4 = a11(b10-b00)
        u4 = [0]*4; u4[3]=1
        v4 = [0]*(m*p); v4[1*p+k_start]=1; v4[0*p+k_start]=-1
        w4 = [0]*(p*n)
        w4[k_start*n+0] += 1
        w4[(k_start+1)*n+0] += 1
        terms.append((u4, v4, w4))
        
        # M5 = (a00+a01)b11
        u5 = [0]*4; u5[0]=1; u5[1]=1
        v5 = [0]*(m*p); v5[1*p+k_start+1]=1
        w5 = [0]*(p*n)
        w5[k_start*n+0] -= 1
        w5[k_start*n+1] += 1
        terms.append((u5, v5, w5))
        
        # M6 = (a10-a11)(b01+b11)
        u6 = [0]*4; u6[2]=1; u6[3]=-1
        v6 = [0]*(m*p); v6[0*p+k_start+1]=1; v6[1*p+k_start+1]=1
        w6 = [0]*(p*n)
        w6[(k_start+1)*n+1] += 1
        terms.append((u6, v6, w6))
        
        # M7 = (a01-a10)(b00+b10)
        u7 = [0]*4; u7[1]=1; u7[2]=-1
        v7 = [0]*(m*p); v7[0*p+k_start]=1; v7[1*p+k_start]=1
        w7 = [0]*(p*n)
        w7[k_start*n+0] += 1
        terms.append((u7, v7, w7))
        
        for u, v, w in terms:
            add_term(u, v, w)

    # Trivial terms for 2x1 block
    def trivial_block(k_start):
        # k_start is 8
        # B col is k_start
        # B indices in V:
        # b0: 0*p + k_start
        # b1: 1*p + k_start
        # W indices:
        # c0: k_start*n + 0
        # c1: k_start*n + 1
        
        # C0 = a00*b0 + a01*b1
        # C1 = a10*b0 + a11*b1
        
        # T1 = a00*b0
        u1 = [0]*4; u1[0]=1
        v1 = [0]*(m*p); v1[0*p+k_start]=1
        w1 = [0]*(p*n); w1[k_start*n+0]=1
        add_term(u1, v1, w1)
        
        # T2 = a01*b1
        u2 = [0]*4; u2[1]=1
        v2 = [0]*(m*p); v2[1*p+k_start]=1
        w2 = [0]*(p*n); w2[k_start*n+0]=1
        add_term(u2, v2, w2)
        
        # T3 = a10*b0
        u3 = [0]*4; u3[2]=1
        v3 = [0]*(m*p); v3[0*p+k_start]=1
        w3 = [0]*(p*n); w3[k_start*n+1]=1
        add_term(u3, v3, w3)
        
        # T4 = a11*b1
        u4 = [0]*4; u4[3]=1
        v4 = [0]*(m*p); v4[1*p+k_start]=1
        w4 = [0]*(p*n); w4[k_start*n+1]=1
        add_term(u4, v4, w4)

    # Apply blocks
    strassen_block(0)
    strassen_block(2)
    strassen_block(4)
    strassen_block(6)
    trivial_block(8)
    
    # Construct matrices
    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        for r in range(n*m):
            U[r][c] = U_cols[c][r]
        for r in range(m*p):
            V[r][c] = V_cols[c][r]
        for r in range(p*n):
            W[r][c] = W_cols[c][r]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
