import time

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_offset, j_offset_a, j_offset_b, k_offset):
        nonlocal col
        # A indices: i in {0,1}, j in {j_offset_a, j_offset_a+1}
        # B indices: j in {j_offset_b, j_offset_b+1}, k in {k_offset, k_offset+1}
        # C indices: k in {k_offset, k_offset+1}, i in {0,1}
        
        # M1: (a+d)(p+s)
        # A: (i=0, j=j0), (i=1, j=j1)
        # B: (j=j0, k=k0), (j=j1, k=k1)
        # C: (k=k0, i=0), (k=k1, i=1)
        
        u_a = i_offset * m + j_offset_a
        u_d = (i_offset + 1) * m + (j_offset_a + 1)
        
        v_p = j_offset_b * p + k_offset
        v_s = (j_offset_b + 1) * p + (k_offset + 1)
        
        w_c11 = k_offset * n + i_offset
        w_c22 = (k_offset + 1) * n + (i_offset + 1)
        
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_p][col] += 1
        V[v_s][col] += 1
        W[w_c11][col] += 1
        W[w_c22][col] += 1
        col += 1
        
        # M2: (c+d)p
        u_c = (i_offset + 1) * m + j_offset_a
        u_d2 = (i_offset + 1) * m + (j_offset_a + 1)
        
        v_p2 = j_offset_b * p + k_offset
        
        w_c21 = k_offset * n + (i_offset + 1)
        w_c222 = (k_offset + 1) * n + (i_offset + 1)
        
        U[u_c][col] += 1
        U[u_d2][col] += 1
        V[v_p2][col] += 1
        W[w_c21][col] += 1
        W[w_c222][col] += 1
        col += 1
        
        # M3: a(q-s)
        u_a2 = i_offset * m + j_offset_a
        u_b = i_offset * m + (j_offset_a + 1)
        
        v_q = j_offset_b * p + (k_offset + 1)
        v_s2 = (j_offset_b + 1) * p + (k_offset + 1)
        
        w_c12 = (k_offset + 1) * n + i_offset
        w_c223 = (k_offset + 1) * n + (i_offset + 1)
        
        U[u_a2][col] += 1
        V[v_q][col] += 1
        V[v_s2][col] -= 1
        W[w_c12][col] += 1
        W[w_c223][col] += 1
        col += 1
        
        # M4: d(r-p)
        u_d3 = (i_offset + 1) * m + (j_offset_a + 1)
        
        v_r = (j_offset_b + 1) * p + k_offset
        v_p3 = j_offset_b * p + k_offset
        
        w_c112 = k_offset * n + i_offset
        w_c212 = k_offset * n + (i_offset + 1)
        
        U[u_d3][col] += 1
        V[v_r][col] += 1
        V[v_p3][col] -= 1
        W[w_c112][col] += 1
        W[w_c212][col] += 1
        col += 1
        
        # M5: (a+b)s
        u_a3 = i_offset * m + j_offset_a
        u_b2 = i_offset * m + (j_offset_a + 1)
        
        v_s3 = (j_offset_b + 1) * p + (k_offset + 1)
        
        w_c113 = k_offset * n + i_offset
        w_c122 = (k_offset + 1) * n + i_offset
        
        U[u_a3][col] += 1
        U[u_b2][col] += 1
        V[v_s3][col] += 1
        W[w_c113][col] += 1
        W[w_c122][col] += 1
        col += 1
        
        # M6: (c-a)(p+q)
        u_c2 = (i_offset + 1) * m + j_offset_a
        u_a4 = i_offset * m + j_offset_a
        
        v_p4 = j_offset_b * p + k_offset
        v_q2 = j_offset_b * p + (k_offset + 1)
        
        w_c224 = (k_offset + 1) * n + (i_offset + 1)
        
        U[u_c2][col] += 1
        U[u_a4][col] -= 1
        V[v_p4][col] += 1
        V[v_q2][col] += 1
        W[w_c224][col] += 1
        col += 1
        
        # M7: (b-d)(r+s)
        u_b3 = i_offset * m + (j_offset_a + 1)
        u_d4 = (i_offset + 1) * m + (j_offset_a + 1)
        
        v_r2 = (j_offset_b + 1) * p + k_offset
        v_s4 = (j_offset_b + 1) * p + (k_offset + 1)
        
        w_c114 = k_offset * n + i_offset
        
        U[u_b3][col] += 1
        U[u_d4][col] -= 1
        V[v_r2][col] += 1
        V[v_s4][col] += 1
        W[w_c114][col] += 1
        col += 1

    def add_schoolbook_block_2x2x1(i_offset, j_offset_a, j_offset_b, k_offset):
        nonlocal col
        # A: i in {0,1}, j in {j0, j1}
        # B: j in {j0, j1}, k in {k0}
        # C: k in {k0}, i in {0,1}
        
        u_a = i_offset * m + j_offset_a
        u_b = i_offset * m + (j_offset_a + 1)
        u_c = (i_offset + 1) * m + j_offset_a
        u_d = (i_offset + 1) * m + (j_offset_a + 1)
        
        v_p = j_offset_b * p + k_offset
        v_r = (j_offset_b + 1) * p + k_offset
        
        w_c1 = k_offset * n + i_offset
        w_c2 = k_offset * n + (i_offset + 1)
        
        # M1: a p
        U[u_a][col] += 1
        V[v_p][col] += 1
        W[w_c1][col] += 1
        col += 1
        
        # M2: b r
        U[u_b][col] += 1
        V[v_r][col] += 1
        W[w_c1][col] += 1
        col += 1
        
        # M3: c p
        U[u_c][col] += 1
        V[v_p][col] += 1
        W[w_c2][col] += 1
        col += 1
        
        # M4: d r
        U[u_d][col] += 1
        V[v_r][col] += 1
        W[w_c2][col] += 1
        col += 1

    # Block 1: i=0, jA=0, jB=0, k=0
    add_strassen_block(0, 0, 0, 0)
    # Block 2: i=0, jA=0, jB=0, k=2
    add_strassen_block(0, 0, 0, 2)
    # Block 3: i=0, jA=0, jB=0, k=4
    add_schoolbook_block_2x2x1(0, 0, 0, 4)
    
    # Block 4: i=0, jA=2, jB=2, k=0
    add_strassen_block(0, 2, 2, 0)
    # Block 5: i=0, jA=2, jB=2, k=2
    add_strassen_block(0, 2, 2, 2)
    # Block 6: i=0, jA=2, jB=2, k=4
    add_schoolbook_block_2x2x1(0, 2, 2, 4)
    
    return {"U": U, "V": V, "W": W}
