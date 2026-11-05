import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    # 2 (m-split) * 4 (p-split) * (4 (1x2x2 schoolbook) + 7 (2x2x2 Strassen))
    # 8 * 11 = 88
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-d)(f+h)
    # p7 = (a-c)(e+g)
    
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[c00, c01], [c10, c11]]
    # c00 = p1 + p4 - p5 + p6
    # c01 = p2 + p4
    # c10 = p3 + p5
    # c11 = p1 - p2 + p3 + p7
    
    # U[i, j, t], V[j, k, t], W[k, i, t]
    # Local indices i,j,k in {0,1}
    
    strassen_terms = [
        # t, U_coeffs (i,j), V_coeffs (j,k), W_coeffs (k,i)
        # p1: (a+d)(e+h)
        {
            'U': [(0,0,1), (1,1,1)], # a+d
            'V': [(0,0,1), (0,1,1), (1,0,1), (1,1,1)], # e+h
            'W': [(0,0,1), (0,1,1), (1,0,1), (1,1,-1)] # c00:+1, c01:0, c10:0, c11:+1? 
            # Wait, W maps to C indices.
            # C00 gets +1, C11 gets +1.
            # W[k, i, t].
            # C00 is i=0, k=0. W[0,0]=1.
            # C11 is i=1, k=1. W[1,1]=1.
            # C01 is i=0, k=1. W[1,0]=0.
            # C10 is i=1, k=0. W[0,1]=0.
        },
        # p2: (c+d)e
        {
            'U': [(1,0,1), (1,1,1)], # c+d
            'V': [(0,0,1)], # e
            'W': [(0,1,1), (1,1,-1)] # C01:+1, C11:-1
        },
        # p3: a(f-h)
        {
            'U': [(0,0,1)], # a
            'V': [(0,1,1), (1,1,-1)], # f-h
            'W': [(1,0,1), (1,1,1)] # C10:+1, C11:+1
        },
        # p4: d(g-e)
        {
            'U': [(1,1,1)], # d
            'V': [(1,0,1), (0,0,-1)], # g-e
            'W': [(0,0,1), (0,1,1)] # C00:+1, C01:+1
        },
        # p5: (a+b)h
        {
            'U': [(0,0,1), (0,1,1)], # a+b
            'V': [(1,1,1)], # h
            'W': [(0,0,-1), (1,0,1)] # C00:-1, C10:+1
        },
        # p6: (c-d)(f+h)
        {
            'U': [(1,0,1), (1,1,-1)], # c-d
            'V': [(0,1,1), (1,1,1)], # f+h
            'W': [(0,0,1)] # C00:+1
        },
        # p7: (a-c)(e+g)
        {
            'U': [(0,0,1), (1,0,-1)], # a-c
            'V': [(0,0,1), (1,0,1)], # e+g
            'W': [(1,1,1)] # C11:+1
        }
    ]
    
    # Schoolbook for 1x2x2
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    # Local i=0, j in {0,1}, k in {0,1}
    schoolbook_1x2x2 = [
        {
            'U': [(0,0,1)], # A0
            'V': [(0,0,1)], # B00
            'W': [(0,0,1)]  # C0
        },
        {
            'U': [(0,1,1)], # A1
            'V': [(1,0,1)], # B10
            'W': [(0,0,1)]  # C0
        },
        {
            'U': [(0,0,1)], # A0
            'V': [(0,1,1)], # B01
            'W': [(1,0,1)]  # C1
        },
        {
            'U': [(0,1,1)], # A1
            'V': [(1,1,1)], # B11
            'W': [(1,0,1)]  # C1
        }
    ]
    
    for m_block in range(2):
        j_offset = m_block * 2
        for p_block in range(4):
            k_offset = p_block * 2
            
            # Add 1x2x2 schoolbook for i=0
            for term in schoolbook_1x2x2:
                # Map local to global
                # Local i=0 -> Global i=0
                # Local j -> Global j = j_offset + local_j
                # Local k -> Global k = k_offset + local_k
                
                # U
                for (li, lj, val) in term['U']:
                    gi = 0
                    gj = j_offset + lj
                    U[gi * m + gj][col] += val
                # V
                for (lj, lk, val) in term['V']:
                    gj = j_offset + lj
                    gk = k_offset + lk
                    V[gj * p + gk][col] += val
                # W
                for (lk, li, val) in term['W']:
                    gk = k_offset + lk
                    gi = 0
                    W[gk * n + gi][col] += val
                col += 1
                
            # Add 2x2x2 Strassen for i=1,2
            for term in strassen_terms:
                # Map local to global
                # Local i in {0,1} -> Global i = 1 + local_i
                # Local j -> Global j = j_offset + local_j
                # Local k -> Global k = k_offset + local_k
                
                # U
                for (li, lj, val) in term['U']:
                    gi = 1 + li
                    gj = j_offset + lj
                    U[gi * m + gj][col] += val
                # V
                for (lj, lk, val) in term['V']:
                    gj = j_offset + lj
                    gk = k_offset + lk
                    V[gj * p + gk][col] += val
                # W
                for (lk, li, val) in term['W']:
                    gk = k_offset + lk
                    gi = 1 + li
                    W[gk * n + gi][col] += val
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
