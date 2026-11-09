"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen M_2,2,2 decomposition
    # A indices: a (0,0), b (0,1), c (1,0), d (1,1)
    # B indices: e (0,0), f (0,1), g (1,0), h (1,1)
    # C indices: c11 (0,0), c12 (0,1), c21 (1,0), c22 (1,1)
    
    def strassen_block(a_base, b_base, c_base):
        # a_base: (i0, j0)
        # b_base: (j0, k0)
        # c_base: (k0, i0)
        i0, j0 = a_base
        j0b, k0 = b_base
        k0c, i0c = c_base
        assert i0 == i0c
        assert j0 == j0b
        
        a_idx = (i0, j0)
        b_idx = (i0, j0+1)
        c_idx = (i0+1, j0)
        d_idx = (i0+1, j0+1)
        
        e_idx = (j0, k0)
        f_idx = (j0, k0+1)
        g_idx = (j0+1, k0)
        h_idx = (j0+1, k0+1)
        
        c11_idx = (k0, i0)
        c12_idx = (k0+1, i0)
        c21_idx = (k0, i0+1)
        c22_idx = (k0+1, i0+1)
        
        # M1 = (a+d)(e+h)
        add_term(a_idx[0]*4+a_idx[1], 1, e_idx[0]*6+e_idx[1], 1, c11_idx[0]*3+c11_idx[1], 1)
        add_term(d_idx[0]*4+d_idx[1], 1, h_idx[0]*6+h_idx[1], 1, c22_idx[0]*3+c22_idx[1], 1)
        
        # M2 = (c+d)e
        add_term(c_idx[0]*4+c_idx[1], 1, e_idx[0]*6+e_idx[1], 1, c21_idx[0]*3+c21_idx[1], 1)
        add_term(d_idx[0]*4+d_idx[1], 1, e_idx[0]*6+e_idx[1], 0, c21_idx[0]*3+c21_idx[1], 0) # Wait, M2 is one multiplication (c+d)*e
        # My add_term adds one column per call.
        # I need to group the U, V, W values for one multiplication.
        
        # Let's redefine add_strassen to handle the 7 terms correctly.
        
    # Reset and use a more direct approach
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_rank1(u_rows, v_rows, w_rows):
        nonlocal col
        if col >= rank:
            return
        for u_idx, u_val in u_rows:
            U[u_idx][col] += u_val
        for v_idx, v_val in v_rows:
            V[v_idx][col] += v_val
        for w_idx, w_val in w_rows:
            W[w_idx][col] += w_val
        col += 1

    # Process Top Block (i=0,1)
    # 6 blocks of M_2,2,2
    for j_block in range(2):
        for k_block in range(3):
            i0 = 0
            j0 = j_block * 2
            k0 = k_block * 2
            
            # Indices
            a = (i0, j0)
            b = (i0, j0+1)
            c = (i0+1, j0)
            d = (i0+1, j0+1)
            
            e = (j0, k0)
            f = (j0, k0+1)
            g = (j0+1, k0)
            h = (j0+1, k0+1)
            
            c11 = (k0, i0)
            c12 = (k0+1, i0)
            c21 = (k0, i0+1)
            c22 = (k0+1, i0+1)
            
            # M1 = (a+d)(e+h)
            add_rank1([(a[0]*4+a[1], 1), (d[0]*4+d[1], 1)],
                      [(e[0]*6+e[1], 1), (h[0]*6+h[1], 1)],
                      [(c11[0]*3+c11[1], 1), (c22[0]*3+c22[1], 1)])
            
            # M2 = (c+d)e
            add_rank1([(c[0]*4+c[1], 1), (d[0]*4+d[1], 1)],
                      [(e[0]*6+e[1], 1)],
                      [(c21[0]*3+c21[1], 1)])
            
            # M3 = a(f-h)
            add_rank1([(a[0]*4+a[1], 1)],
                      [(f[0]*6+f[1], 1), (h[0]*6+h[1], -1)],
                      [(c12[0]*3+c12[1], 1)])
            
            # M4 = d(g-e)
            add_rank1([(d[0]*4+d[1], 1)],
                      [(g[0]*6+g[1], 1), (e[0]*6+e[1], -1)],
                      [(c11[0]*3+c11[1], 1), (c21[0]*3+c21[1], 1)])
            
            # M5 = (a+b)h
            add_rank1([(a[0]*4+a[1], 1), (b[0]*4+b[1], 1)],
                      [(h[0]*6+h[1], 1)],
                      [(c11[0]*3+c11[1], -1), (c12[0]*3+c12[1], 1)])
            
            # M6 = (c-a)(e+f)
            add_rank1([(c[0]*4+c[1], 1), (a[0]*4+a[1], -1)],
                      [(e[0]*6+e[1], 1), (f[0]*6+f[1], 1)],
                      [(c22[0]*3+c22[1], 1)])
            
            # M7 = (b-d)(g+h)
            add_rank1([(b[0]*4+b[1], 1), (d[0]*4+d[1], -1)],
                      [(g[0]*6+g[1], 1), (h[0]*6+h[1], 1)],
                      [(c11[0]*3+c11[1], 1), (c22[0]*3+c22[1], -1)])

    # Process Bottom Block (i=2)
    # C[2, k] = sum_j A[2, j] B[j, k]
    # Split j into 0,1 and 2,3
    for k in range(6):
        # Term 1: A[2,0]B[0,k] + A[2,1]B[1,k]
        # Rank 2
        # r0: A[2,0]B[0,k]
        add_rank1([(2*4+0, 1)],
                  [(0*6+k, 1)],
                  [(k*3+2, 1)])
        # r1: A[2,1]B[1,k]
        add_rank1([(2*4+1, 1)],
                  [(1*6+k, 1)],
                  [(k*3+2, 1)])
        
        # Term 2: A[2,2]B[2,k] + A[2,3]B[3,k]
        # Rank 2
        # r2: A[2,2]B[2,k]
        add_rank1([(2*4+2, 1)],
                  [(2*6+k, 1)],
                  [(k*3+2, 1)])
        # r3: A[2,3]B[3,k]
        add_rank1([(2*4+3, 1)],
                  [(3*6+k, 1)],
                  [(k*3+2, 1)])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
