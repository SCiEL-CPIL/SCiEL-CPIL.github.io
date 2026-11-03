import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 4, 7
    
    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        
        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val
            
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_2x2x2(i_off, j_off, k_off):
        i0 = i_off; i1 = i_off+1
        j0 = j_off; j1 = j_off+1
        k0 = k_off; k1 = k_off+1
        
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        if n_l == 1 and m_l == 1 and p_l == 1:
            add_rank1_term([i_off*m+j_off], [1.0], [j_off*p+k_off], [1.0], [k_off*n+i_off], [1.0])
            return
        if n_l == 1:
            for j in range(m_l):
                for k in range(p_l):
                    add_rank1_term([i_off*m+j_off+j], [1.0], [(j_off+j)*p+k_off+k], [1.0], [(k_off+k)*n+i_off], [1.0])
            return
        if m_l == 1:
            for i in range(n_l):
                for k in range(p_l):
                    add_rank1_term([(i_off+i)*m+j_off], [1.0], [j_off*p+k_off+k], [1.0], [(k_off+k)*n+i_off+i], [1.0])
            return
        if p_l == 1:
            for i in range(n_l):
                for j in range(m_l):
                    add_rank1_term([(i_off+i)*m+j_off+j], [1.0], [(j_off+j)*p+k_off], [1.0], [k_off*n+i_off+i], [1.0])
            return
        if n_l == 2 and m_l == 2 and p_l == 2:
            strassen_2x2x2(i_off, j_off, k_off)
            return
        
        # 3x4x7: try 3x(4x7) with 4x7 optimized
        # 4x7 = 2x7 + 2x7, 2x7 = 2x(2x2+2x2+2x1) = 2x(7+7+4) = 36, so 4x7 = 72
        # 3x4x7 = 3x72 = 216? No, that's wrong.
        # Actually for matrix mult: rank(3x4x7) <= rank(3x4x2)+rank(3x4x2)+rank(3x4x2)+rank(3x4x1)
        # = 3*rank(4x2) + 3*rank(4x1) = 3*7 + 3*4 = 21+12 = 33? 
        # But rank(4x2) for 2x2 is 7, for 4x2 it's different.
        
        # Let me just use the current approach which gives 78
        if n_l == 3 and m_l == 4 and p_l == 7:
            gen_terms(2, 4, 7, i_off, j_off, k_off)
            gen_terms(1, 4, 7, i_off+2, j_off, k_off)
            return
        
        if n_l == 2 and m_l == 4 and p_l == 7:
            gen_terms(2, 2, 7, i_off, j_off, k_off)
            gen_terms(2, 2, 7, i_off, j_off+2, k_off)
            return
        
        if n_l == 2 and m_l == 2 and p_l == 7:
            gen_terms(2, 2, 2, i_off, j_off, k_off)
            gen_terms(2, 2, 2, i_off, j_off, k_off+2)
            gen_terms(2, 2, 2, i_off, j_off, k_off+4)
            gen_terms(2, 2, 1, i_off, j_off, k_off+6)
            return
        
        if n_l == 1 and m_l == 4 and p_l == 7:
            for j in range(4):
                for k in range(7):
                    add_rank1_term([i_off*m+j_off+j], [1.0], [(j_off+j)*p+k_off+k], [1.0], [(k_off+k)*n+i_off], [1.0])
            return
        
        # Fallback
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    add_rank1_term([(i_off+i)*m+j_off+j], [1.0], [(j_off+j)*p+k_off+k], [1.0], [(k_off+k)*n+i_off+i], [1.0])

    gen_terms(n, m, p, 0, 0, 0)
    
    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
