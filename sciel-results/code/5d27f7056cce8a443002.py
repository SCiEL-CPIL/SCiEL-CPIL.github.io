import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
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
        """Strassen for 2x2x2 block."""
        i0 = i_off; i1 = i_off+1
        j0 = j_off; j1 = j_off+1
        k0 = k_off; k1 = k_off+1
        
        # Term 1: M1 = (A11+A22)(B11+B22)
        u1 = [i0*m+j0, i1*m+j1]
        v1 = [j0*p+k0, j1*p+k1]
        w1 = [k0*n+i0, k1*n+i1]
        add_rank1_term(u1, [1, 1], v1, [1, 1], w1, [1, 1])
        
        # Term 2: M2 = (A21+A22)B11
        u2 = [i1*m+j0, i1*m+j1]
        v2 = [j0*p+k0]
        w2 = [k0*n+i1, k1*n+i1]
        add_rank1_term(u2, [1, 1], v2, [1], w2, [1, -1])
        
        # Term 3: M3 = A11(B21-B11)
        u3 = [i0*m+j0]
        v3 = [j0*p+k1, j1*p+k1]
        w3 = [k1*n+i0, k1*n+i1]
        add_rank1_term(u3, [1], v3, [1, -1], w3, [1, 1])
        
        # Term 4: M4 = A22(B12-B11)
        u4 = [i1*m+j1]
        v4 = [j1*p+k0, j0*p+k0]
        w4 = [k0*n+i0, k0*n+i1]
        add_rank1_term(u4, [1], v4, [1, -1], w4, [1, 1])
        
        # Term 5: M5 = (A11+A12)B22
        u5 = [i0*m+j0, i0*m+j1]
        v5 = [j1*p+k1]
        w5 = [k0*n+i0, k1*n+i0]
        add_rank1_term(u5, [1, 1], v5, [1], w5, [-1, 1])
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        u6 = [i1*m+j0, i0*m+j0]
        v6 = [j0*p+k0, j0*p+k1]
        w6 = [k1*n+i1]
        add_rank1_term(u6, [1, -1], v6, [1, 1], w6, [1])
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        u7 = [i0*m+j1, i1*m+j1]
        v7 = [j1*p+k0, j1*p+k1]
        w7 = [k0*n+i0]
        add_rank1_term(u7, [1, -1], v7, [1, 1], w7, [1])

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        """
        Recursively decompose the tensor M_{n_l, m_l, p_l} using known optimal ranks.
        Base cases use schoolbook or specific optimized decompositions.
        """
        
        # Base case: 1x1x1
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off; j = j_off; k = k_off
            add_rank1_term([i*m+j], [1.0], [j*p+k], [1.0], [k*n+i], [1.0])
            return

        # Degenerate cases: Schoolbook is optimal
        if n_l == 1:
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])
            return

        if m_l == 1:
            for i in range(n_l):
                for k in range(p_l):
                    ii = i_off + i
                    j = j_off
                    kk = k_off + k
                    add_rank1_term([ii*m+j], [1.0], [j*p+kk], [1.0], [kk*n+ii], [1.0])
            return

        if p_l == 1:
            for i in range(n_l):
                for j in range(m_l):
                    ii = i_off + i
                    jj = j_off + j
                    k = k_off
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+k], [1.0], [k*n+ii], [1.0])
            return

        # Strassen 2x2x2 (Rank 7)
        if n_l == 2 and m_l == 2 and p_l == 2:
            strassen_2x2x2(i_off, j_off, k_off)
            return

        # 2x2x3 (Rank 11): Split into 2x2x2 (7) + 2x2x1 (4)
        if n_l == 2 and m_l == 2 and p_l == 3:
            strassen_2x2x2(i_off, j_off, k_off)
            k1 = k_off + 2
            for i in range(2):
                for j in range(2):
                    ii = i_off + i
                    jj = j_off + j
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+k1], [1.0], [k1*n+ii], [1.0])
            return

        # 2x2x4 (Rank 14): Split into 2x2x2 (7) + 2x2x2 (7)
        if n_l == 2 and m_l == 2 and p_l == 4:
            strassen_2x2x2(i_off, j_off, k_off)
            strassen_2x2x2(i_off, j_off, k_off+2)
            return

        # 2x2x7 (Rank 25): Split into 2x2x3 (11) + 2x2x4 (14)
        if n_l == 2 and m_l == 2 and p_l == 7:
            gen_terms(2, 2, 3, i_off, j_off, k_off)
            gen_terms(2, 2, 4, i_off, j_off, k_off+3)
            return

        # 2x4x7 (Rank 50): Split into 2x2x7 (25) + 2x2x7 (25)
        if n_l == 2 and m_l == 4 and p_l == 7:
            gen_terms(2, 2, 7, i_off, j_off, k_off)
            gen_terms(2, 2, 7, i_off, j_off+2, k_off)
            return

        # 3x4x7 (Rank 78): Split into 2x4x7 (50) + 1x4x7 (28)
        if n_l == 3 and m_l == 4 and p_l == 7:
            gen_terms(2, 4, 7, i_off, j_off, k_off)
            gen_terms(1, 4, 7, i_off+2, j_off, k_off)
            return

        # Fallback: Schoolbook
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])

    gen_terms(n, m, p, 0, 0, 0)
    
    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
