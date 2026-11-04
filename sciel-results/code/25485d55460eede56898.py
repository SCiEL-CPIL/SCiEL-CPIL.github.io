import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110  # Safety margin before 120s limit
    
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

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        # Check deadline
        if time.time() > DEADLINE:
            return

        # Base cases and specific optimizations
        
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off; j = j_off; k = k_off
            add_rank1_term([i*m+j], [1.0], [j*p+k], [1.0], [k*n+i], [1.0])
            return

        if n_l == 1:
            for j in range(m_l):
                for k in range(p_l):
                    i = i_off
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([i*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+i], [1.0])
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

        if n_l == 2 and m_l == 2 and p_l == 2:
            # Strassen
            i0 = i_off; i1 = i_off+1
            j0 = j_off; j1 = j_off+1
            k0 = k_off; k1 = k_off+1
            
            u1 = [i0*m+j0, i1*m+j1]
            v1 = [j0*p+k0, j1*p+k1]
            w1 = [k0*n+i0, k1*n+i1]
            add_rank1_term(u1, [1, 1], v1, [1, 1], w1, [1, 1])
            
            u2 = [i1*m+j0, i1*m+j1]
            v2 = [j0*p+k0]
            w2 = [k0*n+i1, k1*n+i1]
            add_rank1_term(u2, [1, 1], v2, [1], w2, [1, -1])
            
            u3 = [i0*m+j0]
            v3 = [j0*p+k1, j1*p+k1]
            w3 = [k1*n+i0, k1*n+i1]
            add_rank1_term(u3, [1], v3, [1, -1], w3, [1, 1])
            
            u4 = [i1*m+j1]
            v4 = [j1*p+k0, j0*p+k0]
            w4 = [k0*n+i0, k0*n+i1]
            add_rank1_term(u4, [1], v4, [1, -1], w4, [1, 1])
            
            u5 = [i0*m+j0, i0*m+j1]
            v5 = [j1*p+k1]
            w5 = [k0*n+i0, k1*n+i0]
            add_rank1_term(u5, [1, 1], v5, [1], w5, [-1, 1])
            
            u6 = [i1*m+j0, i0*m+j0]
            v6 = [j0*p+k0, j0*p+k1]
            w6 = [k1*n+i1]
            add_rank1_term(u6, [1, -1], v6, [1, 1], w6, [1])
            
            u7 = [i0*m+j1, i1*m+j1]
            v7 = [j1*p+k0, j1*p+k1]
            w7 = [k0*n+i0]
            add_rank1_term(u7, [1, -1], v7, [1, 1], w7, [1])
            return

        if n_l == 2 and m_l == 2 and p_l == 1:
            # R(2x2x1) = 4 (no better known)
            i0 = i_off; i1 = i_off+1
            j0 = j_off; j1 = j_off+1
            k0 = k_off
            
            # A = M1*M2, B = N1*N2, C = M1*M2, D = N1*N2
            # C1 = A+B+C+D = (M1+M2)(N1+N2)
            add_rank1_term([i0*m+j0, i0*m+j1, i1*m+j0, i1*m+j1], 
                          [1, 1, 1, 1],
                          [j0*p+k0, j1*p+k0], 
                          [1, 1],
                          [k0*n+i0, k0*n+i1], 
                          [1, 1])
            # C2 = A+B = (M1+M2)N1... wait
            # Actually the 4 terms are:
            # (M1)(N1), (M1)(N2), (M2)(N1), (M2)(N2)
            # Or Strassen-like:
            # t1 = (M1+M2)(N1+N2)
            # t2 = M1*N1
            # t3 = M1*N2
            # t4 = M2*N1
            # C = t1 - t2 - t3 - t4... no that gives 0
            # C = t1 + t2 + t3 + t4 is not right.
            # Standard decomposition for 2x2x1 (matrix product 2x2 * 2x1):
            # C11 = A11*B1 + A12*B2
            # C21 = A21*B1 + A22*B2
            # Rank 4 terms:
            # (A11)*B1, (A12)*B2, (A21)*B1, (A22)*B2
            # Can we do better? No, it's 4.
            # Let's just use 4 rank-1 terms.
            add_rank1_term([i0*m+j0], [1], [j0*p+k0], [1], [k0*n+i0], [1])
            add_rank1_term([i0*m+j1], [1], [j1*p+k0], [1], [k0*n+i0], [1])
            add_rank1_term([i1*m+j0], [1], [j0*p+k0], [1], [k0*n+i1], [1])
            add_rank1_term([i1*m+j1], [1], [j1*p+k0], [1], [k0*n+i1], [1])
            return

        if n_l == 2 and m_l == 2 and p_l == 3:
            # R(2x2x3) is known to be 11.
            # Decompose into 2x2x2 (7) + 2x2x1 (4) = 11.
            gen_terms(2, 2, 2, i_off, j_off, k_off)
            gen_terms(2, 2, 1, i_off, j_off, k_off+2)
            return

        if n_l == 2 and m_l == 2 and p_l == 4:
            # R(2x2x4) = 14? Or 2*7 = 14.
            # Decompose into two 2x2x2 blocks.
            gen_terms(2, 2, 2, i_off, j_off, k_off)
            gen_terms(2, 2, 2, i_off, j_off, k_off+2)
            return

        if n_l == 2 and m_l == 4 and p_l == 7:
            # Decompose 2x4x7.
            # Option A: 2x2x7 + 2x2x7 = 2*(25) = 50.
            # Option B: 2x4x3 + 2x4x4. R(2x4x3)=? R(2x4x4)=?
            # R(2x4x4) = 16? 2*2*4 = 16 schoolbook. Strassen on m?
            # Let's stick to 2x2x7 decomposition.
            gen_terms(2, 2, 7, i_off, j_off, k_off)
            gen_terms(2, 2, 7, i_off, j_off+2, k_off)
            return

        if n_l == 2 and m_l == 2 and p_l == 7:
            # R(2x2x7) = 25?
            # 3 * R(2x2x2) + R(2x2x1) = 3*7 + 4 = 25.
            gen_terms(2, 2, 2, i_off, j_off, k_off)
            gen_terms(2, 2, 2, i_off, j_off, k_off+2)
            gen_terms(2, 2, 2, i_off, j_off, k_off+4)
            gen_terms(2, 2, 1, i_off, j_off, k_off+6)
            return

        if n_l == 3 and m_l == 4 and p_l == 7:
            # Decompose 3x4x7 into 2x4x7 + 1x4x7.
            # R(2x4x7) = 50.
            # R(1x4x7) = 28.
            # Total 78.
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
