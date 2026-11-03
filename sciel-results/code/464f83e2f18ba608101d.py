import time
import sys

def construct():
    n, m, p = 4, 5, 5
    
    # Partitions
    n_parts = [2, 2]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    # Calculate global indices for each part
    n_indices = []
    start = 0
    for size in n_parts:
        n_indices.append(list(range(start, start + size)))
        start += size
        
    m_indices = []
    start = 0
    for size in m_parts:
        m_indices.append(list(range(start, start + size)))
        start += size
        
    p_indices = []
    start = 0
    for size in p_parts:
        p_indices.append(list(range(start, start + size)))
        start += size
        
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i, n_sz in enumerate(n_parts):
        for j, m_sz in enumerate(m_parts):
            for k, p_sz in enumerate(p_parts):
                # Get block specific decompositions
                # Local indices are 0..n_sz-1 etc
                
                if n_sz == 2 and m_sz == 2 and p_sz == 2:
                    # Strassen 2x2x2
                    # m1 = (a+d)(e+h)
                    # m2 = (c+d)e
                    # m3 = a(f-h)
                    # m4 = d(g-e)
                    # m5 = (a+b)h
                    # m6 = (c-a)(e+f)
                    # m7 = (b-d)(g+h)
                    
                    # C00 = m1 + m4 - m5 + m7
                    # C01 = m3 + m5
                    # C10 = m2 + m4
                    # C11 = m1 - m2 + m3 + m6
                    
                    terms = [
                        # (U, V, W) coefficients for local indices
                        # U: (row, col), V: (row, col), W: (row, col)
                        # Note: U corresponds to A, V to B, W to C
                        # A indices: (r, c) in 0..1
                        # B indices: (r, c) in 0..1
                        # C indices: (r, c) in 0..1
                        
                        # m1
                        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1), (1,0,-1)]), # Wait, C10 is -1? No.
                        # Let's trace m1:
                        # C00 += m1, C11 += m1.
                        # My list above had (1,0,-1). That's wrong.
                        # Correct W for m1: C00: 1, C11: 1.
                        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
                        
                        # m2 = (c+d)e
                        # A: (1,0)=c, (1,1)=d. B: (0,0)=e.
                        # C10 += m2.
                        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
                        
                        # m3 = a(f-h)
                        # A: (0,0)=a. B: (0,1)=f, (1,1)=h.
                        # C01 += m3.
                        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
                        
                        # m4 = d(g-e)
                        # A: (1,1)=d. B: (1,0)=g, (0,0)=e.
                        # C00 += m4, C10 += m4.
                        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
                        
                        # m5 = (a+b)h
                        # A: (0,0)=a, (0,1)=b. B: (1,1)=h.
                        # C00 -= m5, C01 += m5.
                        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
                        
                        # m6 = (c-a)(e+f)
                        # A: (1,0)=c, (0,0)=-a. B: (0,0)=e, (0,1)=f.
                        # C11 += m6.
                        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
                        
                        # m7 = (b-d)(g+h)
                        # A: (0,1)=b, (1,1)=-d. B: (1,0)=g, (1,1)=h.
                        # C00 += m7.
                        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
                    ]
                    
                    for u_terms, v_terms, w_terms in terms:
                        U_col = {}
                        V_col = {}
                        W_col = {}
                        
                        for r, c, val in u_terms:
                            gr = n_indices[i][r]
                            gc = m_indices[j][c]
                            key = (gr, gc)
                            U_col[key] = U_col.get(key, 0) + val
                            
                        for r, c, val in v_terms:
                            gr = m_indices[j][r]
                            gc = p_indices[k][c]
                            key = (gr, gc)
                            V_col[key] = V_col.get(key, 0) + val
                            
                        for r, c, val in w_terms:
                            gr = n_indices[i][r]
                            gc = p_indices[k][c]
                            key = (gr, gc)
                            W_col[key] = W_col.get(key, 0) + val
                            
                        U_cols.append(U_col)
                        V_cols.append(V_col)
                        W_cols.append(W_col)
                        
                else:
                    # Schoolbook for other sizes
                    # A is n_sz x m_sz
                    # B is m_sz x p_sz
                    # C is n_sz x p_sz
                    for a_r in range(n_sz):
                        for a_c in range(m_sz):
                            for b_c in range(p_sz):
                                U_col = {}
                                V_col = {}
                                W_col = {}
                                
                                gr_a_r = n_indices[i][a_r]
                                gr_a_c = m_indices[j][a_c]
                                
                                gr_b_r = m_indices[j][a_c]
                                gr_b_c = p_indices[k][b_c]
                                
                                gr_c_r = n_indices[i][a_r]
                                gr_c_c = p_indices[k][b_c]
                                
                                U_col[(gr_a_r, gr_a_c)] = 1
                                V_col[(gr_b_r, gr_b_c)] = 1
                                W_col[(gr_c_r, gr_c_c)] = 1
                                
                                U_cols.append(U_col)
                                V_cols.append(V_col)
                                W_cols.append(W_col)
    
    rank = len(U_cols)
    
    # Convert sparse columns to dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r_idx, col in enumerate(U_cols):
        for (row, col_idx), val in col.items():
            # U is indexed by [i*m + j]
            U[row * m + col_idx][r_idx] = val
            
    for r_idx, col in enumerate(V_cols):
        for (row, col_idx), val in col.items():
            # V is indexed by [j*p + k]
            V[row * p + col_idx][r_idx] = val
            
    for r_idx, col in enumerate(W_cols):
        for (row, col_idx), val in col.items():
            # W is indexed by [k*n + i]
            W[col_idx * n + row][r_idx] = val
            
    return {"U": U, "V": V, "W": W}
