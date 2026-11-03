import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    n_splits = [1, 2]
    m_splits = [2, 2]
    p_splits = [1, 2, 2, 2]
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_schoolbook(i0, i1, k0, k1, j0, j1):
        for i in range(i0, i1):
            for k in range(k0, k1):
                for j in range(j0, j1):
                    U_row = [0.0] * (n*m)
                    V_row = [0.0] * (m*p)
                    W_row = [0.0] * (p*n)
                    
                    U_idx = i * m + k
                    V_idx = k * p + j
                    W_idx = j * n + i
                    
                    U_row[U_idx] = 1.0
                    V_row[V_idx] = 1.0
                    W_row[W_idx] = 1.0
                    
                    U_list.append(U_row)
                    V_list.append(V_row)
                    W_list.append(W_row)

    def add_strassen(i0, k0, j0):
        # Local 2x2x2 block starting at i0, k0, j0
        # i0, k0, j0 are global indices
        # i in {i0, i0+1}
        # k in {k0, k0+1}
        # j in {j0, j0+1}
        
        def make_row(coeffs, size):
            row = [0.0] * size
            for idx, val in coeffs:
                row[idx] = val
            return row

        # u1 = e_{i0,k0} + e_{i0+1,k0+1}
        u1 = make_row([(i0*m+k0, 1.0), ((i0+1)*m+(k0+1), 1.0)], n*m)
        # v1 = e_{k0,j0} + e_{k0+1,j0+1}
        v1 = make_row([(k0*p+j0, 1.0), ((k0+1)*p+(j0+1), 1.0)], m*p)
        # w1 = e_{j0,i0} + e_{j0,i0+1} + e_{j0+1,i0} - e_{j0+1,i0+1}
        w1 = make_row([(j0*n+i0, 1.0), (j0*n+(i0+1), 1.0), ((j0+1)*n+i0, 1.0), ((j0+1)*n+(i0+1), -1.0)], p*n)
        
        U_list.append(u1); V_list.append(v1); W_list.append(w1)
        
        # u2 = e_{i0,k0+1} + e_{i0+1,k0+1}
        u2 = make_row([(i0*m+(k0+1), 1.0), ((i0+1)*m+(k0+1), 1.0)], n*m)
        # v2 = e_{k0,j0}
        v2 = make_row([(k0*p+j0, 1.0)], m*p)
        # w2 = e_{j0+1,i0} - e_{j0+1,i0+1}
        w2 = make_row([((j0+1)*n+i0, 1.0), ((j0+1)*n+(i0+1), -1.0)], p*n)
        
        U_list.append(u2); V_list.append(v2); W_list.append(w2)
        
        # u3 = e_{i0,k0}
        u3 = make_row([(i0*m+k0, 1.0)], n*m)
        # v3 = e_{k0+1,j0} - e_{k0+1,j0+1}
        v3 = make_row([((k0+1)*p+j0, 1.0), ((k0+1)*p+(j0+1), -1.0)], m*p)
        # w3 = e_{j0,i0+1} + e_{j0+1,i0}
        w3 = make_row([(j0*n+(i0+1), 1.0), ((j0+1)*n+i0, 1.0)], p*n)
        
        U_list.append(u3); V_list.append(v3); W_list.append(w3)
        
        # u4 = e_{i0+1,k0+1}
        u4 = make_row([((i0+1)*m+(k0+1), 1.0)], n*m)
        # v4 = e_{k0,j0+1} - e_{k0,j0}
        v4 = make_row([(k0*p+(j0+1), 1.0), (k0*p+j0, -1.0)], m*p)
        # w4 = e_{j0,i0} + e_{j0+1,i0}
        w4 = make_row([(j0*n+i0, 1.0), ((j0+1)*n+i0, 1.0)], p*n)
        
        U_list.append(u4); V_list.append(v4); W_list.append(w4)
        
        # u5 = e_{i0,k0} + e_{i0,k0+1}
        u5 = make_row([(i0*m+k0, 1.0), (i0*m+(k0+1), 1.0)], n*m)
        # v5 = e_{k0+1,j0+1}
        v5 = make_row([((k0+1)*p+(j0+1), 1.0)], m*p)
        # w5 = -e_{j0,i0} + e_{j0,i0+1}
        w5 = make_row([(j0*n+i0, -1.0), (j0*n+(i0+1), 1.0)], p*n)
        
        U_list.append(u5); V_list.append(v5); W_list.append(w5)
        
        # u6 = e_{i0,k0+1} - e_{i0,k0}
        u6 = make_row([(i0*m+(k0+1), 1.0), (i0*m+k0, -1.0)], n*m)
        # v6 = e_{k0,j0} + e_{k0,j0+1}
        v6 = make_row([(k0*p+j0, 1.0), (k0*p+(j0+1), 1.0)], m*p)
        # w6 = e_{j0+1,i0+1}
        w6 = make_row([((j0+1)*n+(i0+1), 1.0)], p*n)
        
        U_list.append(u6); V_list.append(v6); W_list.append(w6)
        
        # u7 = e_{i0,k0+1} - e_{i0+1,k0+1}
        u7 = make_row([(i0*m+(k0+1), 1.0), ((i0+1)*m+(k0+1), -1.0)], n*m)
        # v7 = e_{k0+1,j0} + e_{k0+1,j0+1}
        v7 = make_row([((k0+1)*p+j0, 1.0), ((k0+1)*p+(j0+1), 1.0)], m*p)
        # w7 = e_{j0,i0}
        w7 = make_row([(j0*n+i0, 1.0)], p*n)
        
        U_list.append(u7); V_list.append(v7); W_list.append(w7)

    n_start = 0
    for n_size in n_splits:
        m_start = 0
        for m_size in m_splits:
            p_start = 0
            for p_size in p_splits:
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(n_start, m_start, p_start)
                else:
                    add_schoolbook(n_start, n_start+n_size, m_start, m_start+m_size, p_start, p_start+p_size)
                p_start += p_size
            m_start += m_size
        n_start += n_size

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U, "V": V, "W": W}
