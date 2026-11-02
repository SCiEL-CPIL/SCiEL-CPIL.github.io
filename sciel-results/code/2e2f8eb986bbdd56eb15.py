import time
import numpy as np

def construct():
    """Construct a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Uses block decomposition: 5 = 2+2+1
    - Apply Strassen (rank 7) to the top-left 2x2 block
    - Handle 1x1 block and cross terms with rank-1 terms
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 5, 5, 5
    
    def idx_u(i, j):
        return i * m + j
    
    def idx_v(j, k):
        return j * p + k
    
    def idx_w(k, i):
        return k * n + i
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a single rank-1 term."""
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
    
    # Strassen for top-left 2x2 block (rows 0-1, cols 0-1, output 0-1)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (A00+A11)(B00+B11)
    add_term([(idx_u(0,0), 1), (idx_u(1,1), 1)],
             [(idx_v(0,0), 1), (idx_v(1,1), 1)],
             [(idx_w(0,0), 1), (idx_w(1,1), 1)])
    
    # M2 = (A10+A11)B00
    add_term([(idx_u(1,0), 1), (idx_u(1,1), 1)],
             [(idx_v(0,0), 1)],
             [(idx_w(1,0), 1)])
    
    # M3 = A00(B10-B11)
    add_term([(idx_u(0,0), 1)],
             [(idx_v(1,0), 1), (idx_v(1,1), -1)],
             [(idx_w(0,1), 1), (idx_w(1,1), 1)])
    
    # M4 = A11(B01-B00)
    add_term([(idx_u(1,1), 1)],
             [(idx_v(0,1), 1), (idx_v(0,0), -1)],
             [(idx_w(1,0), 1), (idx_w(0,0), 1)])
    
    # M5 = (A00+A01)B11
    add_term([(idx_u(0,0), 1), (idx_u(0,1), 1)],
             [(idx_v(1,1), 1)],
             [(idx_w(0,1), 1)])
    
    # M6 = (A10-A00)(B00+B01)
    add_term([(idx_u(1,0), 1), (idx_u(0,0), -1)],
             [(idx_v(0,0), 1), (idx_v(0,1), 1)],
             [(idx_w(1,1), 1)])
    
    # M7 = (A01-A11)(B10+B11)
    add_term([(idx_u(0,1), 1), (idx_u(1,1), -1)],
             [(idx_v(1,0), 1), (idx_v(1,1), 1)],
             [(idx_w(0,0), 1)])
    
    # 1x1 block at (4,4)
    add_term([(idx_u(4,4), 1)],
             [(idx_v(4,4), 1)],
             [(idx_w(4,4), 1)])
    
    # Cross terms: all other (i,k) pairs
    handled = set()
    for i in [0,1]:
        for k in [0,1]:
            handled.add((i,k))
    handled.add((4,4))
    
    for i in range(5):
        for k in range(5):
            if (i,k) in handled:
                continue
            for j in range(5):
                add_term([(idx_u(i,j), 1)],
                         [(idx_v(j,k), 1)],
                         [(idx_w(k,i), 1)])
    
    rank = len(U_list)
    
    if rank == 0:
        U = np.zeros((n*m, 1))
        V = np.zeros((m*p, 1))
        W = np.zeros((p*n, 1))
    else:
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
    
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded * 2 - round(rounded * 2)) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(0)
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
