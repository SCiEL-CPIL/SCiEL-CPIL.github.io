"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 110  # seconds, safety margin
    
    n, m, p = DIMENSIONS
    # Partition: rows {0,1},{2}; middle {0,1},{2,3}; cols {0,1},{2,3},{4,5},{6,7}
    row_blocks = [(0, 2), (2, 3)]
    mid_blocks = [(0, 2), (2, 4)]
    col_blocks = [(0, 2), (2, 4), (4, 6), (6, 8)]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for r_start, r_end in row_blocks:
        for m_start, m_end in mid_blocks:
            for c_start, c_end in col_blocks:
                r_size = r_end - r_start
                m_size = m_end - m_start
                c_size = c_end - c_start
                
                if r_size == 2 and m_size == 2 and c_size == 2:
                    # Strassen 2x2x2: 7 products
                    strassen_products = [
                        # (U_indices, V_indices, W_indices)
                        # M1 = (A00+A11)(B00+B11)
                        ([(0,0),(1,1)], [(0,0),(1,1)], [(0,0),(1,1)]),
                        # M2 = (A10+A11)B00
                        ([(1,0),(1,1)], [(0,0)], [(1,0)]),
                        # M3 = A00(B10-B11)
                        ([(0,0)], [(1,0),(1,1)], [(0,0),(1,0)]),
                        # M4 = A11(B01-B00)
                        ([(1,1)], [(0,1),(0,0)], [(1,0),(1,1)]),
                        # M5 = (A00+A01)B11
                        ([(0,0),(0,1)], [(1,1)], [(0,0),(1,0)]),
                        # M6 = (A10-A00)(B00+B10)
                        ([(1,0),(0,0)], [(0,0),(1,0)], [(1,0),(1,1)]),
                        # M7 = (A01-A11)(B01+B11)
                        ([(0,1),(1,1)], [(0,1),(1,1)], [(0,0),(1,0)]),
                    ]
                    strassen_signs = [
                        [1, 1], [1, 1], [1, -1], [1, -1], [1, 1], [1, -1], [1, -1],
                    ]
                    # W signs need to be per-output, not per-input
                    # Let me redo this more carefully
                    
                    # Actually, let me just directly assign
                    for prod in range(7):
                        U_col = [0] * (n * m)
                        V_col = [0] * (m * p)
                        W_col = [0] * (p * n)
                        
                        if prod == 0:  # M1 = (A00+A11)(B00+B11)
                            U_col[(r_start+0)*m + (m_start+0)] = 1
                            U_col[(r_start+1)*m + (m_start+1)] = 1
                            V_col[(m_start+0)*p + (c_start+0)] = 1
                            V_col[(m_start+1)*p + (c_start+1)] = 1
                            W_col[(c_start+0)*n + (r_start+0)] = 1
                            W_col[(c_start+1)*n + (r_start+1)] = 1
                        elif prod == 1:  # M2 = (A10+A11)B00
                            U_col[(r_start+1)*m + (m_start+0)] = 1
                            U_col[(r_start+1)*m + (m_start+1)] = 1
                            V_col[(m_start+0)*p + (c_start+0)] = 1
                            W_col[(c_start+0)*n + (r_start+1)] = 1
                        elif prod == 2:  # M3 = A00(B10-B11)
                            U_col[(r_start+0)*m + (m_start+0)] = 1
                            V_col[(m_start+1)*p + (c_start+0)] = 1
                            V_col[(m_start+1)*p + (c_start+1)] = -1
                            W_col[(c_start+0)*n + (r_start+0)] = 1
                            W_col[(c_start+1)*n + (r_start+0)] = 1
                        elif prod == 3:  # M4 = A11(B01-B00)
                            U_col[(r_start+1)*m + (m_start+1)] = 1
                            V_col[(m_start+0)*p + (c_start+1)] = 1
                            V_col[(m_start+0)*p + (c_start+0)] = -1
                            W_col[(c_start+0)*n + (r_start+1)] = 1
                            W_col[(c_start+1)*n + (r_start+1)] = 1
                        elif prod == 4:  # M5 = (A00+A01)B11
                            U_col[(r_start+0)*m + (m_start+0)] = 1
                            U_col[(r_start+0)*m + (m_start+1)] = 1
                            V_col[(m_start+1)*p + (c_start+1)] = 1
                            W_col[(c_start+0)*n + (r_start+0)] = -1
                            W_col[(c_start+1)*n + (r_start+0)] = 1
                        elif prod == 5:  # M6 = (A10-A00)(B00+B10)
                            U_col[(r_start+1)*m + (m_start+0)] = 1
                            U_col[(r_start+0)*m + (m_start+0)] = -1
                            V_col[(m_start+0)*p + (c_start+0)] = 1
                            V_col[(m_start+1)*p + (c_start+0)] = 1
                            W_col[(c_start+0)*n + (r_start+1)] = 1
                            W_col[(c_start+1)*n + (r_start+1)] = 1
                        elif prod == 6:  # M7 = (A01-A11)(B01+B11)
                            U_col[(r_start+0)*m + (m_start+1)] = 1
                            U_col[(r_start+1)*m + (m_start+1)] = -1
                            V_col[(m_start+0)*p + (c_start+1)] = 1
                            V_col[(m_start+1)*p + (c_start+1)] = 1
                            W_col[(c_start+0)*n + (r_start+0)] = 1
                            W_col[(c_start+1)*n + (r_start+0)] = 1
                        
                        U_cols.append(U_col)
                        V_cols.append(V_col)
                        W_cols.append(W_col)
                else:
                    # Schoolbook
                    for dr in range(r_size):
                        for dm in range(m_size):
                            for dc in range(c_size):
                                U_col = [0] * (n * m)
                                V_col = [0] * (m * p)
                                W_col = [0] * (p * n)
                                U_col[(r_start+dr)*m + (m_start+dm)] = 1
                                V_col[(m_start+dm)*p + (c_start+dc)] = 1
                                W_col[(c_start+dc)*n + (r_start+dr)] = 1
                                U_cols.append(U_col)
                                V_cols.append(V_col)
                                W_cols.append(W_col)
    
    rank = len(U_cols)
    U = list(zip(*U_cols)) if U_cols else []
    V = list(zip(*V_cols)) if V_cols else []
    W = list(zip(*W_cols)) if W_cols else []
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
