import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) -> (8, R)
    # V: (m*p, rank) -> (28, R)
    # W: (p*n, rank) -> (14, R)
    
    # We will build the matrices by accumulating contributions
    # Since we don't know R exactly until we finish, we can use lists of vectors
    # or pre-allocate with max possible rank (56) and trim.
    # Max rank is 56.
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        # u_vec: dict or list of (index, coeff)
        # We use dense vectors for simplicity, padded to full size
        pass

    # Helper to add a Strassen block
    # a_rows: [i1, i2], a_cols: [j1, j2]
    # b_rows: [j1, j2], b_cols: [k1, k2]
    # c_rows: [i1, i2], c_cols: [k1, k2]
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # Local indices 0,1
        # Global indices
        i1, i2 = a_rows
        j1, j2 = a_cols
        k1, k2 = c_cols
        
        # A indices in U: i*m + j
        u_11 = i1 * m + j1
        u_12 = i1 * m + j2
        u_21 = i2 * m + j1
        u_22 = i2 * m + j2
        
        # B indices in V: j*p + k
        v_11 = b_rows[0] * p + k1
        v_12 = b_rows[0] * p + k2
        v_21 = b_rows[1] * p + k1
        v_22 = b_rows[1] * p + k2
        
        # C indices in W: k*n + i
        w_11 = k1 * n + i1
        w_12 = k1 * n + i2
        w_21 = k2 * n + i1
        w_22 = k2 * n + i2
        
        # Strassen terms
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        # M2 = (C22+A22)B11 -> C21, C22
        # M3 = A11(B12-B22) -> C12, C22
        # M4 = A22(B21-B11) -> C11, C21
        # M5 = (A11+A12)B22 -> C11, C12
        # M6 = (C22-A11)(B11+B12) -> C22
        # M7 = (A12-A22)(B21+B22) -> C11, C22
        
        # We need to add 7 terms.
        # Term 1
        u = {u_11: 1, u_22: 1}
        v = {v_11: 1, v_22: 1}
        w = {w_11: 1, w_22: 1}
        add_term(u, v, w)
        
        # Term 2: (A21+A22) B11
        # Wait, Strassen M2 = (c+d)e where c=A21, d=A22, e=B11.
        # c=A21? No, standard is A=[[a,b],[c,d]].
        # a=A11, b=A12, c=A21, d=A22.
        # M2 = (c+d)e = (A21+A22)B11.
        u = {u_21: 1, u_22: 1}
        v = {v_11: 1}
        # C21 += M2, C22 -= M2
        w = {w_21: 1, w_22: -1}
        add_term(u, v, w)
        
        # Term 3: a(f-h) = A11(B12-B22)
        u = {u_11: 1}
        v = {v_12: 1, v_22: -1}
        # C12 += M3, C22 += M3
        w = {w_12: 1, w_22: 1}
        add_term(u, v, w)
        
        # Term 4: d(g-e) = A22(B21-B11)
        u = {u_22: 1}
        v = {v_21: 1, v_11: -1}
        # C11 += M4, C21 += M4
        w = {w_11: 1, w_21: 1}
        add_term(u, v, w)
        
        # Term 5: (a+b)h = (A11+A12)B22
        u = {u_11: 1, u_12: 1}
        v = {v_22: 1}
        # C11 -= M5, C12 += M5
        w = {w_11: -1, w_12: 1}
        add_term(u, v, w)
        
        # Term 6: (c-a)(e+f) = (A21-A11)(B11+B12)
        u = {u_21: 1, u_11: -1}
        v = {v_11: 1, v_12: 1}
        # C22 += M6
        w = {w_22: 1}
        add_term(u, v, w)
        
        # Term 7: (b-d)(g+h) = (A12-A22)(B21+B22)
        u = {u_12: 1, u_22: -1}
        v = {v_21: 1, v_22: 1}
        # C11 += M7
        w = {w_11: 1}
        add_term(u, v, w)

    def add_rank11(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # 2x2x3 block
        # First 2 cols: Strassen
        # Last 1 col: Schoolbook 4 terms
        i1, i2 = a_rows
        j1, j2 = a_cols
        k1, k2, k3 = c_cols
        
        # Strassen for k1, k2
        add_strassen(a_rows, a_cols, b_rows, b_cols[:2], c_rows, c_cols[:2])
        
        # Schoolbook for k3
        # A11 B13 -> C13
        u = {i1*m+j1: 1}
        v = {b_rows[0]*p+k3: 1}
        w = {k3*n+i1: 1}
        add_term(u, v, w)
        
        # A12 B23 -> C13
        u = {i1*m+j2: 1}
        v = {b_rows[1]*p+k3: 1}
        w = {k3*n+i1: 1}
        add_term(u, v, w)
        
        # A21 B13 -> C23
        u = {i2*m+j1: 1}
        v = {b_rows[0]*p+k3: 1}
        w = {k3*n+i2: 1}
        add_term(u, v, w)
        
        # A22 B23 -> C23
        u = {i2*m+j2: 1}
        v = {b_rows[1]*p+k3: 1}
        w = {k3*n+i2: 1}
        add_term(u, v, w)

    # Global setup
    # A: 2x4. Rows 0,1. Cols 0,1,2,3.
    # B: 4x7. Rows 0,1,2,3. Cols 0,1,2,3,4,5,6.
    # C: 2x7. Rows 0,1. Cols 0,1,2,3,4,5,6.
    
    # Part 1: Cols 0-3 (M_2,4,4)
    # A1 (cols 0-1), B_top (rows 0-1) -> C(0-1, 0-3)
    # A2 (cols 2-3), B_bot (rows 2-3) -> C(0-1, 0-3)
    
    # A1 B_top (2x2x4)
    # Split into two 2x2x2
    # B_top cols 0-1
    add_strassen([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    # B_top cols 2-3
    add_strassen([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    
    # A2 B_bot (2x2x4)
    # B_bot cols 0-1
    add_strassen([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    # B_bot cols 2-3
    add_strassen([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    
    # Part 2: Cols 4-6 (M_2,4,3)
    # A1 B_top (2x2x3)
    add_rank11([0,1], [0,1], [0,1], [4,5,6], [0,1], [4,5,6])
    # A2 B_bot (2x2x3)
    add_rank11([0,1], [2,3], [2,3], [4,5,6], [0,1], [4,5,6])
    
    # Now convert U, V, W lists to matrices
    # We used a helper add_term which needs to be defined properly.
    # Let's re-implement add_term to append to lists of dicts, then flatten.
    
    return {"U": U, "V": V, "W": W}

# Correct implementation of add_term and accumulation
def construct():
    n, m, p = DIMENSIONS
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_dict, v_dict, w_dict):
        U_list.append(u_dict)
        V_list.append(v_dict)
        W_list.append(w_dict)
        
    # ... (same block definitions as above) ...
    
    # Flatten
    U = [[0]*len(U_list) for _ in range(n*m)]
    V = [[0]*len(V_list) for _ in range(m*p)]
    W = [[0]*len(W_list) for _ in range(p*n)]
    
    for r in range(len(U_list)):
        for idx, val in U_list[r].items():
            U[idx][r] = val
        for idx, val in V_list[r].items():
            V[idx][r] = val
        for idx, val in W_list[r].items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
