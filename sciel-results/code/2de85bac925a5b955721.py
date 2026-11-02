"""Improved program using Strassen for 2x2x2 sub-blocks."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    U = []
    V = []
    W = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term. u_entries: dict {(i,j): val}, v_entries: dict {(j,k): val}, w_entries: dict {(k,i): val}"""
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), val in u_entries.items():
            u_col[i * m + j] = val
        for (j, k), val in v_entries.items():
            v_col[j * p + k] = val
        for (k, i), val in w_entries.items():
            w_col[k * n + i] = val
        
        U.append(u_col)
        V.append(v_col)
        W.append(w_col)
    
    # Strassen for 2x2x2: A is 2x2 (rows i, cols j), B is 2x2 (rows j, cols k), C is 2x2 (rows i, cols k)
    # M1 = (A00+A10)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2(j_offset, k_offset):
        """Strassen for sub-block with j indices j_offset, j_offset+1 and k indices k_offset, k_offset+1"""
        j0 = j_offset
        j1 = j_offset + 1
        k0 = k_offset
        k1 = k_offset + 1
        
        # M1 = (A00+A10)(B00+B11)
        add_term({(0,j0):1,(1,j0):1}, {(j0,k0):1,(j1,k1):1}, {(k0,0):1,(k0,1):-1,(k1,0):-1,(k1,1):1})
        # M2 = (A10+A11)B00
        add_term({(1,j0):1,(1,j1):1}, {(j0,k0):1}, {(k0,1):1})
        # M3 = A00(B01-B11)
        add_term({(0,j0):1}, {(j0,k1):1,(j1,k1):-1}, {(k1,0):1,(k1,1):1})
        # M4 = A11(B10-B00)
        add_term({(1,j1):1}, {(j1,k0):1,(j0,k0):-1}, {(k0,0):-1,(k0,1):1})
        # M5 = (A00+A01)B11
        add_term({(0,j0):1,(0,j1):1}, {(j1,k1):1}, {(k0,0):-1,(k1,0):1})
        # M6 = (A10-A00)(B00+B01)
        add_term({(1,j0):1,(0,j0):-1}, {(j0,k0):1,(j0,k1):1}, {(k1,1):1})
        # M7 = (A01-A11)(B10+B11)
        add_term({(0,j1):1,(1,j1):-1}, {(j1,k0):1,(j1,k1):1}, {(k0,0):-1,(k1,0):1,(k1,1):-1})
    
    # Schoolbook for 2x2x3: A is 2x2 (rows i, cols j), B is 2x3 (rows j, cols k), C is 2x3 (rows i, cols k)
    def schoolbook_2x2x3(j_offset, k_offset, num_k):
        """Schoolbook for sub-block with j indices j_offset, j_offset+1 and k indices k_offset..k_offset+num_k-1"""
        j0 = j_offset
        j1 = j_offset + 1
        for i in range(2):
            for j in [j0, j1]:
                for dk in range(num_k):
                    k = k_offset + dk
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})
    
    # Left half: j in {0,1}
    # k in {0,1}: Strassen 2x2x2
    strassen_2x2x2(0, 0)
    # k in {2,3,4}: Schoolbook 2x2x3
    schoolbook_2x2x3(0, 2, 3)
    
    # Right half: j in {2,3}
    # k in {0,1}: Strassen 2x2x2
    strassen_2x2x2(2, 0)
    # k in {2,3,4}: Schoolbook 2x2x3
    schoolbook_2x2x3(2, 2, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
