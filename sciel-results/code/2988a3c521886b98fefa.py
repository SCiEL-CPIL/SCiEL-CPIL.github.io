"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    # We'll use a tiled Strassen-based approach
    # Split j (4) into 2 groups of 2: j_groups = [(0,1), (2,3)]
    # Split k (4) into 2 groups of 2: k_groups = [(0,1), (2,3)]
    # For each (j_group, k_group) tile:
    #   - Rows 0,1: 2x2x2 Strassen, rank 7
    #   - Row 2: 1x2x2 schoolbook, rank 4
    # Total: 4 tiles * (7 + 4) = 44
    
    U = []
    V = []
    W = []
    
    # Strassen 2x2x2 rank-7 factors
    # For 2x2 matrix mult: A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    # M1 = (a+d)(e+h): U=[1,1,0,0], V=[1,0,1,0], W=[1,0,0,1]
    # M2 = (c+d)e:     U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
    # M3 = a(f-h):     U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
    # M4 = d(g-e):     U=[0,0,0,1], V=[0,0,1,-1], W=[0,0,1,0]
    # M5 = (a+b)h:     U=[1,1,0,0], V=[0,0,0,1], W=[-1,0,0,1]
    # M6 = (c-a)(e+f): U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
    # M7 = (b-d)(g+h): U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]
    
    strassen_U = [
        [1, 1, 0, 0],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    strassen_V = [
        [1, 0, 1, 0],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [0, 0, 1, -1],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 0],   # M2
        [0, 1, 0, 0],   # M3
        [0, 0, 1, 0],   # M4
        [-1, 0, 0, 1],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]
    
    # 1x2x2 schoolbook rank-4 factors
    # A_row = [a, b], B = [[e,f],[g,h]], C = [C0, C1]
    # C0 = a*e + b*g
    # C1 = a*f + b*h
    # Term 1: a*e -> U=[1,0], V=[1,0,0,0], W=[1,0]
    # Term 2: b*g -> U=[0,1], V=[0,0,1,0], W=[1,0]
    # Term 3: a*f -> U=[1,0], V=[0,1,0,0], W=[0,1]
    # Term 4: b*h -> U=[0,1], V=[0,0,0,1], W=[0,1]
    
    school1_U = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
    ]
    school1_V = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    school1_W = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ]
    
    j_groups = [(0, 1), (2, 3)]
    k_groups = [(0, 1), (2, 3)]
    
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    for jg in j_groups:
        for kg in k_groups:
            # Strassen part for rows 0,1 (2 rows), jg (2 cols), kg (2 cols)
            for s in range(7):
                # U: 3x4 matrix, we fill row 0 and row 1, cols jg[0], jg[1]
                u_row = [0] * 4
                u_row[jg[0]] = strassen_U[s][0]
                u_row[jg[1]] = strassen_U[s][1]
                U_rows.append(u_row)  # row 0 of A
                u_row2 = [0] * 4
                u_row2[jg[0]] = strassen_U[s][2]
                u_row2[jg[1]] = strassen_U[s][3]
                U_rows.append(u_row2)  # row 1 of A
                U_rows.append([0] * 4)  # row 2 of A (not used in Strassen part)
                
                # V: 4x4 matrix, we fill rows jg[0], jg[1], cols kg[0], kg[1]
                v_row0 = [0] * 4
                v_row0[kg[0]] = strassen_V[s][0]
                v_row0[kg[1]] = strassen_V[s][1]
                V_rows.append(v_row0)  # row jg[0] of B
                v_row1 = [0] * 4
                v_row1[kg[0]] = strassen_V[s][2]
                v_row1[kg[1]] = strassen_V[s][3]
                V_rows.append(v_row1)  # row jg[1] of B
                V_rows.append([0] * 4)  # other rows of B
                V_rows.append([0] * 4)
                
                # W: 4x3 matrix (p*n = 4*3), index k*n+i
                # For kg[0]: cols 0,1,2 (i=0,1,2)
                # For kg[1]: cols 0,1,2 (i=0,1,2)
                w_row0 = [0] * 3
                w_row0[0] = strassen_W[s][0]
                w_row0[1] = strassen_W[s][1]
                w_row0[2] = 0
                W_rows.append(w_row0)  # k=kg[0]
                w_row1 = [0] * 3
                w_row1[0] = strassen_W[s][2]
                w_row1[1] = strassen_W[s][3]
                w_row1[2] = 0
                W_rows.append(w_row1)  # k=kg[1]
                W_rows.append([0] * 3)  # other k rows
                W_rows.append([0] * 3)
                
                rank += 1
            
            # Schoolbook part for row 2, jg (2 cols), kg (2 cols)
            for s in range(4):
                # U: row 2 of A, cols jg[0], jg[1]
                u_row = [0] * 4
                u_row[jg[0]] = school1_U[s][0]
                u_row[jg[1]] = school1_U[s][1]
                U_rows.append([0] * 4)  # row 0
                U_rows.append([0] * 4)  # row 1
                U_rows.append(u_row)     # row 2
                
                # V: rows jg[0], jg[1] of B, cols kg[0], kg[1]
                v_row0 = [0] * 4
                v_row0[kg[0]] = school1_V[s][0]
                v_row0[kg[1]] = school1_V[s][1]
                V_rows.append(v_row0)
                v_row1 = [0] * 4
                v_row1[kg[0]] = school1_V[s][2]
                v_row1[kg[1]] = school1_V[s][3]
                V_rows.append(v_row1)
                V_rows.append([0] * 4)
                V_rows.append([0] * 4)
                
                # W: k=kg[0] and k=kg[1], only i=2
                w_row0 = [0, 0, school1_W[s][0]]  # k=kg[0], i=2
                W_rows.append(w_row0)
                w_row1 = [0, 0, school1_W[s][1]]  # k=kg[1], i=2
                W_rows.append(w_row1)
                W_rows.append([0] * 3)
                W_rows.append([0] * 3)
                
                rank += 1
    
    # Build final matrices
    U = U_rows  # shape (12, rank)
    V = V_rows  # shape (16, rank)
    W = W_rows  # shape (12, rank)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
