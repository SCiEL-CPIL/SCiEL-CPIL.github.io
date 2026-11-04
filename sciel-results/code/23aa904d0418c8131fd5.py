import numpy as np

def construct():
    n, m, p = 2, 5, 6
    # Total rank estimate:
    # Part 1 (j=0,1): 3 * Rank(2x2x2) = 3 * 7 = 21
    # Part 2 (j=2,3,4): 3 * Rank(2x3x2) = 3 * 11 = 33
    # Total Rank = 54
    
    rank = 54
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper: Strassen 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    def get_strassen_2x2x2():
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # A indices: 0:a, 1:b, 2:c, 3:d
        # B indices: 0:e, 1:f, 2:g, 3:h
        # C indices: 0:c00, 1:c01, 2:c10, 3:c11
        
        U_s = np.zeros((4, 7), dtype=int)
        V_s = np.zeros((4, 7), dtype=int)
        W_s = np.zeros((4, 7), dtype=int)
        
        # r=0: m1
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        # r=1: m2
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = -1
        
        # r=2: m3
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1
        
        # r=3: m4
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[2, 3] = 1
        
        # r=4: m5
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1
        
        # r=5: m6
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1
        
        # r=6: m7
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1
        
        return U_s, V_s, W_s

    # Helper: Hybrid 2x3x2
    # U: 6x11, V: 6x11, W: 4x11
    def get_hybrid_2x3x2():
        # A indices: 0:a00, 1:a01, 2:a02, 3:a10, 4:a11, 5:a12
        # B indices: 0:b00, 1:b01, 2:b10, 3:b11, 4:b20, 5:b21
        # C indices: 0:c00, 1:c01, 2:c10, 3:c11
        
        U_h = np.zeros((6, 11), dtype=int)
        V_h = np.zeros((6, 11), dtype=int)
        W_h = np.zeros((4, 11), dtype=int)
        
        # Part 1: Strassen on A[0,1,3,4] and B[0,1,2,3]
        U_s, V_s, W_s = get_strassen_2x2x2()
        
        # Map Strassen indices to Hybrid indices
        # Strassen A: 0,1,2,3 -> Hybrid A: 0,1,3,4
        # Strassen B: 0,1,2,3 -> Hybrid B: 0,1,2,3
        # Strassen C: 0,1,2,3 -> Hybrid C: 0,1,2,3
        
        idx_A_map = [0, 1, 3, 4]
        idx_B_map = [0, 1, 2, 3]
        
        for r in range(7):
            for i in range(4):
                if U_s[i, r] != 0:
                    U_h[idx_A_map[i], r] += U_s[i, r]
            for i in range(4):
                if V_s[i, r] != 0:
                    V_h[idx_B_map[i], r] += V_s[i, r]
            for i in range(4):
                if W_s[i, r] != 0:
                    W_h[i, r] += W_s[i, r]
                    
        # Part 2: Schoolbook on A[2,5] and B[4,5]
        # Terms:
        # a02*b20 -> c00
        # a02*b21 -> c01
        # a12*b20 -> c10
        # a12*b21 -> c11
        
        # r=7
        U_h[2, 7] = 1
        V_h[4, 7] = 1
        W_h[0, 7] = 1
        
        # r=8
        U_h[2, 8] = 1
        V_h[5, 8] = 1
        W_h[1, 8] = 1
        
        # r=9
        U_h[5, 9] = 1
        V_h[4, 9] = 1
        W_h[2, 9] = 1
        
        # r=10
        U_h[5, 10] = 1
        V_h[5, 10] = 1
        W_h[3, 10] = 1
        
        return U_h, V_h, W_h

    # Assemble Global Matrices
    
    # Part 1: j in {0, 1}
    # 3 sub-problems for k pairs (0,1), (2,3), (4,5)
    col_offset = 0
    for s in range(3):
        k_start = s * 2
        k_end = k_start + 2
        
        U_s, V_s, W_s = get_strassen_2x2x2()
        
        # Map to Global
        # U: i in {0,1}, j in {0,1} -> global i*5+j
        # V: j in {0,1}, k in {k_start, k_start+1} -> global j*6+k
        # W: k in {k_start, k_start+1}, i in {0,1} -> global k*2+i
        
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    if U_s[i, j if False else 0] != 0: # Wait, U_s is 4x7. Indices are 0..3.
                        pass # Logic below is safer
            
            # Explicit mapping
            # U_s row 0 -> global i=0, j=0 -> 0*5+0=0
            # U_s row 1 -> global i=0, j=1 -> 0*5+1=1
            # U_s row 2 -> global i=1, j=0 -> 1*5+0=5
            # U_s row 3 -> global i=1, j=1 -> 1*5+1=6
            
            u_map = [0, 1, 5, 6]
            for i in range(4):
                if U_s[i, r] != 0:
                    U[u_map[i], col_offset + r] = U_s[i, r]
                    
            # V_s row 0 -> global j=0, k=k_start -> 0*6+k_start
            # V_s row 1 -> global j=0, k=k_start+1 -> 0*6+k_start+1
            # V_s row 2 -> global j=1, k=k_start -> 1*6+k_start
            # V_s row 3 -> global j=1, k=k_start+1 -> 1*6+k_start+1
            
            v_map = [
                0*6 + k_start, 0*6 + k_start + 1,
                1*6 + k_start, 1*6 + k_start + 1
            ]
            for i in range(4):
                if V_s[i, r] != 0:
                    V[v_map[i], col_offset + r] = V_s[i, r]
                    
            # W_s row 0 -> global k=k_start, i=0 -> k_start*2+0
            # W_s row 1 -> global k=k_start, i=1 -> k_start*2+1
            # W_s row 2 -> global k=k_start+1, i=0 -> (k_start+1)*2+0
            # W_s row 3 -> global k=k_start+1, i=1 -> (k_start+1)*2+1
            
            w_map = [
                k_start*2 + 0, k_start*2 + 1,
                (k_start+1)*2 + 0, (k_start+1)*2 + 1
            ]
            for i in range(4):
                if W_s[i, r] != 0:
                    W[w_map[i], col_offset + r] = W_s[i, r]
                    
        col_offset += 7

    # Part 2: j in {2, 3, 4}
    # 3 sub-problems for k pairs (0,1), (2,3), (4,5)
    for s in range(3):
        k_start = s * 2
        k_end = k_start + 2
        
        U_h, V_h, W_h = get_hybrid_2x3x2()
        
        # Map to Global
        # U_h row 0 -> global i=0, j=2 -> 0*5+2=2
        # U_h row 1 -> global i=0, j=3 -> 0*5+3=3
        # U_h row 2 -> global i=0, j=4 -> 0*5+4=4
        # U_h row 3 -> global i=1, j=2 -> 1*5+2=7
        # U_h row 4 -> global i=1, j=3 -> 1*5+3=8
        # U_h row 5 -> global i=1, j=4 -> 1*5+4=9
        
        u_map = [2, 3, 4, 7, 8, 9]
        
        # V_h row 0 -> global j=2, k=k_start -> 2*6+k_start
        # V_h row 1 -> global j=2, k=k_start+1 -> 2*6+k_start+1
        # V_h row 2 -> global j=3, k=k_start -> 3*6+k_start
        # V_h row 3 -> global j=3, k=k_start+1 -> 3*6+k_start+1
        # V_h row 4 -> global j=4, k=k_start -> 4*6+k_start
        # V_h row 5 -> global j=4, k=k_start+1 -> 4*6+k_start+1
        
        v_map = [
            2*6 + k_start, 2*6 + k_start + 1,
            3*6 + k_start, 3*6 + k_start + 1,
            4*6 + k_start, 4*6 + k_start + 1
        ]
        
        # W_h row 0 -> global k=k_start, i=0 -> k_start*2+0
        # W_h row 1 -> global k=k_start, i=1 -> k_start*2+1
        # W_h row 2 -> global k=k_start+1, i=0 -> (k_start+1)*2+0
        # W_h row 3 -> global k=k_start+1, i=1 -> (k_start+1)*2+1
        
        w_map = [
            k_start*2 + 0, k_start*2 + 1,
            (k_start+1)*2 + 0, (k_start+1)*2 + 1
        ]
        
        for r in range(11):
            for i in range(6):
                if U_h[i, r] != 0:
                    U[u_map[i], col_offset + r] = U_h[i, r]
            for i in range(6):
                if V_h[i, r] != 0:
                    V[v_map[i], col_offset + r] = V_h[i, r]
            for i in range(4):
                if W_h[i, r] != 0:
                    W[w_map[i], col_offset + r] = W_h[i, r]
                    
        col_offset += 11

    return {"U": U, "V": V, "W": W}
