DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    # Schoolbook would be rank 64. We use Strassen-based rank 56.
    
    # Strassen M_{2,2,2} rank-7 decomposition
    # U1 (4x7), V1 (4x7), W1 (4x7)
    U1 = [
        [1,  0, 1,  0, 1, -1,  0],  # a11
        [0,  0, 0,  0, 1,  0,  1],  # a12
        [0,  1, 0,  0, 0,  1,  0],  # a21
        [1,  1, 0,  1, 0,  0, -1],  # a22
    ]
    V1 = [
        [1,  1, 0, -1, 0, 1, 0],  # b11
        [0,  0, 0,  1, 0, 1, 0],  # b12
        [0,  0, 1,  0, 0, 0, 1],  # b21
        [1,  0, -1, 0, 1, 0, 1],  # b22
    ]
    W1 = [
        [1, 0,  0, -1, 1, 0, -1],  # c11 (k=0,i=0)
        [0, 0,  1,  0, 1, 0,  0],  # c12 (k=0,i=1)
        [0, 1,  0,  1, 0, 0,  0],  # c21 (k=1,i=0)
        [1, 1, -1,  0, 0, -1, 0],  # c22 (k=1,i=1)
    ]
    
    rank = 56
    U = [[0]*rank for _ in range(n*m)]  # 8 x 56
    V = [[0]*rank for _ in range(m*p)]  # 32 x 56
    W = [[0]*rank for _ in range(p*n)]  # 16 x 56
    
    # Two halves: left (cols 0,1 of A; rows 0,1 of B) and right (cols 2,3 of A; rows 2,3 of B)
    for half in range(2):
        # half=0: left, half=1: right
        col_offset = half * 2  # 0 or 2
        row_offset_B = half * 2  # 0 or 2
        base = half * 28  # 0 or 28
        
        for t1 in range(7):
            for t2 in range(4):
                t = base + t1 * 4 + t2
                
                # U: A elements
                # For half=0: A[0,0],A[0,1],A[1,0],A[1,1] at full indices 0,1,4,5
                # For half=1: A[0,2],A[0,3],A[1,2],A[1,3] at full indices 2,3,6,7
                for i1 in range(2):
                    for j1 in range(2):
                        full_i = i1
                        full_j = col_offset + j1
                        U_idx = full_i * m + full_j
                        U[U_idx][t] += U1[i1 * 2 + j1][t1]
                
                # V: B elements
                # For half=0: B[0,0..7], B[1,0..7] at full indices 0-7, 8-15
                # For half=1: B[2,0..7], B[3,0..7] at full indices 16-23, 24-31
                for j1 in range(2):
                    for k1 in range(8):
                        full_j = row_offset_B + j1
                        full_k = k1
                        V_idx = full_j * p + full_k
                        # V1[j1*2 + k1_outer, t1] where k1_outer is 0 or 1
                        # k1 = k1_outer * 4 + k2
                        k1_outer = k1 // 4
                        k2 = k1 % 4
                        if k2 == t2:
                            V[V_idx][t] += V1[j1 * 2 + k1_outer][t1]
                
                # W: C elements
                # C[i1, k] at full index k*2 + i1
                for i1 in range(2):
                    for k1 in range(8):
                        full_i = i1
                        full_k = k1
                        W_idx = full_k * n + full_i
                        k1_outer = k1 // 4
                        k2 = k1 % 4
                        if k2 == t2:
                            W[W_idx][t] += W1[k1_outer * 2 + i1][t1]
    
    return {"U": U, "V": V, "W": W}
