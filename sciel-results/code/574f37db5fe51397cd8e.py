import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 130

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    def lader_223():
        """M(2,2,3) rank-10 Lader decomposition."""
        R = 10
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(6)]
        W = [[0]*R for _ in range(6)]
        
        # Terms: (u_list, v_list, w_list)
        terms = [
            ([(0,1)], [(1,1),(3,1)], [(1,1),(3,1)]),
            ([(3,1)], [(0,1),(4,1)], [(0,1),(4,1)]),
            ([(0,1),(1,1)], [(2,1),(3,1)], [(2,1),(3,1)]),
            ([(2,1),(3,1)], [(0,1),(4,1)], [(0,1),(4,1)]),
            ([(2,1),(3,1)], [(1,1),(4,1)], [(1,1),(4,1)]),
            ([(0,1),(1,1)], [(2,1),(4,1)], [(2,1),(4,1)]),
            ([(2,1),(3,1)], [(2,1),(4,1)], [(2,1),(5,1)]),
            ([(0,1)], [(1,1),(-4,1)], [(1,1),(4,1)]),
            ([(3,1)], [(0,1),(-3,1)], [(0,1),(3,1)]),
            ([(0,1),(1,1)], [(2,1),(-3,1)], [(2,1),(5,1)]),
        ]
        
        for col, (uis, vis, wis) in enumerate(terms):
            for idx, sign in uis:
                U[idx][col] += sign
            for idx, sign in vis:
                V[idx][col] += sign
            for idx, sign in wis:
                W[idx][col] += sign
        return U, V, W

    def m213_rank3():
        """M(2,1,3) rank-3 decomposition."""
        R = 3
        U = [[0]*R for _ in range(2)]
        V = [[0]*R for _ in range(3)]
        W = [[0]*R for _ in range(6)] # 2x3 output
        
        # A is 2x1: indices 0,1
        # B is 1x3: indices 0,1,2
        # C is 2x3: indices 0,1,2 (row 0), 3,4,5 (row 1)
        
        # Term 0: (a+d) * r -> C00, C10
        U[0][0] = 1; U[1][0] = 1
        V[0][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # Term 1: (a+d) * s -> C01, C11
        U[0][1] = 1; U[1][1] = 1
        V[1][1] = 1
        W[1][1] = 1; W[4][1] = 1
        
        # Term 2: (a+d) * t -> C02, C12
        U[0][2] = 1; U[1][2] = 1
        V[2][2] = 1
        W[2][2] = 1; W[5][2] = 1
        
        return U, V, W

    def m233_rank13():
        """M(2,3,3) rank-13 decomposition using M(2,2,3) + M(2,1,3)."""
        R1 = 10
        R2 = 3
        R = R1 + R2
        
        U = [[0]*R for _ in range(6)]
        V = [[0]*R for _ in range(9)]
        W = [[0]*R for _ in range(6)]
        
        # Block 1: M(2,2,3)
        # A indices: 0,1 (row 0), 3,4 (row 1) -> Local 0,1,2,3
        # B indices: 0,1 (row 0), 3,4 (row 1) -> Local 0,1,2,3,4,5
        # C indices: 0,1 (col 0,1) -> Local 0,1,2,3,4,5
        U1, V1, W1 = lader_223()
        
        # Map U1 (4 rows) to U (6 rows)
        # Local 0 -> Global 0 (a)
        # Local 1 -> Global 1 (b)
        # Local 2 -> Global 3 (d)
        # Local 3 -> Global 4 (e)
        u_map1 = [0, 1, 3, 4]
        for li, gi in enumerate(u_map1):
            for r in range(R1):
                U[gi][r] += U1[li][r]
                
        # Map V1 (6 rows) to V (9 rows)
        # Local 0 -> Global 0 (x)
        # Local 1 -> Global 1 (y)
        # Local 2 -> Global 2 (z)
        # Local 3 -> Global 3 (u)
        # Local 4 -> Global 4 (v)
        # Local 5 -> Global 5 (w)
        v_map1 = [0, 1, 2, 3, 4, 5]
        for li, gi in enumerate(v_map1):
            for r in range(R1):
                V[gi][r] += V1[li][r]
                
        # Map W1 (6 rows) to W (6 rows)
        w_map1 = [0, 1, 2, 3, 4, 5]
        for li, gi in enumerate(w_map1):
            for r in range(R1):
                W[gi][r] += W1[li][r]
                
        # Block 2: M(2,1,3)
        # A indices: 2 (c), 5 (f) -> Local 0,1
        # B indices: 6 (r), 7 (s), 8 (t) -> Local 0,1,2
        # C indices: 0,1,2,3,4,5
        U2, V2, W2 = m213_rank3()
        
        u_map2 = [2, 5]
        for li, gi in enumerate(u_map2):
            for r in range(R2):
                U[gi][R1 + r] += U2[li][r]
                
        v_map2 = [6, 7, 8]
        for li, gi in enumerate(v_map2):
            for r in range(R2):
                V[gi][R1 + r] += V2[li][r]
                
        w_map2 = [0, 1, 2, 3, 4, 5]
        for li, gi in enumerate(w_map2):
            for r in range(R2):
                W[gi][R1 + r] += W2[li][r]
                
        return U, V, W

    def verify_block(Ub, Vb, Wb, ni, mi, pi):
        Ua = np.array(Ub, float)
        Va = np.array(Vb, float)
        Wa = np.array(Wb, float)
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    s = float(np.dot(Ua[i*mi+j], Va[j*pi+k] * Wa[k*ni+i]))
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    def verify_full(U, V, W):
        Ua = np.array(U, float)
        Va = np.array(V, float)
        Wa = np.array(W, float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i]))
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    # Construct full decomposition
    # A (2x5) split into A1 (2x2) and A2 (2x3)
    # B (5x6) split into B1 (2x6) and B2 (3x6)
    
    blocks = []
    
    # 1. A1 (2x2) * B1a (2x3) -> C[0:2, 0:3]
    U1, V1, W1 = lader_223()
    if verify_block(U1, V1, W1, 2, 2, 3):
        blocks.append(((U1, V1, W1), [0,1,5,6], [0,1,2,6,7,8], [0,1,2,3,4,5]))
    else:
        return schoolbook()
        
    # 2. A1 (2x2) * B1b (2x3) -> C[0:2, 3:6]
    U2, V2, W2 = lader_223()
    if verify_block(U2, V2, W2, 2, 2, 3):
        blocks.append(((U2, V2, W2), [0,1,5,6], [3,4,5,9,10,11], [6,7,8,9,10,11]))
    else:
        return schoolbook()
        
    # 3. A2 (2x3) * B2a (3x3) -> C[0:2, 0:3]
    U3, V3, W3 = m233_rank13()
    if verify_block(U3, V3, W3, 2, 3, 3):
        blocks.append(((U3, V3, W3), [2,3,4,7,8,9], [12,13,14,18,19,20,24,25,26], [0,1,2,3,4,5]))
    else:
        return schoolbook()
        
    # 4. A2 (2x3) * B2b (3x3) -> C[0:2, 3:6]
    U4, V4, W4 = m233_rank13()
    if verify_block(U4, V4, W4, 2, 3, 3):
        blocks.append(((U4, V4, W4), [2,3,4,7,8,9], [15,16,17,21,22,23,27,28,29], [6,7,8,9,10,11]))
    else:
        return schoolbook()
        
    # Merge
    total_rank = sum(len(Ub[0]) for (Ub, Vb, Wb), _, _, _ in blocks)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    offset = 0
    for (Ub, Vb, Wb), umap, vmap, wmap in blocks:
        R = len(Ub[0])
        for li, gi in enumerate(umap):
            for r in range(R):
                U[gi][offset + r] += Ub[li][r]
        for lj, gj in enumerate(vmap):
            for r in range(R):
                V[gj][offset + r] += Vb[lj][r]
        for lk, gk in enumerate(wmap):
            for r in range(R):
                W[gk][offset + r] += Wb[lk][r]
        offset += R
    
    if time.time() < deadline:
        if not verify_full(U, V, W):
            return schoolbook()
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
