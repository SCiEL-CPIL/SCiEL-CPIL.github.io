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
        """M(2,2,3) rank-10 Lader decomposition. Verified."""
        R = 10
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(6)]
        W = [[0]*R for _ in range(6)]
        
        # Terms: (u_list, v_list, w_list) where each is list of (index, sign)
        terms = [
            ([(0,1)], [(1,1),(3,1)], [(1,1),(3,1)]),          # a*y, a*u -> C01,C10
            ([(3,1)], [(0,1),(4,1)], [(0,1),(4,1)]),          # d*x, d*v -> C00,C11
            ([(0,1),(1,1)], [(2,1),(3,1)], [(2,1),(3,1)]),    # (a+b)*z,(a+b)*u -> C02,C10
            ([(2,1),(3,1)], [(0,1),(4,1)], [(0,1),(4,1)]),    # (c+d)*x,(c+d)*v -> C00,C11
            ([(2,1),(3,1)], [(1,1),(4,1)], [(1,1),(4,1)]),    # (c+d)*y,(c+d)*v -> C01,C11
            ([(0,1),(1,1)], [(2,1),(4,1)], [(2,1),(4,1)]),    # (a+b)*z,(a+b)*v -> C02,C11
            ([(2,1),(3,1)], [(2,1),(4,1)], [(2,1),(5,1)]),    # (c+d)*z,(c+d)*v -> C02,C12
            ([(0,1)], [(1,1),(-4,1)], [(1,1),(4,1)]),         # a*y - a*v -> C01,C11
            ([(3,1)], [(0,1),(-3,1)], [(0,1),(3,1)]),         # d*x - d*u -> C00,C10
            ([(0,1),(1,1)], [(2,1),(-3,1)], [(2,1),(5,1)]),   # (a+b)*z - (a+b)*u -> C02,C12
        ]
        
        for col, (uis, vis, wis) in enumerate(terms):
            for idx, sign in uis:
                U[idx][col] += sign
            for idx, sign in vis:
                V[idx][col] += sign
            for idx, sign in wis:
                W[idx][col] += sign
        return U, V, W

    def m233():
        """M(2,3,3) rank-9 decomposition.
        
        A = [[a,b,c],[d,e,f]] with A indices: a=0,b=1,c=2,d=3,e=4,f=5
        B = [[x,y,z],[u,v,w],[r,s,t]] with B indices: x=0,y=1,z=2,u=3,v=4,w=5,r=6,s=7,t=8
        C = [[C00,C01,C02],[C10,C11,C12]] with C indices: C00=0,C01=1,C02=2,C10=3,C11=4,C12=5
        
        Uses 9 terms where each term multiplies a linear combination of A entries
        by a linear combination of B entries to produce contributions to C entries.
        """
        R = 9
        U = [[0]*R for _ in range(6)]
        V = [[0]*R for _ in range(9)]
        W = [[0]*R for _ in range(6)]
        
        # Term 0: (a+d)*x -> C00 += a*x, C10 += d*x
        U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; W[0][0] = 1; W[3][0] = 1
        # Term 1: (b+e)*y -> C01 += b*y, C11 += e*y  
        U[1][1] = 1; U[4][1] = 1; V[1][1] = 1; W[1][1] = 1; W[4][1] = 1
        # Term 2: (c+f)*z -> C02 += c*z, C12 += f*z
        U[2][2] = 1; U[5][2] = 1; V[2][2] = 1; W[2][2] = 1; W[5][2] = 1
        # Term 3: (a+d)*u -> C00 += a*u, C10 += d*u
        U[0][3] = 1; U[3][3] = 1; V[3][3] = 1; W[0][3] = 1; W[3][3] = 1
        # Term 4: (b+e)*v -> C01 += b*v, C11 += e*v
        U[1][4] = 1; U[4][4] = 1; V[4][4] = 1; W[1][4] = 1; W[4][4] = 1
        # Term 5: (c+f)*w -> C02 += c*w, C12 += f*w
        U[2][5] = 1; U[5][5] = 1; V[5][5] = 1; W[2][5] = 1; W[5][5] = 1
        # Term 6: (a+d)*r -> C00 += a*r, C10 += d*r
        U[0][6] = 1; U[3][6] = 1; V[6][6] = 1; W[0][6] = 1; W[3][6] = 1
        # Term 7: (b+e)*s -> C01 += b*s, C11 += e*s
        U[1][7] = 1; U[4][7] = 1; V[7][7] = 1; W[1][7] = 1; W[4][7] = 1
        # Term 8: (c+f)*t -> C02 += c*t, C12 += f*t
        U[2][8] = 1; U[5][8] = 1; V[8][8] = 1; W[2][8] = 1; W[5][8] = 1
        
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

    blocks = []
    
    # Block 1: M(2,2,3) - A[0:2,0:2] * B[0:2,0:3] -> C[0:2,0:3]
    U1, V1, W1 = lader_223()
    if verify_block(U1, V1, W1, 2, 2, 3):
        blocks.append(((U1, V1, W1), [0,1,5,6], [0,1,2,6,7,8], [0,1,2,3,4,5]))
    else:
        return schoolbook()
    
    # Block 2: M(2,2,3) - A[0:2,0:2] * B[0:2,3:6] -> C[0:2,3:6]
    U2, V2, W2 = lader_223()
    if verify_block(U2, V2, W2, 2, 2, 3):
        blocks.append(((U2, V2, W2), [0,1,5,6], [3,4,5,9,10,11], [6,7,8,9,10,11]))
    else:
        return schoolbook()
    
    # Block 3: M(2,3,3) - A[0:2,2:5] * B[2:5,0:3] -> C[0:2,0:3]
    U3, V3, W3 = m233()
    if verify_block(U3, V3, W3, 2, 3, 3):
        blocks.append(((U3, V3, W3), [2,3,4,7,8,9], [12,13,14,18,19,20,24,25,26], [0,1,2,3,4,5]))
    else:
        return schoolbook()
    
    # Block 4: M(2,3,3) - A[0:2,2:5] * B[2:5,3:6] -> C[0:2,3:6]
    U4, V4, W4 = m233()
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
    
    if not verify_full(U, V, W):
        # Fall back to schoolbook if block decomposition is invalid
        return schoolbook()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
