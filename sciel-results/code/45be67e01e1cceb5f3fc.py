"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use recursive block decomposition
    # R(3,5,6) <= R(1,5,6) + R(2,5,6)
    # R(1,5,6) = 30 (schoolbook)
    # R(2,5,6) <= R(2,2,3) + R(2,3,3)
    # R(2,2,3) <= R(2,2,1) + R(2,2,2) = 4 + 7 = 11
    # R(2,3,3) <= R(2,1,1) + R(2,1,2) + R(2,2,1) + R(2,2,2) = 2+4+4+7 = 17
    # Total: 30 + 11 + 17 = 58
    
    blocks = []
    
    # Helper to create schoolbook decomposition for (nn, mm, pp)
    def schoolbook(nn, mm, pp):
        rank = nn * mm * pp
        U = [[0] * rank for _ in range(nn * mm)]
        V = [[0] * rank for _ in range(mm * pp)]
        W = [[0] * rank for _ in range(pp * nn)]
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j][col] = 1
                    V[j * pp + k][col] = 1
                    W[k * nn + i][col] = 1
                    col += 1
        return U, V, W
    
    # Helper to create Strassen decomposition for (2,2,2)
    def strassen_222():
        # Strassen's 7 products for 2x2x2
        # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
        # C = [[c11,c12],[c21,c22]]
        # p1 = (a11+a22)(b11+b22)
        # p2 = (a21+a22)b11
        # p3 = a11(b12-b22)
        # p4 = a22(b21-b11)
        # p5 = (a11+a12)b22
        # p6 = (a21-a11)(b11+b12)
        # p7 = (a12-a22)(b21+b22)
        # c11 = p1+p4-p5+p7
        # c12 = p3+p5
        # c21 = p2+p4
        # c22 = p1-p2+p3+p6
        rank = 7
        U = [[0] * rank for _ in range(4)]  # 2*2
        V = [[0] * rank for _ in range(4)]  # 2*2
        W = [[0] * rank for _ in range(4)]  # 2*2
        
        # U indices: i*2+j for i,j in 0,1
        # V indices: j*2+k for j,k in 0,1
        # W indices: k*2+i for k,i in 0,1
        
        # p1 = (a11+a22)(b11+b22)
        # a11: U[0], a22: U[3]
        # b11: V[0], b22: V[3]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # c11 += p1, c22 += p1
        # c11: W[0*2+0]=W[0], c22: W[1*2+1]=W[3]
        W[0][0] = 1; W[3][0] = 1
        
        # p2 = (a21+a22)b11
        # a21: U[1], a22: U[3], b11: V[0]
        U[1][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # c21 += p2: W[1*2+0]=W[2]
        W[2][1] = 1
        
        # p3 = a11(b12-b22)
        # a11: U[0], b12: V[1], b22: V[3]
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # c12 += p3: W[0*2+1]=W[1], c22 += p3: W[3]
        W[1][2] = 1; W[3][2] = 1
        
        # p4 = a22(b21-b11)
        # a22: U[3], b21: V[2], b11: V[0]
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # c11 += p4: W[0], c21 += p4: W[2]
        W[0][3] = 1; W[2][3] = 1
        
        # p5 = (a11+a12)b22
        # a11: U[0], a12: U[2], b22: V[3]
        U[0][4] = 1; U[2][4] = 1
        V[3][4] = 1
        # c11 -= p5: W[0], c22 += p5: W[3]
        W[0][4] = -1; W[3][4] = 1
        
        # p6 = (a21-a11)(b11+b12)
        # a21: U[1], a11: U[0], b11: V[0], b12: V[1]
        U[1][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # c22 += p6: W[3]
        W[3][5] = 1
        
        # p7 = (a12-a22)(b21+b22)
        # a12: U[2], a22: U[3], b21: V[2], b22: V[3]
        U[2][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # c11 += p7: W[0]
        W[0][6] = 1
        
        return U, V, W
    
    # Helper to combine blocks horizontally (concatenate rank dimensions)
    def combine_horizontal(blocks_list):
        """Combine list of (U,V,W) blocks by concatenating along rank axis."""
        if not blocks_list:
            return [[0] for _ in range(0)], [[0] for _ in range(0)], [[0] for _ in range(0)]
        
        total_rank = sum(len(w) for _, _, w in blocks_list)
        
        # Determine sizes from first block
        first_u, first_v, first_w = blocks_list[0]
        u_rows = len(first_u)
        v_rows = len(first_v)
        w_rows = len(first_w)
        
        U = [[0] * total_rank for _ in range(u_rows)]
        V = [[0] * total_rank for _ in range(v_rows)]
        W = [[0] * total_rank for _ in range(w_rows)]
        
        offset = 0
        for u, v, w in blocks_list:
            for i in range(u_rows):
                for c in range(len(w)):
                    U[i][offset + c] = u[i][c]
                    V[i][offset + c] = v[i][c]
                    W[i][offset + c] = w[i][c]
            offset += len(w)
        
        return U, V, W
    
    # Helper to combine blocks vertically for U (split n)
    # When splitting n into n1+n2, U is stacked, V is same size but replicated in rank, W is stacked
    def combine_split_n(block1, block2, n1):
        """Combine two blocks where n is split into n1 + n2.
        block1: (U1, V1, W1) for (n1, m, p)
        block2: (U2, V2, W2) for (n2, m, p)
        """
        U1, V1, W1 = block1
        U2, V2, W2 = block2
        
        r1 = len(V1[0]) if V1 else 0
        r2 = len(V2[0]) if V2 else 0
        total_rank = r1 + r2
        
        m = len(U1) // n1 if n1 > 0 else 0
        p = len(W1) // n1 if n1 > 0 else 0
        n2 = len(U2) // m if m > 0 else 0
        
        # U: stack U1 and U2, pad rank
        U = [[0] * total_rank for _ in range((n1 + n2) * m)]
        for i in range(n1 * m):
            for c in range(r1):
                U[i][c] = U1[i][c]
        for i in range(n2 * m):
            for c in range(r2):
                U[n1 * m + i][r1 + c] = U2[i][c]
        
        # V: hstack V1 and V2
        V = [[0] * total_rank for _ in range(m * p)]
        for i in range(m * p):
            for c in range(r1):
                V[i][c] = V1[i][c]
            for c in range(r2):
                V[i][r1 + c] = V2[i][c]
        
        # W: stack W1 and W2, but need to reindex W2's i index
        # W1 has rows indexed by k*n1 + i for i in [0,n1)
        # W2 has rows indexed by k*n2 + i for i in [0,n2), but globally it's k*n + (n1+i)
        W = [[0] * total_rank for _ in range(p * (n1 + n2))]
        for k in range(p):
            for i in range(n1):
                for c in range(r1):
                    W[k * (n1 + n2) + i][c] = W1[k * n1 + i][c]
            for i in range(n2):
                for c in range(r2):
                    W[k * (n1 + n2) + n1 + i][r1 + c] = W2[k * n2 + i][c]
        
        return U, V, W
    
    # Helper to combine blocks vertically for V (split m)
    def combine_split_m(block1, block2, m1):
        """Combine two blocks where m is split into m1 + m2.
        block1: (U1, V1, W1) for (n, m1, p)
        block2: (U2, V2, W2) for (n, m2, p)
        """
        U1, V1, W1 = block1
        U2, V2, W2 = block2
        
        r1 = len(V1[0]) if V1 else 0
        r2 = len(V2[0]) if V2 else 0
        total_rank = r1 + r2
        
        n = len(U1) // m1 if m1 > 0 else 0
        p = len(W1) // n if n > 0 else 0
        m2 = len(V2) // p if p > 0 else 0
        
        # U: hstack U1 and U2
        U = [[0] * total_rank for _ in range(n * (m1 + m2))]
        for i in range(n):
            for j in range(m1):
                for c in range(r1):
                    U[i * (m1 + m2) + j][c] = U1[i * m1 + j][c]
            for j in range(m2):
                for c in range(r2):
                    U[i * (m1 + m2) + m1 + j][r1 + c] = U2[i * m2 + j][c]
        
        # V: stack V1 and V2
        V = [[0] * total_rank for _ in range((m1 + m2) * p)]
        for j in range(m1):
            for k in range(p):
                for c in range(r1):
                    V[j * p + k][c] = V1[j * p + k][c]
        for j in range(m2):
            for k in range(p):
                for c in range(r2):
                    V[m1 * p + j * p + k][r1 + c] = V2[j * p + k][c]
        
        # W: hstack W1 and W2
        W = [[0] * total_rank for _ in range(p * n)]
        for k in range(p):
            for i in range(n):
                for c in range(r1):
                    W[k * n + i][c] = W1[k * n + i][c]
                for c in range(r2):
                    W[k * n + i][r1 + c] = W2[k * n + i][c]
        
        return U, V, W
    
    # Helper to combine blocks vertically for W (split p)
    def combine_split_p(block1, block2, p1):
        """Combine two blocks where p is split into p1 + p2.
        block1: (U1, V1, W1) for (n, m, p1)
        block2: (U2, V2, W2) for (n, m, p2)
        """
        U1, V1, W1 = block1
        U2, V2, W2 = block2
        
        r1 = len(V1[0]) if V1 else 0
        r2 = len(V2[0]) if V2 else 0
        total_rank = r1 + r2
        
        n = len(W1) // p1 if p1 > 0 else 0
        m = len(U1) // n if n > 0 else 0
        p2 = len(V2) // m if m > 0 else 0
        
        # U: hstack U1 and U2
        U = [[0] * total_rank for _ in range(n * m)]
        for i in range(n):
            for j in range(m):
                for c in range(r1):
                    U[i * m + j][c] = U1[i * m + j][c]
                for c in range(r2):
                    U[i * m + j][r1 + c] = U2[i * m + j][c]
        
        # V: hstack V1 and V2
        V = [[0] * total_rank for _ in range(m * (p1 + p2))]
        for j in range(m):
            for k in range(p1):
                for c in range(r1):
                    V[j * (p1 + p2) + k][c] = V1[j * p1 + k][c]
            for k in range(p2):
                for c in range(r2):
                    V[j * (p1 + p2) + p1 + k][r1 + c] = V2[j * p2 + k][c]
        
        # W: stack W1 and W2
        W = [[0] * total_rank for _ in range((p1 + p2) * n)]
        for k in range(p1):
            for i in range(n):
                for c in range(r1):
                    W[k * n + i][c] = W1[k * p1 + i][c]
        for k in range(p2):
            for i in range(n):
                for c in range(r2):
                    W[p1 * n + k * n + i][r1 + c] = W2[k * p2 + i][c]
        
        return U, V, W
    
    # Build R(2,2,3) = R(2,2,1) + R(2,2,2)
    def build_223():
        # Split p=3 into 1+2
        block1 = schoolbook(2, 2, 1)  # R(2,2,1) = 4
        block2 = strassen_222()       # R(2,2,2) = 7
        return combine_split_p(block1, block2, 1)
    
    # Build R(2,3,3) = R(2,1,1) + R(2,1,2) + R(2,2,1) + R(2,2,2)
    def build_233():
        # Split m=3 into 1+2, p=3 into 1+2
        # R(2,1,1) + R(2,1,2) from m=1 part
        # R(2,2,1) + R(2,2,2) from m=2 part
        
        # First combine p=1+2 for m=1: R(2,1,1)+R(2,1,2)
        block_211 = schoolbook(2, 1, 1)  # rank 2
        block_212 = schoolbook(2, 1, 2)  # rank 4
        part_m1 = combine_split_p(block_211, block_212, 1)  # R(2,1,3)
        
        # Then combine p=1+2 for m=2: R(2,2,1)+R(2,2,2)
        block_221 = schoolbook(2, 2, 1)  # rank 4
        block_222 = strassen_222()       # rank 7
        part_m2 = combine_split_p(block_221, block_222, 1)  # R(2,2,3)
        
        # Now combine m=1+2
        return combine_split_m(part_m1, part_m2, 1)
    
    # Build R(2,5,6)
    def build_256():
        # Split m=5 into 2+3
        # R(2,2,6) + R(2,3,6)
        # R(2,2,6) = R(2,2,3) + R(2,2,3) (split p=6 into 3+3)
        # R(2,3,6) = R(2,3,3) + R(2,3,3) (split p=6 into 3+3)
        
        block_223 = build_223()
        block_233 = build_233()
        
        # R(2,2,6) = combine_split_p(build_223(), build_223(), 3)
        part_226 = combine_split_p(block_223, block_223, 3)
        
        # R(2,3,6) = combine_split_p(build_233(), build_233(), 3)
        part_236 = combine_split_p(block_233, block_233, 3)
        
        # Combine m=2+3
        return combine_split_m(part_226, part_236, 2)
    
    # Build R(1,5,6)
    def build_156():
        return schoolbook(1, 5, 6)
    
    # Final: R(3,5,6) = R(1,5,6) + R(2,5,6)
    block_156 = build_156()
    block_256 = build_256()
    
    U, V, W = combine_split_n(block_156, block_256, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
