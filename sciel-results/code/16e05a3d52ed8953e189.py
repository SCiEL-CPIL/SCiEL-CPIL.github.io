"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS  # 4, 5, 6

    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook fallback
    def schoolbook():
        n, m, p = DIMENSIONS
        R = n*m*p
        U = [[0.0]*R for _ in range(n*m)]
        V = [[0.0]*R for _ in range(m*p)]
        W = [[0.0]*R for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1.0
                    V[j*p+k][c] = 1.0
                    W[k*n+i][c] = 1.0
                    c += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best = (U_sb, V_sb, W_sb, 120)

    if time.time() > deadline - 5:
        return {"U": best[0], "V": best[1], "W": best[2]}

    # 2x2x2 Strassen (rank 7)
    def build_strassen_2x2():
        U = [[0.0]*7 for _ in range(4)]
        V = [[0.0]*7 for _ in range(4)]
        W = [[0.0]*7 for _ in range(4)]
        terms = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
            ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1)]),
            ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,-1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,-1),(1,1,1)]),
        ]
        for s, (ac, bc, cc) in enumerate(terms):
            for (r,c,v) in ac: U[r*2+c][s] += v
            for (r,c,v) in bc: V[r*2+c][s] += v
            for (r,c,v) in cc: W[r*2+c][s] += v
        return U, V, W

    U2, V2, W2 = build_strassen_2x2()

    def verify2(U,V,W):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U[i*2+j][r]*V[j*2+k][r]*W[k*2+i][r] for r in range(7))
                    if abs(s-1.0) > 1e-9:
                        return False
        return True

    if not verify2(U2, V2, W2):
        return {"U": best[0], "V": best[1], "W": best[2]}

    # Tensor square -> 4x4x4 rank 49
    def tensor_square(U2, V2, W2):
        R = 49
        U4 = [[0.0]*R for _ in range(16)]
        V4 = [[0.0]*R for _ in range(16)]
        W4 = [[0.0]*R for _ in range(16)]
        for i1 in range(2):
            for i2 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        for s1 in range(7):
                            for s2 in range(7):
                                col = s1*7+s2
                                u = U2[i1*2+j1][s1]*U2[i2*2+j2][s2]
                                U4[(i1*2+i2)*4 + (j1*2+j2)][col] += u
        for j1 in range(2):
            for j2 in range(2):
                for k1 in range(2):
                    for k2 in range(2):
                        for s1 in range(7):
                            for s2 in range(7):
                                col = s1*7+s2
                                v = V2[j1*2+k1][s1]*V2[j2*2+k2][s2]
                                V4[(j1*2+j2)*4 + (k1*2+k2)][col] += v
        for k1 in range(2):
            for k2 in range(2):
                for i1 in range(2):
                    for i2 in range(2):
                        for s1 in range(7):
                            for s2 in range(7):
                                col = s1*7+s2
                                w = W2[k1*2+i1][s1]*W2[k2*2+i2][s2]
                                W4[(k1*2+k2)*4 + (i1*2+i2)][col] += w
        return U4, V4, W4

    U4, V4, W4 = tensor_square(U2, V2, W2)

    def verify4(U,V,W):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = sum(U[i*4+j][r]*V[j*4+k][r]*W[k*4+i][r] for r in range(49))
                    if abs(s-1.0) > 1e-9:
                        return False
        return True

    if not verify4(U4, V4, W4):
        return {"U": best[0], "V": best[1], "W": best[2]}

    # Build 4x4x2 decomposition using two Strassen 2x2x2 operations
    # A (4x4) split into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # B (4x2) split into top and bottom: B = [B_top; B_bottom] where B_top, B_bottom are 2x2
    # C = A*B = A11*B_top + A12*B_bottom
    # Each 2x2x2 uses Strassen rank 7, total rank 14
    def build_4x4x2():
        R = 14
        U = [[0.0]*R for _ in range(16)]  # 4x4 = 16 rows
        V = [[0.0]*R for _ in range(8)]   # 4x2 = 8 rows
        W = [[0.0]*R for _ in range(8)]   # 2x4 = 8 rows (k*n+i where k=0,1; i=0..3)
        
        # First Strassen: A11 (rows 0-1, cols 0-1) * B_top (rows 0-1, cols 0-1) -> C_top (rows 0-1, cols 0-1)
        # U indices: i*4+j for i=0,1; j=0,1 -> rows 0,1,4,5
        # V indices: j*2+k for j=0,1; k=0,1 -> rows 0,1,2,3
        # W indices: k*4+i for k=0,1; i=0,1 -> rows 0,1,4,5
        
        for s in range(7):
            for (r,c,v) in [(0,0,U2[0][s]), (0,1,U2[1][s]), (1,0,U2[4][s]), (1,1,U2[5][s])]:
                if v != 0:
                    U[r*4+c][s] += v
            for (r,c,v) in [(0,0,V2[0][s]), (0,1,V2[1][s]), (1,0,V2[2][s]), (1,1,V2[3][s])]:
                if v != 0:
                    V[r*2+c][s] += v
            for (r,c,v) in [(0,0,W2[0][s]), (0,1,W2[1][s]), (1,0,W2[4][s]), (1,1,W2[5][s])]:
                if v != 0:
                    W[r*4+c][s] += v
        
        # Second Strassen: A12 (rows 0-1, cols 2-3) * B_bottom (rows 2-3, cols 0-1) -> C_bottom (rows 2-3, cols 0-1)
        # U indices: i*4+j for i=0,1; j=2,3 -> rows 2,3,6,7
        # V indices: j*2+k for j=2,3; k=0,1 -> rows 4,5,6,7
        # W indices: k*4+i for k=0,1; i=2,3 -> rows 2,3,6,7
        
        for s in range(7):
            for (r,c,v) in [(0,0,U2[0][s]), (0,1,U2[1][s]), (1,0,U2[4][s]), (1,1,U2[5][s])]:
                if v != 0:
                    U[r*4+(c+2)][7+s] += v
            for (r,c,v) in [(0,0,V2[0][s]), (0,1,V2[1][s]), (1,0,V2[2][s]), (1,1,V2[3][s])]:
                if v != 0:
                    V[(r+2)*2+c][7+s] += v
            for (r,c,v) in [(0,0,W2[0][s]), (0,1,W2[1][s]), (1,0,W2[4][s]), (1,1,W2[5][s])]:
                if v != 0:
                    W[r*4+(c+2)][7+s] += v
        
        return U, V, W

    U42, V42, W42 = build_4x4x2()

    def verify42(U,V,W):
        # Verify 4x4x2: U is 16x14, V is 8x14, W is 8x14
        # For i in 0..3, k in 0..1: sum_r U[i*4+j, r] * V[j*2+k, r] * W[k*4+i, r] should give delta
        # Actually, we need to verify the full tensor constraint
        for i in range(4):
            for k in range(2):
                for j in range(4):
                    s = sum(U[i*4+j][r]*V[j*2+k][r]*W[k*4+i][r] for r in range(14))
                    # This should be 1 if this is the "diagonal" term, but actually
                    # the 4x4x2 tensor has T[i,j,k] = 1 for all i,j,k (it's the full multiplication)
                    # Wait, no. The tensor for 4x4x2 multiplication is T[i,j,k] where
                    # C[i,k] = sum_j A[i,j]*B[j,k]
                    # So T[i,j,k] = 1 for all valid i,j,k
                    # But in the decomposition, we're only computing part of the full 4x5x6
                    pass
        # For now, just trust the construction and verify in the full context
        return True

    # Assemble 4x5x6, rank 49+14+1 = 64
    R = 64
    U = [[0.0]*R for _ in range(20)]
    V = [[0.0]*R for _ in range(30)]
    W = [[0.0]*R for _ in range(24)]

    # Part 1: A1(4x4) x B1a(4x4) -> C[0:4,0:4], cols 0..48
    for col in range(49):
        for i in range(4):
            for j in range(4):
                if U4[i*4+j][col] != 0:
                    U[i*5+j][col] += U4[i*4+j][col]
        for j in range(4):
            for k in range(4):
                if V4[j*4+k][col] != 0:
                    V[j*6+k][col] += V4[j*4+k][col]
        for k in range(4):
            for i in range(4):
                if W4[k*4+i][col] != 0:
                    W[k*4+i][col] += W4[k*4+i][col]

    # Part 2: A1(4x4) x B1b(4x2) -> C[4:6,0:4], cols 49..62
    off = 49
    for col in range(14):
        for i in range(4):
            for j in range(4):
                if U42[i*4+j][col] != 0:
                    U[i*5+j][off+col] += U42[i*4+j][col]
        for j in range(4):
            for k in range(2):
                if V42[j*2+k][col] != 0:
                    V[j*6 + (4+k)][off+col] += V42[j*2+k][col]
        for k in range(2):
            for i in range(4):
                if W42[k*4+i][col] != 0:
                    W[(4+k)*4 + i][off+col] += W42[k*4+i][col]

    # Part 3: A2(4x1) x B2(1x6) -> C[0:6,0:4], cols 63
    off = 63
    for i in range(4):
        U[i*5+4][off] = 1.0
    for k in range(6):
        V[4*6+k][off] = 1.0
    for k in range(6):
        for i in range(4):
            W[k*4+i][off] = 1.0

    if verify(U, V, W):
        best = (U, V, W, 64)

    return {"U": best[0], "V": best[1], "W": best[2]}
# EVOLVE-BLOCK-END
