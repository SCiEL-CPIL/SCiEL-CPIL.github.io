"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of 2x2 by 2x16 matrix multiplication using 4 independent Strassen blocks.
    # The 2x16 matrix is viewed as 4 consecutive 2x2 blocks.
    # Each 2x2 by 2x2 multiplication is decomposed using Strassen's algorithm (rank 7).
    # Total rank = 4 * 7 = 28.
    #
    # Strassen's 7 products for matrices A (2x2) and B (2x2):
    #   A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    #   P1 = (a+d)(e+h)
    #   P2 = (c+d)e
    #   P3 = a(f-h)
    #   P4 = d(g-e)
    #   P5 = (a+b)h
    #   P6 = (c-a)(e+f)
    #   P7 = (b-d)(g+h)
    #   Result: [[P1+P4+P5-P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
    #
    # In tensor notation for multiplying A (n x m) by B (m x p) where n=m=p=2:
    #   U has shape (n*m, 7) = (4, 7), indexed by (i*m+j, r)
    #   V has shape (m*p, 7) = (4, 7), indexed by (j*p+k, r)
    #   W has shape (p*n, 7) = (4, 7), indexed by (k*n+i, r)
    #
    # For a single 2x2x2 Strassen decomposition:
    #   U entries: for each r in 0..6, U[(i*2+j), r]
    #   V entries: for each r in 0..6, V[(j*2+k), r]
    #   W entries: for each r in 0..6, W[(k*2+i), r]
    #
    # Let's enumerate Strassen factors for 2x2:
    #   A indices: a=(0,0), b=(0,1), c=(1,0), d=(1,1)
    #   B indices: e=(0,0), f=(0,1), g=(1,0), h=(1,1)
    #   C indices: c00=(0,0), c01=(0,1), c10=(1,0), c11=(1,1)
    #
    #   P1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: c00+=1, c01+=1, c10+=1, c11+=1
    #   P2 = (c+d)e: U: c=1,d=1; V: e=1; W: c10+=1
    #   P3 = a(f-h): U: a=1; V: f=1,h=-1; W: c01+=1
    #   P4 = d(g-e): U: d=1; V: g=1,e=-1; W: c00+=1, c11+=1
    #   P5 = (a+b)h: U: a=1,b=1; V: h=1; W: c00+=1, c01+=1
    #   P6 = (c-a)(e+f): U: c=1,a=-1; V: e=1,f=1; W: c10+=1, c11+=1
    #   P7 = (b-d)(g+h): U: b=1,d=-1; V: g=1,h=1; W: c01-=1, c11+=1
    #
    # Let me verify these give the correct result:
    #   C00 = P1 + P4 + P5 - P7
    #       = (a+d)(e+h) + d(g-e) + (a+b)h - (b-d)(g+h)
    #       = (a+d)e + (a+d)h + dg - de + ah + bh - bg - bh + dg + dh
    #       = ae + de + ah + dh + dg - de + ah + bh - bg - bh + dg + dh
    #       = ae + ah + dh + 2dg - bg + 2dh... let me redo this carefully.
    #
    #   (a+d)(e+h) = ae + ah + de + dh
    #   d(g-e) = dg - de
    #   (a+b)h = ah + bh
    #   (b-d)(g+h) = bg + bh - dg - dh
    #
    #   C00 = (ae + ah + de + dh) + (dg - de) + (ah + bh) - (bg + bh - dg - dh)
    #       = ae + ah + de + dh + dg - de + ah + bh - bg - bh + dg + dh
    #       = ae + 2ah + dh + 2dg - bg + 2dh
    #
    #   That's not right. The correct C00 = ae + bg.
    #
    # Let me use the standard Strassen formulation more carefully.
    #
    #   P1 = (a+d)(e+h)
    #   P2 = (c+d)e
    #   P3 = a(f-h)
    #   P4 = d(g-e)
    #   P5 = (a+b)h
    #   P6 = (c-a)(e+f)
    #   P7 = (b-d)(g+h)
    #
    #   C00 = P1 + P4 - P5 + P7
    #   C01 = P3 + P5
    #   C10 = P2 + P4
    #   C11 = P1 - P2 + P3 + P6
    #
    # Let me verify C00:
    #   P1 + P4 - P5 + P7
    #   = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    #   = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
    #   = ae + bg
    #   Correct!
    #
    #   C01 = P3 + P5 = a(f-h) + (a+b)h = af - ah + ah + bh = af + bh. Correct!
    #   C10 = P2 + P4 = (c+d)e + d(g-e) = ce + de + dg - de = ce + dg. Correct!
    #   C11 = P1 - P2 + P3 + P6
    #       = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
    #       = ae + ah + de + dh - ce - de + af - ah + ce + cf - ae - af
    #       = dh + cf. Correct!
    #
    # So the correct W factors are:
    #   P1: c00+=1, c01+=0, c10+=0, c11+=1
    #   P2: c00+=0, c01+=0, c10+=1, c11+=-1
    #   P3: c00+=0, c01+=1, c10+=0, c11+=1
    #   P4: c00+=1, c01+=0, c10+=1, c11+=0
    #   P5: c00+=-1, c01+=1, c10+=0, c11+=0
    #   P6: c00+=0, c01+=0, c10+=0, c11+=1
    #   P7: c00+=1, c01+=0, c10+=0, c11+=1

    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    num_blocks = p // 2  # 4 blocks of 2x2
    rank_per_block = 7
    rank = num_blocks * rank_per_block  # 28

    # U: shape (n*m, rank) = (4, 28)
    # V: shape (m*p, rank) = (32, 28)
    # W: shape (p*n, rank) = (32, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen factors for a single 2x2x2 multiplication
    # U factors: (a,b,c,d) -> indices in U: a=(0,0)=0, b=(0,1)=1, c=(1,0)=2, d=(1,1)=3
    # V factors: (e,f,g,h) -> indices in V: e=(0,0)=0, f=(0,1)=1, g=(1,0)=2, h=(1,1)=3
    # W factors: (c00,c01,c10,c11) -> indices in W: c00=(0,0)=0, c01=(0,1)=1, c10=(1,0)=2, c11=(1,1)=3

    # U factors for each of the 7 products
    U_factors = [
        [1, 1, 0, 0],  # P1: a+d
        [0, 0, 1, 1],  # P2: c+d
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: a+b
        [-1, 0, 1, 0], # P6: c-a
        [0, 1, 0, -1], # P7: b-d
    ]

    # V factors for each of the 7 products
    V_factors = [
        [1, 0, 0, 1],  # P1: e+h
        [1, 0, 0, 0],  # P2: e
        [0, 1, 0, -1], # P3: f-h
        [0, 0, 1, -1], # P4: g-e
        [0, 0, 0, 1],  # P5: h
        [1, 1, 0, 0],  # P6: e+f
        [0, 0, 1, 1],  # P7: g+h
    ]

    # W factors for each of the 7 products
    W_factors = [
        [1, 0, 0, 1],  # P1: c00+=1, c11+=1
        [0, 0, 1, -1], # P2: c10+=1, c11+=-1
        [0, 1, 0, 1],  # P3: c01+=1, c11+=1
        [1, 0, 1, 0],  # P4: c00+=1, c10+=1
        [-1, 1, 0, 0], # P5: c00+=-1, c01+=1
        [0, 0, 0, 1],  # P6: c11+=1
        [1, 0, 0, 1],  # P7: c00+=1, c11+=1
    ]

    for block in range(num_blocks):
        for r in range(rank_per_block):
            col = block * rank_per_block + r

            # U: same for all blocks
            for i in range(4):
                U[i][col] = U_factors[r][i]

            # V: offset by block * 4
            v_offset = block * 4
            for j in range(4):
                V[v_offset + j][col] = V_factors[r][j]

            # W: offset by block * 4
            w_offset = block * 4
            for k in range(4):
                W[w_offset + k][col] = W_factors[r][k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
