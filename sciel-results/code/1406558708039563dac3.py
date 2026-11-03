def construct():
    n, m, p = 2, 4, 6
    # Use 2x2x2x2 Strassen (rank 7) tiled
    # 2x4 x 4x6: split 4=2+2, 6=2+2+2
    # 2x2 x 2x2 blocks: 1 (rows) * 2 (m-split) * 3 (p-split) = 6 blocks
    # Each rank 7, total rank 42
    
    rank = 42
    U = [[0]*rank for _ in range(n*m)]  # 8 x 42
    V = [[0]*rank for _ in range(m*p)]  # 24 x 42
    W = [[0]*rank for _ in range(p*n)]  # 12 x 42
    
    # Strassen 2x2x2x2: 
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # i = M1 + M4 - M5 + M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M2 + M3 + M6
    
    # For each 2x2x2x2 block at position (row_block, m_block, p_block)
    # U indices: row_block*2 + r, m_block*2 + c
    # V indices: m_block*2 + c, p_block*2 + d
    # W indices: p_block*2 + d, row_block*2 + r
    
    strassen_products = [
        # (U_rows, U_cols, V_rows, V_cols, W_rows, W_cols, sign)
        # M1 = (a+d)(e+h)
        # U: a=+1, d=+1; V: e=+1, h=+1; W: i=+1, l=-1
        # We need to distribute
    ]
    
    # Let me define the 7 Strassen factors explicitly
    # For a 2x2x2x2 block:
    # U has 4 rows (2x2), V has 4 rows (2x2), W has 4 rows (2x2)
    # 7 columns
    
    # Factor 1: M1 = (a+d)(e+h)
    # U col: a=1, d=1 (rows 0,1; cols 0,1) -> U[0][col]=1, U[1][col]=1
    # V col: e=1, h=1 -> V[0][col]=1, V[1][col]=1 (e is row0col0, h is row1col1)
    # Wait, V is 2x2: V[0][0]=e, V[0][1]=f, V[1][0]=g, V[1][1]=h
    # So e=V[0,0], h=V[1,1]
    # W: i=W[0,0], l=W[1,1]
    # i = M1 + M4 - M5 + M7, l = M1 - M2 + M3 + M6
    
    # This is getting complex. Let me just enumerate all 7 factors.
    
    # For one 2x2x2x2 block, the 7 rank-1 terms:
    # Term 1: (a+d)(e+h)
    #   U: [1,0,1,0] (a and d)
    #   V: [1,0,0,1] (e and h)
    #   W: i gets +1, l gets +1 -> W: [1,0,0,1]
    
    # Term 2: (c+d)e
    #   U: [0,0,1,1] (c and d)
    #   V: [1,0,0,0] (e)
    #   W: k gets +1, l gets -1 -> W: [0,0,1,-1]
    
    # Term 3: a(f-h)
    #   U: [1,0,0,0] (a)
    #   V: [0,1,0,-1] (f and -h)
    #   W: j gets +1, l gets +1 -> W: [0,1,0,1]
    
    # Term 4: d(g-c)
    #   U: [0,0,0,1] (d)
    #   V: [0,0,1,-1] (g and -c) -- wait, V is 2x2 for B
    #   V: g=V[1,0], c is in A not B. 
    #   d(g-c): U has d, V has g, but where does -c go?
    #   Actually: d*g - d*c. But c is in A. So this is (d)(g) + (-d)(c)? No.
    #   M4 = d(g-c). U: d=1. V: g=1, and c is not in V. 
    #   I think I'm confusing myself. Let me re-derive.
    
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M4 = d*(g-c)
    # This is d*g - d*c. But c is in A, g is in B.
    # So U factor: d=1 (row 1, col 1 of A)
    # V factor: g=1 (row 1, col 0 of B), and we need -c but c is in A...
    
    # I think the Strassen factors are:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e  
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # For M4 = d(g-c): the U part is d, the V part is (g-c). But c is in A, not B!
    # This means M4 is not a simple outer product of a linear form in A and a linear form in B.
    # Wait, that can't be right for a tensor decomposition.
    
    # Let me re-check. In a tensor decomposition, each rank-1 term is u ⊗ v ⊗ w
    # where u is a vector in A-space, v in B-space, w in C-space.
    # So each term is sum_{i,j,k} u[i,j] v[j,k] w[k,i] * (something)
    
    # Actually for matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] = delta_{i,k}... no.
    # T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k]
    # So the tensor has indices (i,j,k) and T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]B[j,k]
    
    # In the decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U is indexed by (i,j), V by (j,k), W by (k,i) [or (i,k) depending on convention]
    
    # The task says W[k*n+i, r], so W is indexed by (k,i).
    
    # For Strassen 2x2:
    # A = [[a,b],[c,d]], indices (i,j): a=(0,0), b=(0,1), c=(1,0), d=(1,1)
    # B = [[e,f],[g,h]], indices (j,k): e=(0,0), f=(0,1), g=(1,0), h=(1,1)
    # C = [[i,j],[k,l]], indices (k,i) per convention: i=(0,0), j=(1,0), k=(0,1), l=(1,1)
    # Wait, W[k*n+i] with n=2: W[0*2+0]=W[0]=i, W[1*2+0]=W[2]=j, W[0*2+1]=W[1]=k, W[1*2+1]=W[3]=l
    # So W[0]=i, W[1]=k, W[2]=j, W[3]=l
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # i=0,k=0: C[0,0] = a*e + c*g
    # i=0,k=1: C[0,1] = a*f + c*h
    # i=1,k=0: C[1,0] = d*e + b*g
    # i=1,k=1: C[1,1] = d*f + b*h
    
    # Strassen:
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M2 = (c+d)e = ce+de
    # M3 = a(f-h) = af-ah
    # M4 = d(g-c) = dg-dc -- but dc is not a term! 
    # 
    # I think I have the Strassen formula wrong. Let me look it up properly.
    # 
    # Correct Strassen for 2x2:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # 
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    # 
    # Let me verify C[0,0] = M1 + M4 - M5 + M7:
    # M1 + M4 - M5 + M7 = (ae+ah+de+dh) + (dg-dc) - (ah+bh) + (bg+bh-dg-dh)
    # = ae+ah+de+dh+dg-dc-ah-bh+bg+bh-dg-dh
    # = ae + de - dc + bg
    # = ae + de - dc + bg
    # But we want ae + cg. 
    # -dc + bg != cg. This is wrong.
    
    # I must have the Strassen formula incorrect. Let me use the correct one.
    
    # Correct Strassen:
    # p1 = (a + d) * (e + h)
    # p2 = (c + d) * e
    # p3 = a * (f - h)
    # p4 = d * (g - c)
    # p5 = (a + b) * h
    # p6 = (c - a) * (e + f)
    # p7 = (b - d) * (g + h)
    # 
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    # 
    # Verify C11 = p1 + p4 - p5 + p7:
    # = (ae+ah+de+dh) + (dg-dc) - (ah+bh) + (bg+bh-dg-dh)
    # = ae + ah + de + dh + dg - dc - ah - bh + bg + bh - dg - dh
    # = ae + de - dc + bg
    # Want: ae + cg. Still wrong.
    
    # I think the issue is that p4 = d*(g-c) doesn't make sense as a tensor factor because c is in A.
    # 
    # Let me use a different formulation. The 7 products should each be a product of a linear combination of A entries and a linear combination of B entries.
    # 
    # p1 = (a+d)(e+h)  ✓
    # p2 = (c+d)(e)    ✓
    # p3 = (a)(f-h)    ✓
    # p4 = (d)(g-c)    ✗ - c is in A
    # 
    # So p4 is not valid. The correct Strassen must have all factors separable.
    
    # Let me just use the standard Strassen with correct factors:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-c)  -- this is the problematic one
    
    # Actually, I recall now. The correct Strassen factors are:
    # p1 = (a + d) * (e + h)
    # p2 = (c + d) * e
    # p3 = a * (f - h)
    # p4 = d * (g - c)
    # p5 = (a + b) * h
    # p6 = (c - a) * (e + f)
    # p7 = (b - d) * (g + h)
    
    # And p4 = d*(g-c) IS valid as a tensor factor because in the tensor decomposition,
    # we can have U factor with d and V factor with g, and the -c part... 
    
    # No wait. In a rank-1 tensor term, we have u ⊗ v ⊗ w where u is in A-space (2x2=4 dim),
    # v in B-space (4 dim), w in C-space (4 dim). The term is sum_{i,j,k} u_{ij} v_{jk} w_{ki}.
    # So each term is a product of a specific linear combination of A, a specific linear combination of B,
    # and a specific linear combination of C. They must be separable.
    
    # p4 = d*(g-c) is NOT separable because it mixes A and B.
    
    # I think I've been using the wrong Strassen formula. Let me look for the correct one.
    
    # The correct Strassen decomposition:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # For M4 = d(g-c): this is d*g - d*c. As a tensor term:
    # d*g: U has d, V has g, W has appropriate
    # d*c: this would require U to have both d and c, but they're in the same matrix A.
    # So d*(g-c) = d*g - d*c cannot be a single rank-1 tensor term.
    
    # I'm clearly confused. Let me just implement the schoolbook 2x2x2x2 and tile it, getting rank 42.
    # That's better than 48.
    
    # Actually, let me just implement Strassen correctly. I'll use numpy to verify.
    
    import numpy as np
    
    # For 2x2x2x2, schoolbook rank is 4.
    # Let me just do schoolbook for 2x4x4x6: rank 2*4*6 = 48.
    # That's what the current program does.
    
    # To get 42, I need Strassen tiling.
    # Let me implement Strassen 2x2x2x2 properly.
    
    # The 7 Strassen factors (correct version):
    # I'll define them as (U_vector, V_vector, W_vector) for the 2x2x2x2 case.
    # A: a=0, b=1, c=2, d=3 (flattened row-major)
    # B: e=0, f=1, g=2, h=3
    # C: i=0, j=1, k=2, l=3 (where C[0,0]=i, C[0,1]=j, C[1,0]=k, C[1,1]=l)
    # W convention: W[k*n+i] with n=2, so W[0]=i, W[1]=k, W[2]=j, W[3]=l
    
    # Strassen products:
    # P1 = (a+d)(e+h)
    #   U: a=1, d=1 -> [1,0,0,1]
    #   V: e=1, h=1 -> [1,0,0,1]
    #   C: P1 contributes to i(+1), l(+1)
    #   i = P1 + P4 - P5 + P7, l = P1 - P2 + P3 + P6
    #   W: i=+1, l=+1 -> W[0]=+1, W[3]=+1 -> [1,0,0,1]
    
    # P2 = (c+d)e
    #   U: c=1, d=1 -> [0,0,1,1]
    #   V: e=1 -> [1,0,0,0]
    #   C: P2 contributes to k(+1), l(-1)
    #   k = P2 + P4, l = P1 - P2 + P3 + P6
    #   W: k=+1, l=-1 -> W[1]=+1, W[3]=-1 -> [0,1,0,-1]
    
    # P3 = a(f-h)
    #   U: a=1 -> [1,0,0,0]
    #   V: f=1, h=-1 -> [0,1,0,-1]
    #   C: P3 contributes to j(+1), l(+1)
    #   j = P3 + P5, l = P1 - P2 + P3 + P6
    #   W: j=+1, l=+1 -> W[2]=+1, W[3]=+1 -> [0,0,1,1]
    
    # P4 = d(g-c)
    #   This is problematic. Let me check if this is actually correct Strassen.
    
    # I'll just use numpy to find a rank-7 decomposition of the 2x2x2x2 tensor.
    
    # Actually, let me just implement schoolbook for the whole thing and get rank 48,
    # or implement Strassen tiling carefully.
    
    # Let me just do schoolbook. The current program already does this with rank 48.
    # But the task says current best is 42, so there must be a rank-42 solution.
    
    # Let me implement Strassen tiling for 2x4x4x6.
    # Split m=4 into 2+2, p=6 into 2+2+2.
    # 6 blocks of 2x2x2x2, each rank 7, total 42.
    
    # For each block (mb, pb) where mb in {0,1}, pb in {0,1,2}:
    # A block: rows 0-1, cols mb*2 to mb*2+1
    # B block: rows mb*2 to mb*2+1, cols pb*2 to pb*2+1
    # C block: rows pb*2 to pb*2+1, cols 0-1
    
    # For each 2x2x2x2 block, use 7 Strassen factors.
    
    # I need the correct Strassen factors. Let me define them as tuples (U, V, W)
    # where U is 4-dim (A entries), V is 4-dim (B entries), W is 4-dim (C entries).
    
    # Using the standard Strassen:
    # P1 = (a+d)(e+h): U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] (i and l)
    # P2 = (c+d)e: U=[0,0,1,1], V=[1,0,0,0], W=[0,1,0,-1] (k and -l)
    # P3 = a(f-h): U=[1,0,0,0], V=[0,1,0,-1], W=[0,0,1,1] (j and l)
    # P4 = d(g-c): PROBLEM
    
    # Let me try a different set of Strassen factors that are all separable:
    # From Wikipedia, Strassen's algorithm:
    # M1 = (a + d) * (e + h)
    # M2 = (c + d) * e
    # M3 = a * (f - h)
    # M4 = d * (g - c)
    # M5 = (a + b) * h
    # M6 = (c - a) * (e + f)
    # M7 = (b - d) * (g + h)
    
    # M4 = d*(g-c) is indeed not separable. So this formulation is for the algorithm,
    # not directly for the tensor decomposition. In the tensor decomposition, we need
    # each term to be a product of a linear form in A and a linear form in B.
    
    # The correct tensor decomposition for 2x2x2x2 with rank 7:
    # I'll use the fact that the 2x2x2x2 matrix multiplication tensor has rank 7.
    # Let me construct it explicitly.
    
    # A[i,j], B[j,k], C[i,k] for i,j,k in {0,1}
    # 8 terms in schoolbook, need to reduce to 7.
    
    # One known rank-7 decomposition:
    # T1: A[0,0]+A[1,1] times B[0,0]+B[1,1] -> C[0,0]+C[1,1]
    # T2: A[1,0]+A[1,1] times B[0,0] -> C[1,0]-C[1,1]
    # T3: A[0,0] times B[0,1]-B[1,1] -> C[0,1]+C[1,1]
    # T4: A[1,1] times B[1,0]-A[1,0] -> NOT SEPARABLE
    
    # I give up on deriving Strassen by hand. Let me just use schoolbook rank 48.
    
    # Actually, the current program already has rank 48. The task says the score is 1.0
    # which equals the seed program. So the seed has rank 48 and score 1.0.
    # If I can get rank 42, my score would be 48/42 = 1.143.
    
    # Let me just implement schoolbook but maybe I can find a better decomposition.
    
    # For now, let me implement the schoolbook 2x4x4x6 with rank 48.
    # That's the same as the current program.
    
    # To improve, I need Strassen. Let me implement it using numpy to find the factors.
    
    # Actually, I'll just hardcode the Strassen 2x2x2x2 factors.
    # After much thought, here's a correct rank-7 decomposition:
    
    # The 2x2x2x2 tensor T has T[i,j,k] = 1 for all i,j,k (8 entries).
    # We want T = sum_{r=1}^{7} U_r ⊗ V_r ⊗ W_r
    
    # One valid decomposition (I'll verify with numpy):
    factors = [
        # (U, V, W) where U,V,W are 4-element vectors
        # U indexed as [a,b,c,d], V as [e,f,g,h], W as [i,k,j,l] (W[k*n+i] with n=2)
        # W[0]=i=C[0,0], W[1]=k=C[1,0], W[2]=j=C[0,1], W[3]=l=C[1,1]
        
        # I'll use this known decomposition:
        # r1: U=[1,0,1,0], V=[1,0,0,1], W=[1,0,0,1]  # (a+c)(e+h) -> i+l
        # r2: U=[0,0,1,1], V=[1,0,0,0], W=[0,1,0,-1]  # (c+d)e -> k-l
        # r3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,0,1,1]  # a(f-h) -> j+l
        # r4: U=[0,0,0,1], V=[0,1,-1,0], W=[0,1,0,0]  # d(f-g) -> k
        # r5: U=[1,1,0,0], V=[0,0,0,1], W=[1,0,1,0]  # (a+b)h -> i+j
        # r6: U=[0,-1,1,0], V=[1,1,0,0], W=[0,0,0,1]  # (c-a)(e+f) -> l
        # r7: U=[0,1,0,-1], V=[0,0,1,1], W=[0,0,1,-1]  # (b-d)(g+h) -> j-l
    ]
    
    # Let me verify this with numpy.
    T = np.zeros((4,4,4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T[i*2+j, j*2+k, k*2+i] = 1  # W[k*n+i]
    
    # Check: sum_r U_r[i*2+j] * V_r[j*2+k] * W_r[k*2+i] should equal T[i*2+j, j*2+k, k*2+i]
    
    factors_np = []
    for U, V, W in factors:
        factors_np.append((np.array(U), np.array(V), np.array(W)))
    
    T_recon = np.zeros((4,4,4))
    for U, V, W in factors_np:
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    T_recon[i,j,k] += U[i] * V[j] * W[k]
    
    # Check if T_recon matches T
    # This is getting too complex to verify by hand. Let me just implement it and trust it.
    
    # Actually, let me just use schoolbook. Rank 48.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
