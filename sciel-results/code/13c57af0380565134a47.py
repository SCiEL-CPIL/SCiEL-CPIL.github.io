import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create Strassen 2x2 factors
    # Returns U, V, W of shape (4, 7)
    def strassen_2x2():
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # Indices: A: 0=a, 1=b, 2=c, 3=d
        #          B: 0=e, 1=f, 2=g, 3=h
        #          C: 0=c11, 1=c12, 2=c21, 3=c22
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # p1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        W[1, 0] = 1; W[2, 0] = 1 # c11, c22? No, W maps to C indices.
        # C indices: 0=c11, 1=c12, 2=c21, 3=c22
        # p1 contributes to c11 and c22.
        # c11 = p1 + p4 - p5 + p7
        # c12 = p3 + p5
        # c21 = p2 + p4
        # c22 = p1 - p2 + p3 + p6
        
        # Let's define W contributions carefully.
        # W[i, k] is the coefficient for C_i in term k.
        
        # Term 0: p1
        W[0, 0] = 1 # c11
        W[3, 0] = 1 # c22
        
        # Term 1: p2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # c21
        
        # Term 2: p3 = (a-b)(f-h)
        U[0, 2] = 1; U[1, 2] = -1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # c12
        W[3, 2] = 1 # c22
        
        # Term 3: p4 = d(g-h)
        U[3, 3] = 1
        V[2, 3] = 1; V[3, 3] = -1
        W[0, 3] = 1 # c11
        W[2, 3] = 1 # c21
        
        # Term 4: p5 = a(e+g)
        U[0, 4] = 1
        V[0, 4] = 1; V[2, 4] = 1
        W[0, 4] = -1 # c11
        W[1, 4] = 1 # c12
        
        # Term 5: p6 = b(f+g) ? No, p6 = (b)(f+g) is not standard.
        # Standard p6 = (c)(h-g)? No.
        # Let's check c22 = p1 - p2 + p3 + p6.
        # We need p6 to provide the remaining terms for c22.
        # c22 = dh + bf.
        # Current sum for c22: p1 (dh + ... ) - p2 + p3.
        # p1 = ae + ah + de + dh
        # -p2 = -ce - de
        # p3 = af - ah - bf + bh
        # Sum = ae + dh - ce + af - bf + bh
        # Target = dh + bf
        # Missing: -ae + ce - af + 2bf - bh?
        # This implies my p6 definition is wrong or the combination is wrong.
        
        # Correct Strassen:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)(e)
        # p3 = (a-b)(f-h)
        # p4 = d(g-h)
        # p5 = a(e+g)
        # p6 = (c)(h-g) ? No.
        # p6 = (b)(f+g) ?
        # p7 = (a+c)(e-f)
        
        # Let's use a known correct set.
        # p1 = (a+d)(e+h)
        # p2 = (c+d)(e)
        # p3 = (a-b)(f-h)
        # p4 = d(g-h)
        # p5 = a(e+g)
        # p6 = (c)(h-g)
        # p7 = (a+c)(e-f)
        
        # c11 = p1 + p4 - p5 + p7
        # c12 = p3 + p5
        # c21 = p2 + p4
        # c22 = p1 - p2 + p3 + p6
        
        # Let's verify c22 with p6 = c(h-g)
        # Sum = ae + ah + de + dh - ce - de + af - ah - bf + bh + ch - cg
        # = ae + dh - ce + af - bf + bh + ch - cg
        # Target = dh + bf
        # Still doesn't match.
        
        # Actually, p6 is usually (b)(f+g) in some variants?
        # Let's just use the schoolbook for 2x2 if Strassen is too hard to verify.
        # Schoolbook 2x2 rank 4.
        # 4x4 rank 16.
        # 4x4x9 rank 16+16+16 = 48?
        # No, 4x4 schoolbook is 16.
        # 4x4x9 = 16 + 16 + 16 = 48.
        # Score 144/48 = 3.0.
        # This is better than 1.0 and easier to implement correctly.
        
        # Let's use Schoolbook 2x2 (rank 4).
        # A: a,b,c,d. B: e,f,g,h.
        # c11 = ae + bg
        # c12 = af + bh
        # c21 = ce + dg
        # c22 = cf + dh
        
        U = np.zeros((4, 4), dtype=int)
        V = np.zeros((4, 4), dtype=int)
        W = np.zeros((4, 4), dtype=int)
        
        # Term 0: ae
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1 # c11
        
        # Term 1: bg
        U[1, 1] = 1
        V[2, 1] = 1
        W[0, 1] = 1 # c11
        
        # Term 2: af
        U[0, 2] = 1
        V[1, 2] = 1
        W[1, 2] = 1 # c12
        
        # Term 3: bh
        U[1, 3] = 1
        V[3, 3] = 1
        W[1, 3] = 1 # c12
        
        # Term 4: ce
        U[2, 4] = 1
        V[0, 4] = 1
        W[2, 4] = 1 # c21
        
        # Term 5: dg
        U[3, 5] = 1
        V[2, 5] = 1
        W[2, 5] = 1 # c21
        
        # Term 6: cf
        U[2, 6] = 1
        V[1, 6] = 1
        W[3, 6] = 1 # c22
        
        # Term 7: dh
        U[3, 7] = 1
        V[3, 7] = 1
        W[3, 7] = 1 # c22
        
        return U, V, W

    # Get 2x2 factors
    U2, V2, W2 = strassen_2x2()
    rank_2 = U2.shape[1] # 4
    
    # Construct 4x4 factors using 2x2 blocks
    # A is 4x4, flattened 16.
    # Blocks: A11 (0-3), A12 (4-7), A21 (8-11), A22 (12-15)
    # B is 4x4, flattened 16.
    # Blocks: B11 (0-3), B12 (4-7), B21 (8-11), B22 (12-15)
    # C is 4x4, flattened 16.
    # Blocks: C11 (0-3), C12 (4-7), C21 (8-11), C22 (12-15)
    
    # We need 4 block multiplications? No, 4x4 multiplication is:
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # Each block multiplication is a 2x2 multiplication.
    # So we need 4 separate 2x2 multiplications?
    # No, the inputs to the 2x2 multiplications are sums of blocks.
    # But if we use schoolbook for the blocks, we can just treat the 4x4 as 16x16?
    # No, we want to use the 2x2 structure to reduce rank.
    # If we use schoolbook 2x2 (rank 4), then 4x4 rank is 16.
    # This is the same as schoolbook 4x4 (rank 16).
    # So it doesn't help.
    
    # We MUST use Strassen (rank 7) to get 4x4 rank 49.
    # Let's try to fix the Strassen factors.
    
    # Correct Strassen 2x2:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e)
    # p3 = (a-b)(f-h)
    # p4 = d(g-h)
    # p5 = a(e+g)
    # p6 = (c)(h-g)
    # p7 = (a+c)(e-f)
    
    # c11 = p1 + p4 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    
    # Let's re-verify c22.
    # p1 = ae + ah + de + dh
    # -p2 = -ce - de
    # p3 = af - ah - bf + bh
    # p6 = ch - cg
    # Sum = ae + ah + de + dh - ce - de + af - ah - bf + bh + ch - cg
    # = ae + dh - ce + af - bf + bh + ch - cg
    # Target = dh + bf
    # Difference = ae - ce + af - 2bf + bh + ch - cg
    # This is not zero.
    
    # The correct p6 is (b)(f+g)?
    # If p6 = bf + bg
    # Sum = ae + dh - ce + af - bf + bh + bf + bg
    # = ae + dh - ce + af + bh + bg
    # Still not dh + bf.
    
    # I will use the known fact that R(2x2) = 7.
    # I will hardcode the 7 triples from a reliable source.
    
    # Triples for 2x2:
    # (1,1,1,1, 1,1,1,1, 1,0,0,1) -> (a+d)(e+h) -> c11, c22
    # (0,0,1,1, 1,0,0,0, 0,0,1,0) -> (c+d)e -> c21
    # (1,-1,0,0, 0,1,0,-1, 0,1,0,1) -> (a-b)(f-h) -> c12, c22
    # (0,0,0,1, 0,0,1,-1, 1,0,0,0) -> d(g-h) -> c11, c21
    # (1,0,0,0, 1,0,1,0, -1,1,0,0) -> a(e+g) -> c11, c12
    # (0,1,0,0, 0,1,1,0, 0,0,0,-1) -> b(f+g) ? No.
    # (0,0,1,0, 0,0,0,1, 0,0,0,1) -> c(h) ?
    
    # This is too error-prone.
    
    # Fallback: Use Schoolbook for 4x4 (rank 16) and 4x4x1 (rank 16).
    # Total rank 48.
    # Score 144/48 = 3.0.
    # This is a solid improvement.
    
    rank_4x4 = 16
    rank_4x1 = 16
    total_rank = rank_4x4 + rank_4x4 + rank_4x1
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Block 1: B[:, 0:4]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 4 + j][col] = 1
                V[j * 9 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
                
    # Block 2: B[:, 4:8]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 4 + j][col] = 1
                V[j * 9 + (k + 4)][col] = 1
                W[(k + 4) * 4 + i][col] = 1
                col += 1
                
    # Block 3: B[:, 8:9]
    for i in range(4):
        for j in range(4):
            for k in range(1):
                U[i * 4 + j][col] = 1
                V[j * 9 + (k + 8)][col] = 1
                W[(k + 8) * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
