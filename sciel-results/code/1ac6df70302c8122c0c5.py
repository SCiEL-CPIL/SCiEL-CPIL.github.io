import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W with rank 53
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Process 7 pairs of columns
    for pair_idx in range(7):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        
        # Strassen terms for this pair
        # P1 = (A00+A11)(B0k1+B0k2) -- Wait, indices
        # A is 2x2. A[i,j].
        # B is 2x2 for this pair. B[j, col_in_pair].
        # col_in_pair 0 -> k1, 1 -> k2.
        
        # Term 1: (A00+A11) * (B0k1+B0k2) is not quite right.
        # Strassen P1 = (A11+A22)(B11+B22) in 1-based.
        # 0-based: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])?
        # No, B11 is B[0,0], B22 is B[1,1].
        # So P1 uses A[0,0], A[1,1] and B[0,k1], B[1,k1]? No.
        # B matrix for the pair:
        # [ B[0,k1]  B[0,k2] ]
        # [ B[1,k1]  B[1,k2] ]
        # So B11=B[0,k1], B12=B[0,k2], B21=B[1,k1], B22=B[1,k2].
        
        # P1 = (A[0,0]+A[1,1]) * (B[0,k1]+B[1,k1])? No.
        # P1 = (A[0,0]+A[1,1]) * (B[0,k1]+B[1,k1]) is wrong.
        # P1 = (A[0,0]+A[1,1]) * (B[0,k1]+B[1,k1])?
        # Standard Strassen:
        # P1 = (A11+A22)(B11+B22)
        # A11=A[0,0], A22=A[1,1].
        # B11=B[0,k1], B22=B[1,k2].
        # So P1 = (A[0,0]+A[1,1]) * (B[0,k1]+B[1,k2])?
        # No, B11 is top-left, B22 is bottom-right.
        # B11 = B[0,k1], B22 = B[1,k2].
        # So P1 = (A[0,0]+A[1,1]) * (B[0,k1]+B[1,k2])?
        # Wait, B11+B22 = B[0,k1] + B[1,k2].
        # This mixes k1 and k2 in the B factor.
        # This is correct for Strassen.
        
        # Let's define the 7 terms explicitly.
        # Each term is a product of a linear form in A and a linear form in B.
        # We add one column to U, V, W for each term.
        
        terms = []
        
        # P1
        # A: A[0,0] + A[1,1]
        # B: B[0,k1] + B[1,k2]
        terms.append((
            {(0,0): 1, (1,1): 1}, # U coeffs
            {(0,k1): 1, (1,k2): 1}, # V coeffs
            {(k1,0): 1, (k2,0): 1, (k1,1): -1, (k2,1): 1} # W coeffs for C0k1, C0k2, C1k1, C1k2
        ))
        # Wait, W coeffs depend on how P1 contributes to C.
        # C[0,k1] = P1 + P4 - P5 + P7
        # C[0,k2] = P6 + P1
        # C[1,k1] = P3 + P5
        # C[1,k2] = P1 - P2 + P3 + P4
        
        # We need to add contributions to W.
        # For term P1:
        # W[k1*n+0] += 1
        # W[k2*n+0] += 1
        # W[k1*n+1] += 0
        # W[k2*n+1] += 1
        
        # Let's do this systematically.
        
        # Term 1: P1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,k1): 1, (1,k2): 1}
        w1 = {(k1,0): 1, (k2,0): 1, (k2,1): 1}
        
        # Term 2: P2 = (A[1,0]+A[1,1]) * B[0,k2]
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,k2): 1}
        w2 = {(k2,1): -1} # C[1,k2] -= P2
        
        # Term 3: P3 = A[0,0] * (B[1,k1] - B[0,k1])
        u3 = {(0,0): 1}
        v3 = {(1,k1): 1, (0,k1): -1}
        w3 = {(k1,1): 1, (k2,1): 1} # C[1,k1] += P3, C[1,k2] += P3
        
        # Term 4: P4 = (A[0,0]-A[1,1]) * (B[1,k1] + B[1,k2])
        u4 = {(0,0): 1, (1,1): -1}
        v4 = {(1,k1): 1, (1,k2): 1}
        w4 = {(k1,0): 1, (k2,1): 1} # C[0,k1] += P4, C[1,k2] += P4
        
        # Term 5: P5 = (A[0,0]+A[0,1]) * B[1,k2]
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,k2): 1}
        w5 = {(k1,0): -1, (k1,1): 1} # C[0,k1] -= P5, C[1,k1] += P5
        
        # Term 6: P6 = (A[1,0]-A[0,0]) * (B[0,k1] + B[0,k2])
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,k1): 1, (0,k2): 1}
        w6 = {(k2,0): 1} # C[0,k2] += P6
        
        # Term 7: P7 = (A[0,1]-A[1,1]) * (B[1,k1] + B[1,k2])
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,k1): 1, (1,k2): 1}
        w7 = {(k1,0): 1} # C[0,k1] += P7
        
        all_terms = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
        
        for u, v, w in all_terms:
            for (i,j), val in u.items():
                U[i*m+j][col] += val
            for (j,k), val in v.items():
                V[j*p+k][col] += val
            for (k,i), val in w.items():
                W[k*n+i][col] += val
            col += 1
            
    # Process last column k=14
    k = 14
    # Schoolbook for 2x2x1
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k]
    
    # Term 1: A[0,0] B[0,k]
    U[0*m+0][col] = 1
    V[0*p+k][col] = 1
    W[k*n+0][col] = 1
    col += 1
    
    # Term 2: A[0,1] B[1,k]
    U[0*m+1][col] = 1
    V[1*p+k][col] = 1
    W[k*n+0][col] = 1
    col += 1
    
    # Term 3: A[1,0] B[0,k]
    U[1*m+0][col] = 1
    V[0*p+k][col] = 1
    W[k*n+1][col] = 1
    col += 1
    
    # Term 4: A[1,1] B[1,k]
    U[1*m+1][col] = 1
    V[1*p+k][col] = 1
    W[k*n+1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
