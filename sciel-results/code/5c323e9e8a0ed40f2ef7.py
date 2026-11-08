import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a 2+3 block decomposition.
    # Block sizes: 2 and 3.
    # Total rank = Rank(2x2) + Rank(3x3) + Cross Terms
    
    # Strassen decomposition for 2x2 (Rank 7)
    # Indices for 2x2: 0,1,2,3
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # We define the rank-1 terms for 2x2 Strassen.
    # Each term is (u, v, w) where u, v, w are vectors of length 4.
    
    # M1 = (A0+A3)(B0+B3)
    # M2 = (A2+A3)B0
    # M3 = A0(B1-B3)
    # M4 = A3(B2-B0)
    # M5 = (A0+A1)B3
    # M6 = (A2-A0)(B0+B1)
    # M7 = (A1-A3)(B2+B3)
    
    # C0 = M1 + M4 - M5 + M7
    # C1 = M3 + M5
    # C2 = M2 + M4
    # C3 = M1 - M2 + M3 + M6
    
    # We construct the U, V, W matrices for the 2x2 block.
    # We will accumulate terms into lists.
    
    terms = []
    
    # Helper to add a term
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    # 2x2 Strassen Terms
    # M1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    # M1 contributes to C0 (+1), C3 (+1)
    # We need to split this into rank-1 tensors for the output.
    # Actually, the tensor decomposition is a sum of rank-1 tensors.
    # Each multiplication M_k is a product of two linear forms.
    # The output C is a linear combination of M_k.
    # So we can just add the terms (u_k, v_k, w_c) for each output c.
    
    # M1: u=[1,0,0,1], v=[1,0,0,1]
    # C0 += M1 -> w=[1,0,0,0]
    add_term(u1, v1, [1,0,0,0])
    # C3 += M1 -> w=[0,0,0,1]
    add_term(u1, v1, [0,0,0,1])
    
    # M2: u=[0,0,1,1], v=[1,0,0,0]
    # C2 += M2 -> w=[0,0,1,0]
    add_term(u1, [1,0,0,0], [0,0,1,0]) # Wait, u for M2 is [0,0,1,1]
    # Let's be precise.
    
    # Clear terms and restart with precise definitions.
    terms = []
    
    # M1
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    add_term(u, v, [1, 0, 0, 0]) # C0
    add_term(u, v, [0, 0, 0, 1]) # C3
    
    # M2
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    add_term(u, v, [0, 0, 1, 0]) # C2
    
    # M3
    u = [1, 0, 0, 0]
    v = [0, 1, 0, -1]
    add_term(u, v, [0, 1, 0, 0]) # C1
    add_term(u, v, [0, 0, 0, 1]) # C3
    
    # M4
    u = [0, 0, 0, 1]
    v = [0, 0, 1, -1]
    add_term(u, v, [1, 0, 0, 0]) # C0
    add_term(u, v, [0, 0, 1, 0]) # C2
    
    # M5
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    add_term(u, v, [-1, 0, 0, 0]) # C0
    add_term(u, v, [0, 1, 0, 0]) # C1
    
    # M6
    u = [0, 0, 1, -1] # A2 - A0
    v = [1, 1, 0, 0] # B0 + B1
    add_term(u, v, [0, 0, 0, 1]) # C3
    
    # M7
    u = [0, 1, 0, -1] # A1 - A3
    v = [0, 0, 1, 1] # B2 + B3
    add_term(u, v, [1, 0, 0, 0]) # C0
    
    # 3x3 Schoolbook Terms (Rank 27)
    # We will add 27 terms for the 3x3 block.
    # Indices for 3x3: 0..8
    # A: i*3+j
    # B: k*3+l
    # C: i*3+l
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0]*9
                v = [0]*9
                w = [0]*9
                u[i*3+j] = 1
                v[k*3+l] = 1 # l is the output col index? No.
                # In schoolbook, C[i,l] += A[i,j] * B[j,l]
                # So j is the inner index.
                # A index: i*3+j
                # B index: j*3+l
                # C index: i*3+l
                
                # Wait, my loop is i, j, k. Let's use i, j, l.
                # C[i,l] = sum_j A[i,j] B[j,l]
                pass

    # Correct 3x3 Schoolbook
    for i in range(3):
        for j in range(3):
            for l in range(3):
                u = [0]*9
                v = [0]*9
                w = [0]*9
                u[i*3+j] = 1
                v[j*3+l] = 1
                w[i*3+l] = 1
                add_term(u, v, w)
                
    # Now we have terms for 2x2 (7*4=28 terms? No, 7 multiplications, each split into outputs)
    # And terms for 3x3 (27 terms).
    
    # We need to embed these into the 5x5 space.
    # 2x2 block is at rows 0-1, cols 0-1.
    # 3x3 block is at rows 2-4, cols 2-4.
    
    # Cross terms:
    # A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2] (2x3 * 3x2)
    # A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5] (2x3 * 3x3)
    # A[2:5, 0:2] * B[0:2, 0:2] -> C[2:5, 0:2] (3x2 * 2x2)
    # A[2:5, 0:2] * B[0:2, 2:5] -> C[2:5, 2:5] (3x2 * 2x3)
    # A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5] (2x2 * 2x3)
    # A[0:2, 0:2] * B[2:5, 0:2] -> C[0:2, 0:2] (2x2 * 3x2)
    # A[2:5, 2:5] * B[2:5, 0:2] -> C[2:5, 0:2] (3x3 * 3x2)
    # A[2:5, 2:5] * B[0:2, 2:5] -> C[2:5, 2:5] (3x3 * 2x3)
    
    # We will add these cross terms using rank-1 tensors.
    # For A[i,j] * B[k,l] -> C[i,l]
    # u[i*5+j] = 1
    # v[k*5+l] = 1
    # w[i*5+l] = 1
    
    # Cross term 1: A[0:2, 2:5] * B[2:5, 0:2]
    for i in range(2):
        for j in range(3): # j is 0..2 relative to 2..4
            for k in range(3): # k is 0..2 relative to 2..4
                for l in range(2):
                    # A index: i*5 + (2+j)
                    # B index: (2+k)*5 + l
                    # C index: i*5 + l
                    u = [0]*25
                    v = [0]*25
                    w = [0]*25
                    u[i*5 + 2 + j] = 1
                    v[(2 + k)*5 + l] = 1
                    w[i*5 + l] = 1
                    add_term(u, v, w)
                    
    # Cross term 2: A[0:2, 2:5] * B[2:5, 2:5]
    for i in range(2):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    u = [0]*25
                    v = [0]*25
                    w = [0]*25
                    u[i*5 + 2 + j] = 1
                    v[(2 + k)*5 + (2 + l)] = 1
                    w[i*5 + (2 + l)] = 1
                    add_term(u, v, w)

    # Cross term 3: A[2:5, 0:2] * B[0:2, 0:2]
    for i in range(3):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    u = [0]*25
                    v = [0]*25
                    w = [0]*25
                    u[(2 + i)*5 + j] = 1
                    v[k*5 + l] = 1
                    w[(2 + i)*5 + l] = 1
                    add_term(u, v, w)

    # Cross term 4: A[2:5, 0:2] * B[0:2, 2:5]
    for i in range(3):
        for j in range(2):
            for k in range(2):
                for l in range(3):
                    u = [0]*25
                    v = [0]*25
                    w = [0]*25
                    u[(2 + i)*5 + j] = 1
                    v[k*5 + (2 + l)] = 1
                    w[(2 + i)*5 + (2 + l)] = 1
                    add_term(u, v, w)

    # Cross term 5: A[0:2, 0:2] * B[0:2, 2:5]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(3):
                    u = [0]*25
                    v = [0]*25
                    w = [0]*25
                    u[i*5 + j] = 1
                    v[k*5 + (2 + l)] = 1
                    w[i*5 + (2 + l)] = 1
                    add_term(u, v, w)

    # Cross term 6: A[0:2, 0:2] * B[2:5, 0:2]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                for l in range(2):
                    u = [0]*25
                    v = [0]*25
                    w = [0]*25
                    u[i*5 + j] = 1
                    v[(2 + k)*5 + l] = 1
                    w[i*5 + l] = 1
                    add_term(u, v, w)

    # Cross term 7: A[2:5, 2:5] * B[2:5, 0:2]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(2):
                    u = [0]*25
                    v = [0]*25
                    w = [0]*25
                    u[(2 + i)*5 + (2 + j)] = 1
                    v[(2 + k)*5 + l] = 1
                    w[(2 + i)*5 + l] = 1
                    add_term(u, v, w)

    # Cross term 8: A[2:5, 2:5] * B[0:2, 2:5]
    for i in range(3):
        for j in range(3):
            for k in range(2):
                for l in range(3):
                    u = [0]*25
                    v = [0]*25
                    w = [0]*25
                    u[(2 + i)*5 + (2 + j)] = 1
                    v[k*5 + (2 + l)] = 1
                    w[(2 + i)*5 + (2 + l)] = 1
                    add_term(u, v, w)

    # Now we have all terms in the global 25-dim space.
    # We need to combine them into U, V, W matrices.
    
    rank = len(terms)
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    
    for col, (u, v, w) in enumerate(terms):
        for i in range(25):
            U[i][col] = u[i]
            V[i][col] = v[i]
            W[i][col] = w[i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
