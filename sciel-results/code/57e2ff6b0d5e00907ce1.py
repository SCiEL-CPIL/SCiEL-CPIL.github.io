"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    # U, V, W for 2x2x2
    U_2 = [
        [1, 0, 1, 0, 1, 0, 1], # a, d, a, b, c, d, b-d? No.
        # Let's use the standard Strassen factors
    ]
    
    # Standard Strassen 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+g)
    
    # U (4x7): rows (00, 01, 10, 11)
    U_s = [
        [1, 0, 1, 0, 1, 0, 0], # a
        [0, 0, 0, 0, 1, 0, 1], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 1, 0, 1, 0, -1, -1] # d
    ]
    
    # V (4x7): rows (00, 01, 10, 11)
    V_s = [
        [1, 1, 0, -1, 0, 0, 1], # e
        [0, 0, 1, 0, 1, 1, 0], # f
        [0, 0, 0, 1, 0, 1, 1], # g
        [1, 0, -1, 0, 1, 0, 0] # h
    ]
    
    # W (4x7): rows (00, 10, 01, 11) -> index k*2+i
    # C00 = M1+M4-M5+M7
    # C10 = M2+M4
    # C01 = M3+M5
    # C11 = M1-M2+M3+M6
    W_s = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 1, 0, 1, 0, 0, 0], # C10
        [0, 0, 1, 0, 1, 0, 0], # C01
        [1, -1, 1, 0, 0, 1, 0] # C11
    ]
    
    # Kronecker product for 4x4x4
    # U_4 = U_s \otimes U_s
    # V_4 = V_s \otimes V_s
    # W_4 = W_s \otimes W_s
    
    def kron(A, B):
        r1, c1 = len(A), len(A[0])
        r2, c2 = len(B), len(B[0])
        res = [[0] * (c1 * c2) for _ in range(r1 * r2)]
        for i in range(r1):
            for j in range(c1):
                for k in range(r2):
                    for l in range(c2):
                        res[i * r2 + k][j * c2 + l] = A[i][j] * B[k][l]
        return res

    U_4 = kron(U_s, U_s)
    V_4 = kron(V_s, V_s)
    W_4 = kron(W_s, W_s)
    
    # Rank 49
    rank_444 = 49
    
    # Add rank 1 for 4x4x1 (5th column)
    # U_extra: 16x1
    U_extra = [[1] for _ in range(16)]
    # V_extra: 20x1 (rows j*5+k, k=4)
    # Indices: j in 0..3, k=4. index = j*5+4.
    V_extra = [[0] * 20 for _ in range(20)] # No, V is (m*p) x rank
    # V shape: (4*5) x 1 = 20 x 1
    V_extra = [[0] for _ in range(20)]
    for j in range(4):
        idx = j * 5 + 4
        V_extra[idx][0] = 1
        
    # W_extra: 20x1 (rows k*4+i, k=4)
    # Indices: k=4, i in 0..3. index = 4*4+i = 16+i.
    W_extra = [[0] for _ in range(20)]
    for i in range(4):
        idx = 4 * 4 + i
        W_extra[idx][0] = 1
        
    # Combine
    U_final = [row + [U_extra[i][0]] for i, row in enumerate(U_4)]
    V_final = [row + [V_extra[i][0]] for i, row in enumerate(V_4)]
    W_final = [row + [W_extra[i][0]] for i, row in enumerate(W_4)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
