import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * (7 + 4) = 44
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Indices for U: i * m + j
    # Indices for V: j * p + k
    # Indices for W: k * n + i
    
    for b in range(4):
        base_rank = b * 11
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # W indices for this block
        # C[i, k0] -> k0 * n + i
        # C[i, k1] -> k1 * n + i
        
        # Strassen Part (2x2 by 2x2)
        # A_L: i in {0,1}, j in {0,1}
        # B_T: j in {0,1}, k in {k0, k1}
        
        # 1. P1 = a(e+h)
        r = base_rank + 0
        U[0, r] = 1 # a (0,0)
        V[k0, r] = 1 # e (0, k0)
        V[p + k1, r] = 1 # h (1, k1) -> index 1*8 + k1
        W[k0 * n + 0, r] = 1 # C[0, k0]
        W[k0 * n + 1, r] = 1 # C[1, k0]
        W[k1 * n + 1, r] = -1 # C[1, k1]
        
        # 2. P2 = (a+b)h
        r = base_rank + 1
        U[0, r] = 1 # a
        U[1, r] = 1 # b (0,1)
        V[p + k1, r] = 1 # h
        W[k0 * n + 0, r] = -1 # C[0, k0]
        W[k1 * n + 1, r] = 1 # C[1, k1]
        
        # 3. P3 = (c+d)e
        r = base_rank + 2
        U[m + 0, r] = 1 # c (1,0) -> index 1*3+0=3
        U[m + 1, r] = 1 # d (1,1) -> index 1*3+1=4
        V[k0, r] = 1 # e
        W[k1 * n + 0, r] = 1 # C[0, k1]
        W[k1 * n + 1, r] = 1 # C[1, k1]
        
        # 4. P4 = d(g+h)
        r = base_rank + 3
        U[m + 1, r] = 1 # d
        V[p + k0, r] = 1 # g (1, k0)
        V[p + k1, r] = 1 # h
        W[k0 * n + 0, r] = 1 # C[0, k0]
        W[k0 * n + 1, r] = 1 # C[1, k0]
        W[k1 * n + 1, r] = 1 # C[1, k1]
        
        # 5. P5 = (a+c)(e+f)
        r = base_rank + 4
        U[0, r] = 1 # a
        U[m + 0, r] = 1 # c
        V[k0, r] = 1 # e
        V[k1, r] = 1 # f (0, k1)
        W[k0 * n + 1, r] = 1 # C[1, k0]
        W[k1 * n + 1, r] = 1 # C[1, k1]
        
        # 6. P6 = (b-d)(g+h)
        r = base_rank + 5
        U[1, r] = 1 # b
        U[m + 1, r] = -1 # d
        V[p + k0, r] = 1 # g
        V[p + k1, r] = 1 # h
        W[k0 * n + 0, r] = 1 # C[0, k0]
        
        # 7. P7 = (c-b)(e+f)
        r = base_rank + 6
        U[m + 0, r] = 1 # c
        U[1, r] = -1 # b
        V[k0, r] = 1 # e
        V[k1, r] = 1 # f
        W[k1 * n + 0, r] = 1 # C[0, k1]
        
        # Schoolbook Part (2x1 by 1x2)
        # A_R: i in {0,1}, j=2
        # B_B: j=2, k in {k0, k1}
        
        # 8. a02 * b2_k0
        r = base_rank + 7
        U[2, r] = 1 # a02 (0,2)
        V[2 * p + k0, r] = 1 # b2_k0 (2, k0)
        W[k0 * n + 0, r] = 1 # C[0, k0]
        
        # 9. a02 * b2_k1
        r = base_rank + 8
        U[2, r] = 1
        V[2 * p + k1, r] = 1
        W[k1 * n + 0, r] = 1 # C[0, k1]
        
        # 10. a12 * b2_k0
        r = base_rank + 9
        U[m + 2, r] = 1 # a12 (1,2) -> index 1*3+2=5
        V[2 * p + k0, r] = 1
        W[k0 * n + 1, r] = 1 # C[1, k0]
        
        # 11. a12 * b2_k1
        r = base_rank + 10
        U[m + 2, r] = 1
        V[2 * p + k1, r] = 1
        W[k1 * n + 1, r] = 1 # C[1, k1]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
