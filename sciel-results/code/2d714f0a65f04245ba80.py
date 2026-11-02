"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x2x6 matrix multiplication.
    # This decomposes the problem into three 2x2x2 blocks, each of rank 7.
    # Total rank = 3 * 7 = 21.
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors definitions
    # Each factor is defined by coefficients in U, V, and W
    # U coeffs: (i,j) -> val
    # V coeffs: (j,k) -> val
    # W coeffs: (k,i) -> val
    
    # Predefined Strassen terms for 2x2x2
    # t: 0..6
    # u_coeffs: list of (idx, val) where idx = 2*i + j
    # v_coeffs: list of (idx, val) where idx = 2*j + k
    # w_coeffs: list of (idx, val) where idx = 2*k + i
    
    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        {
            'u': [(0, 1), (3, 1)], 
            'v': [(0, 1), (3, 1)], 
            'w': [(0, 1), (3, 1)] # C00+=1, C11+=1 -> W[0,0]=1, W[1,1]=1 (idx 0 and 3)
        },
        # M2: (A10+A11)B00
        {
            'u': [(2, 1), (3, 1)], 
            'v': [(0, 1)], 
            'w': [(2, 1), (3, -1)] # C10+=1, C11-=1 -> W[0,1]=1 (idx 2), W[1,1]=-1 (idx 3)
        },
        # M3: A00(B01-B11)
        {
            'u': [(0, 1)], 
            'v': [(1, 1), (3, -1)], 
            'w': [(1, 1), (3, 1)] # C01+=1, C11+=1 -> W[1,0]=1 (idx 1), W[1,1]=1 (idx 3)
        },
        # M4: A11(B10-B00)
        {
            'u': [(3, 1)], 
            'v': [(2, 1), (0, -1)], 
            'w': [(0, 1), (2, 1)] # C00+=1, C10+=1 -> W[0,0]=1 (idx 0), W[0,1]=1 (idx 2)
        },
        # M5: (A00+A01)B11
        {
            'u': [(0, 1), (1, 1)], 
            'v': [(3, 1)], 
            'w': [(0, -1), (1, 1)] # C00-=1, C01+=1 -> W[0,0]=-1 (idx 0), W[1,0]=1 (idx 1)
        },
        # M6: (A10-A00)(B00+B01)
        {
            'u': [(2, 1), (0, -1)], 
            'v': [(0, 1), (1, 1)], 
            'w': [(3, 1)] # C11+=1 -> W[1,1]=1 (idx 3)
        },
        # M7: (A11-A01)(B10+B11)
        {
            'u': [(3, 1), (1, -1)], 
            'v': [(2, 1), (3, 1)], 
            'w': [(0, 1)] # C00+=1 -> W[0,0]=1 (idx 0)
        },
    ]
    
    for s in range(3):
        for t in range(7):
            col = s * 7 + t
            term = strassen_terms[t]
            
            # U: idx = 2*i + j. No shift for i,j.
            for idx, val in term['u']:
                U[idx][col] += val
                
            # V: idx = 2*j + k. k is shifted by 2*s.
            for idx, val in term['v']:
                j = idx // 2
                k = idx % 2
                global_idx = 2 * j + (2 * s + k)
                V[global_idx][col] += val
                
            # W: idx = 2*k + i. k is shifted by 2*s.
            for idx, val in term['w']:
                k = idx // 2
                i = idx % 2
                global_idx = 2 * (2 * s + k) + i
                W[global_idx][col] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
