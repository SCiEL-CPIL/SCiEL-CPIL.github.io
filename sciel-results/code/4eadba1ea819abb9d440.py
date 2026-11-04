"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x3x10 matrix multiplication.
    # Uses 2x2x2 Strassen (rank 7) for pairs of k-columns, and a rank-1 term for j=2.
    # Total rank = 5 * 7 + 5 * 1 = 40.
    n, m, p = DIMENSIONS
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 6 x 40
    V = [[0] * rank for _ in range(m * p)]  # 30 x 40
    W = [[0] * rank for _ in range(p * n)]  # 20 x 40
    
    # Strassen coefficients for a 2x2x2 block (local indices i,j,k in {0,1})
    # 7 terms with their U,V,W patterns (values in {-1, 0, 1})
    strassen_terms = [
        # Term 0: U=(1,1,1,1), V=(1,1,1,1), W=(1,1,1,1)  [P1: (A11+A22)(B11+B22)]
        ([1,1,1,1], [1,1,1,1], [1,1,1,1]),
        # Term 1: U=(0,0,1,1), V=(1,0,1,0), W=(0,1,0,-1)  [P2: (A21+A22)B11]
        ([0,0,1,1], [1,0,1,0], [0,1,0,-1]),
        # Term 2: U=(1,0,0,0), V=(0,1,-1,1), W=(0,0,1,1)  [P3: A11(B21-B11)]
        ([1,0,0,0], [0,1,-1,1], [0,0,1,1]),
        # Term 3: U=(0,0,0,1), V=(0,1,1,-1), W=(1,1,0,0)  [P4: A22(B12-B22)]
        ([0,0,0,1], [0,1,1,-1], [1,1,0,0]),
        # Term 4: U=(1,1,0,0), V=(0,0,0,1), W=(-1,1,0,0)  [P5: (A11+A12)B22]
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # Term 5: U=(1,-1,1,0), V=(1,1,0,0), W=(0,0,0,1)  [P6: (A21-A11)(B11+B12)]
        ([1,-1,1,0], [1,1,0,0], [0,0,0,1]),
        # Term 6: U=(1,0,-1,0), V=(0,1,0,1), W=(1,0,0,1)  [P7: (A12-A22)(B21+B22)]
        ([1,0,-1,0], [0,1,0,1], [1,0,0,1]),
    ]
    
    # Process 5 blocks of 2 k-columns each (k=0,1; 2,3; 4,5; 6,7; 8,9)
    for block in range(5):
        k_base = 2 * block
        
        # 7 Strassen terms for j=0,1 (local j indices 0,1)
        for t in range(7):
            col = block * 8 + t
            u_vals, v_vals, w_vals = strassen_terms[t]
            # u_vals indexed by (i, j_local) where i in {0,1}, j_local in {0,1}
            # U global index: i * m + j_local = i * 3 + j_local
            for i in range(2):
                for j_local in range(2):
                    idx = i * 2 + j_local
                    if u_vals[idx] != 0:
                        U[i * m + j_local][col] = u_vals[idx]
            
            # V global index: j_local * p + (k_base + k_local)
            for j_local in range(2):
                for k_local in range(2):
                    idx = j_local * 2 + k_local
                    if v_vals[idx] != 0:
                        V[j_local * p + k_base + k_local][col] = v_vals[idx]
            
            # W global index: (k_base + k_local) * n + i
            for k_local in range(2):
                for i in range(2):
                    idx = k_local * 2 + i
                    if w_vals[idx] != 0:
                        W[(k_base + k_local) * n + i][col] = w_vals[idx]
        
        # 1 schoolbook-like term for j=2, using both k columns
        col = block * 8 + 7
        # U: for i=0,1; j=2 -> U[i*3+2][col] = 1
        for i in range(2):
            U[i * m + 2][col] = 1
        # V: for j=2; k=k_base, k_base+1 -> V[2*10 + k][col] = 1
        V[2 * p + k_base][col] = 1
        V[2 * p + k_base + 1][col] = 1
        # W: for k=k_base, k_base+1; i=0,1 -> W[k*2 + i][col] = 1
        for k_local in range(2):
            for i in range(2):
                W[(k_base + k_local) * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
