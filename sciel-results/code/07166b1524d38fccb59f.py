"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=15
    
    # We use 7 Strassen blocks for 14 columns, and 4 naive terms for the last column.
    # Total rank = 7 * 7 + 4 = 53.
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u: [A00, A01, A10, A11]
    # v_local: [B00, B01, B10, B11] (indices j*2+k)
    # w_local: [C00, C01, C10, C11] (indices k*2+i) -> wait, W index is k*n+i.
    # n=2. So W index is 2*k + i.
    # C00: k=0, i=0 -> idx 0
    # C01: k=1, i=0 -> idx 2
    # C10: k=0, i=1 -> idx 1
    # C11: k=1, i=1 -> idx 3
    
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # w: C00(1), C11(1) -> idx 0:1, idx 3:1. 
        # Wait, w_local list above was [C00, C01, C10, C11]?
        # Let's map w_local list indices to W indices.
        # W idx 0: k=0, i=0 (C00)
        # W idx 1: k=0, i=1 (C10)
        # W idx 2: k=1, i=0 (C01)
        # W idx 3: k=1, i=1 (C11)
        
        # Let's re-verify w for each term based on this mapping.
    ]
    
    # Term 1: M1
    # u: [1, 0, 0, 1]
    # v: [1, 0, 0, 1] (B00, B01, B10, B11) -> B00, B11
    # C: C00 += 1, C11 += 1
    # w: idx 0 += 1, idx 3 += 1. w = [1, 0, 0, 1]
    
    # Term 2: M2
    # u: [0, 0, 1, 1] (A10, A11)
    # v: [1, 0, 0, 0] (B00)
    # C: C10 += 1, C11 -= 1
    # w: idx 1 += 1, idx 3 -= 1. w = [0, 1, 0, -1]
    
    # Term 3: M3
    # u: [1, 0, 0, 0] (A00)
    # v: [0, 1, 0, -1] (B01, -B11)
    # C: C01 += 1, C11 += 1
    # w: idx 2 += 1, idx 3 += 1. w = [0, 0, 1, 1]
    
    # Term 4: M4
    # u: [0, 0, 0, 1] (A11)
    # v: [-1, 0, 1, 0] (-B00, B10)
    # C: C00 += 1, C10 += 1
    # w: idx 0 += 1, idx 1 += 1. w = [1, 1, 0, 0]
    
    # Term 5: M5
    # u: [1, 1, 0, 0] (A00, A01)
    # v: [0, 0, 0, 1] (B11)
    # C: C00 -= 1, C01 += 1
    # w: idx 0 -= 1, idx 2 += 1. w = [-1, 0, 1, 0]
    
    # Term 6: M6
    # u: [-1, 0, 1, 0] (-A00, A10)
    # v: [1, 1, 0, 0] (B00, B01)
    # C: C11 += 1
    # w: idx 3 += 1. w = [0, 0, 0, 1]
    
    # Term 7: M7
    # u: [0, 1, 0, -1] (A01, -A11)
    # v: [0, 0, 1, 1] (B10, B11)
    # C: C00 += 1
    # w: idx 0 += 1. w = [1, 0, 0, 0]

    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    col = 0
    # Process 7 pairs of columns
    for q in range(7):
        for u_vec, v_vec, w_vec in terms:
            # Fill U
            for idx in range(4):
                U[idx][col] = u_vec[idx]
            
            # Fill V
            # v_vec is [B00, B01, B10, B11]
            # Indices: j=0, k=0 -> 0; j=0, k=1 -> 1; j=1, k=0 -> 2; j=1, k=1 -> 3
            # Global V index: j * 15 + k_global
            # k_global = 2*q + k_local
            # v_vec[0] is B00 (j=0, k_loc=0) -> V[0*15 + 2q][col]
            # v_vec[1] is B01 (j=0, k_loc=1) -> V[0*15 + 2q+1][col]
            # v_vec[2] is B10 (j=1, k_loc=0) -> V[1*15 + 2q][col]
            # v_vec[3] is B11 (j=1, k_loc=1) -> V[1*15 + 2q+1][col]
            
            V[0 * 15 + (2 * q)][col] = v_vec[0]
            V[0 * 15 + (2 * q + 1)][col] = v_vec[1]
            V[1 * 15 + (2 * q)][col] = v_vec[2]
            V[1 * 15 + (2 * q + 1)][col] = v_vec[3]
            
            # Fill W
            # w_vec is [C00, C10, C01, C11]
            # Indices in w_vec:
            # 0: C00 (k_loc=0, i=0) -> W[2*(2q+0) + 0][col]
            # 1: C10 (k_loc=0, i=1) -> W[2*(2q+0) + 1][col]
            # 2: C01 (k_loc=1, i=0) -> W[2*(2q+1) + 0][col]
            # 3: C11 (k_loc=1, i=1) -> W[2*(2q+1) + 1][col]
            
            W[2 * (2 * q) + 0][col] = w_vec[0]
            W[2 * (2 * q) + 1][col] = w_vec[1]
            W[2 * (2 * q + 1) + 0][col] = w_vec[2]
            W[2 * (2 * q + 1) + 1][col] = w_vec[3]
            
            col += 1
            
    # Process last column (k=14) with 4 naive terms
    # C_i,14 = A_i,0 B_0,14 + A_i,1 B_1,14
    # Term 1: A_0,0 B_0,14 -> C_0,14
    U[0][col] = 1 # A00
    V[0 * 15 + 14][col] = 1 # B0,14
    W[2 * 14 + 0][col] = 1 # C0,14
    col += 1
    
    # Term 2: A_0,1 B_1,14 -> C_0,14
    U[1][col] = 1 # A01
    V[1 * 15 + 14][col] = 1 # B1,14
    W[2 * 14 + 0][col] = 1 # C0,14
    col += 1
    
    # Term 3: A_1,0 B_0,14 -> C_1,14
    U[2][col] = 1 # A10
    V[0 * 15 + 14][col] = 1 # B0,14
    W[2 * 14 + 1][col] = 1 # C1,14
    col += 1
    
    # Term 4: A_1,1 B_1,14 -> C_1,14
    U[3][col] = 1 # A11
    V[1 * 15 + 14][col] = 1 # B1,14
    W[2 * 14 + 1][col] = 1 # C1,14
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
