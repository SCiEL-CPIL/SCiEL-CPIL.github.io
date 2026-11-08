"""Matrix multiplication 4x4 x 4x8 via recursive Strassen (rank 98)."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 4x4 x 4x8 = two independent 4x4 x 4x4 multiplications
    # Each 4x4 x 4x4 uses recursive Strassen with rank 49
    # Total rank = 98
    
    base_terms = strassen_2x2_terms()
    
    # Strassen combinations for 4x4 x 4x4
    # Each combo: (A_block_list, B_block_list, C_block_list)
    # Block indices: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    combs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),                  # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),                 # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(3, 1)]),                 # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, -1)]),         # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),         # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (2, -1)]) # M7
    ]
    
    # Generate 49 terms for 4x4 x 4x4
    terms_4x4_4x4 = []
    for A_blocks, B_blocks, C_blocks in combs:
        for u, v, w in base_terms:
            U_full = [0] * 16
            V_full = [0] * 16
            W_full = [0] * 16
            
            # Expand A-side: each base term's 4 entries map to 2x2 block
            for idx, sign in A_blocks:
                row_off = (idx // 2) * 2
                col_off = (idx % 2) * 2
                for i in range(2):
                    for j in range(2):
                        U_full[(row_off + i) * 4 + (col_off + j)] += sign * u[i * 2 + j]
            
            # Expand B-side: same block structure
            for idx, sign in B_blocks:
                row_off = (idx // 2) * 2
                col_off = (idx % 2) * 2
                for i in range(2):
                    for j in range(2):
                        V_full[(row_off + i) * 4 + (col_off + j)] += sign * v[i * 2 + j]
            
            # Expand C-side: W is indexed by [k*n+i] = [col*4 + row]
            # C block at (row_off, col_off) has entries W[(col_off+j)*4 + (row_off+i)]
            for idx, sign in C_blocks:
                row_off = (idx // 2) * 2
                col_off = (idx % 2) * 2
                for i in range(2):
                    for j in range(2):
                        W_full[(col_off + j) * 4 + (row_off + i)] += sign * w[i * 2 + j]
            
            terms_4x4_4x4.append((U_full, V_full, W_full))
    
    # Now build 4x4 x 4x8 terms
    # B is 4x8: left half (cols 0-3) and right half (cols 4-7)
    # C is 8x4: left half (cols 0-3) and right half (cols 4-7)
    # W is indexed by [k*4 + i] where k=col, i=row
    # Left half: k in 0..3, so W indices 0..15
    # Right half: k in 4..7, so W indices 16..31
    
    terms = []
    for U_full, V_full, W_full in terms_4x4_4x4:
        # Left half of B: cols 0-3
        V_left = V_full[:]  # V is 4x8, left half is entries 0-15
        # Right half of B: cols 4-7
        V_right = [0] * 32
        for i in range(16):
            V_right[16 + i] = V_full[i]
        
        # Left half of C: cols 0-3
        W_left = [0] * 32
        for i in range(16):
            W_left[i] = W_full[i]
        
        # Right half of C: cols 4-7
        W_right = [0] * 32
        for i in range(16):
            W_right[16 + i] = W_full[i]
        
        terms.append((U_full, V_left, W_left))
        terms.append((U_full, V_right, W_right))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            U[i][c] = u[i]
        for i in range(m * p):
            V[i][c] = v[i]
        for i in range(p * n):
            W[i][c] = w[i]
    
    return {"U": U, "V": V, "W": W}

def strassen_2x2_terms():
    """Return 7 (u, v, w) terms for 2x2 x 2x2 multiplication."""
    terms = []
    # M1 = (a+d)(e+h)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (c+d)e
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]))
    # M3 = a(f-h)
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]))
    # M4 = d(g-e)
    terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]))
    # M5 = (a+b)h
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]))
    # M6 = (c-a)(e+f)
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7 = (b-d)(g+h)
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]))
    return terms
# EVOLVE-BLOCK-END
