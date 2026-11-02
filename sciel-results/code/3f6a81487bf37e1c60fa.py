"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (32, 56)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition factors
    # Indices: U(i,j), V(j,k), W(k,i) where i,j in {0,1}, k in {0,1}
    # Global indices:
    # U: i*m + j
    # V: j*p + k_global
    # W: k_global*n + i
    # k_global = 2*t + k_local
    
    # Define the 7 Strassen terms
    # Each term is a list of (u_idx, v_idx, w_idx, coeff)
    # But since it's a tensor decomposition, each term contributes to one rank dimension.
    # For a specific term s and block t:
    # col = t * 7 + s
    # U entries: for (i, j, c) in u_entries: U[i*m+j][col] += c
    # V entries: for (j, k_local, c) in v_entries: V[j*p + 2*t + k_local][col] += c
    # W entries: for (i, k_local, c) in w_entries: W[(2*t + k_local)*n + i][col] += c
    
    # Strassen factors (U, V, W contributions)
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10, C11
    # M3 = A00(B01-B11) -> C01, C11
    # M4 = A11(B10-B00) -> C00, C10
    # M5 = (A00+A01)B11 -> C00, C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00
    
    strassen = [
        # (U_entries, V_entries, W_entries)
        # U: (i, j, coeff)
        # V: (j, k_local, coeff)
        # W: (i, k_local, coeff)
        
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_entries, v_entries, w_entries = strassen[s]
            
            for i, j, c in u_entries:
                U[i * m + j][col] = c
                
            for j, k_local, c in v_entries:
                k_global = 2 * t + k_local
                V[j * p + k_global][col] = c
                
            for i, k_local, c in w_entries:
                k_global = 2 * t + k_local
                W[k_global * n + i][col] = c
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
