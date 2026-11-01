"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split p=15 into seven 2s and one 1
    # 7 * 2 + 1 = 15
    rank = 7 * 7 + 4 # 49 + 4 = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    # U_local[i*m+j, r_loc], V_local[j*p_loc+k_loc, r_loc], W_local[k_loc*n+i, r_loc]
    # Here m=2, p_loc=2, n=2
    # We define the non-zero entries for the 7 Strassen products
    
    # Product 1: (A00 + A11)(B00 + B11) -> C00 +=, C11 +=
    # U: (0,0), (1,1)
    # V: (0,0), (1,1)
    # W: (0,0) idx 0, (1,1) idx 3
    strassen_U = [
        {0: 1, 3: 1}, # r0: U00, U11
        {2: 1, 3: 1}, # r1: U10, U11
        {0: 1},       # r2: U00
        {3: 1},       # r3: U11
        {0: 1, 1: 1}, # r4: U00, U01
        {2: 1, 0: -1},# r5: U10, U00
        {1: 1, 3: -1} # r6: U01, U11
    ]
    strassen_V = [
        {0: 1, 3: 1}, # r0: V00, V11
        {0: 1},       # r1: V00
        {1: 1, 3: -1},# r2: V01, V11
        {2: 1, 0: -1},# r3: V10, V00
        {3: 1},       # r4: V11
        {0: 1, 1: 1}, # r5: V00, V01
        {2: 1, 3: 1}  # r6: V10, V11
    ]
    strassen_W = [
        {0: 1, 3: 1}, # r0: W00, W11
        {1: 1, 3: -1},# r1: W10, W11
        {2: 1, 3: 1}, # r2: W01, W11
        {0: 1, 1: 1}, # r3: W00, W10
        {0: -1, 2: 1},# r4: W00, W01
        {3: 1},       # r5: W11
        {0: 1}        # r6: W00
    ]
    
    # Schoolbook 2x2x1 factors
    # Product 1: A00 B00 -> C00
    # Product 2: A01 B10 -> C00
    # Product 3: A10 B00 -> C10
    # Product 4: A11 B10 -> C10
    
    schoolbook_U = [
        {0: 1}, # r0: U00
        {1: 1}, # r1: U01
        {2: 1}, # r2: U10
        {3: 1}  # r3: U11
    ]
    schoolbook_V = [
        {0: 1}, # r0
        {1: 1}, # r1
        {0: 1}, # r2
        {1: 1}  # r3
    ]
    schoolbook_W = [
        {0: 1}, # r0: W00 (k=0, i=0) -> idx 0
        {0: 1}, # r1: W00 (k=0, i=0) -> idx 0
        {1: 1}, # r2: W10 (k=0, i=1) -> idx 1
        {1: 1}  # r3: W10 (k=0, i=1) -> idx 1
    ]

    # Fill U, V, W for 7 Strassen blocks
    for b in range(7):
        k_offset = 2 * b
        r_offset = 7 * b
        
        for r_loc in range(7):
            r_glob = r_offset + r_loc
            
            # Fill U
            for u_idx, val in strassen_U[r_loc].items():
                U[u_idx][r_glob] += val
                
            # Fill V
            for v_idx, val in strassen_V[r_loc].items():
                # v_idx is local j * 2 + k_loc
                # global idx = j * 15 + (k_offset + k_loc)
                j = v_idx // 2
                k_loc = v_idx % 2
                v_glob_idx = j * 15 + k_offset + k_loc
                V[v_glob_idx][r_glob] += val
                
            # Fill W
            for w_idx, val in strassen_W[r_loc].items():
                # w_idx is local k_loc * 2 + i
                # global idx = (k_offset + k_loc) * 2 + i
                k_loc = w_idx // 2
                i = w_idx % 2
                w_glob_idx = (k_offset + k_loc) * 2 + i
                W[w_glob_idx][r_glob] += val

    # Fill U, V, W for 1 Schoolbook block
    b = 7
    k_offset = 14
    r_offset = 49
    
    for r_loc in range(4):
        r_glob = r_offset + r_loc
        
        # Fill U
        for u_idx, val in schoolbook_U[r_loc].items():
            U[u_idx][r_glob] += val
            
        # Fill V
        for v_idx, val in schoolbook_V[r_loc].items():
            # v_idx is local j * 1 + k_loc (k_loc=0)
            j = v_idx
            k_loc = 0
            v_glob_idx = j * 15 + k_offset + k_loc
            V[v_glob_idx][r_glob] += val
            
        # Fill W
        for w_idx, val in schoolbook_W[r_loc].items():
            # w_idx is local k_loc * 2 + i (k_loc=0)
            k_loc = 0
            i = w_idx
            w_glob_idx = (k_offset + k_loc) * 2 + i
            W[w_glob_idx][r_glob] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
