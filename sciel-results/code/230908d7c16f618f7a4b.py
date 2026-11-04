import time

DIMENSIONS = [4, 4, 9]
DEADLINE = 125.0

# Strassen 2x2x2
# Returns list of (a_vec, b_vec, c_vec) where c_vec is the coefficient for C blocks
# a_vec, b_vec are 4-vectors (for 2x2 matrix)
STRASSEN_2x2 = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1]), # M0 -> C00, C11
    ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 1, 1, 0]), # M1 -> C10, C11 (-M1)
    ([0, 0, 1, 1], [0, 1, 0, 0], [0, 0, 1, 1]),  # M2 -> C01, C11
    ([1, 0, 0, 0], [0, 1, 0, -1], [1, 1, 0, 0]), # M3 -> C00, C10
    ([1, 1, 0, 0], [1, 0, 0, 1], [-1, 0, 0, 1]), # M4 -> C00, C01
    ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, -1]),# M5 -> C11
    ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]), # M6 -> C00
]
# Note: The c_vec above is for the 2x2 C matrix.
# C00 = M0 + M3 - M4 + M6
# C01 = M2 + M4
# C10 = M1 + M3
# C11 = M0 - M1 + M2 - M5

def generate_strassen_4x4_components():
    """
    Generates the 18 rank-1 components for 4x4 * 4x4 -> 4x4.
    Returns a list of (U_comp, V_comp, W_comp)
    U_comp: 4x4 matrix
    V_comp: 4x4 matrix
    W_comp: 4x4 matrix
    """
    components = []
    
    # Outer Strassen terms (7)
    for outer_s in range(7):
        a_outer = STRASSEN_2x2[outer_s][0]
        b_outer = STRASSEN_2x2[outer_s][1]
        c_outer = STRASSEN_2x2[outer_s][2]
        
        # c_outer is a 4-vector for the 2x2 C blocks: [C00, C01, C10, C11]
        # We need to combine with inner Strassen terms
        for inner_s in range(7):
            a_inner = STRASSEN_2x2[inner_s][0]
            b_inner = STRASSEN_2x2[inner_s][1]
            c_inner = STRASSEN_2x2[inner_s][2]
            
            # Check if this combination is needed
            # The outer term contributes to C blocks based on c_outer
            # The inner term contributes to sub-blocks based on c_inner
            # We need to see if the product is non-zero
            
            # Construct U_comp (4x4)
            U_comp = [[0]*4 for _ in range(4)]
            for oi in range(2):
                for oj in range(2):
                    a_val = a_outer[oi*2+oj]
                    if a_val == 0: continue
                    for ii in range(2):
                        for ij in range(2):
                            a_i_val = a_inner[ii*2+ij]
                            if a_i_val == 0: continue
                            i = oi*2 + ii
                            j = oj*2 + ij
                            U_comp[i][j] = a_val * a_i_val
            
            # Construct V_comp (4x4)
            V_comp = [[0]*4 for _ in range(4)]
            for oj in range(2):
                for ok in range(2):
                    b_val = b_outer[oj*2+ok]
                    if b_val == 0: continue
                    for ij in range(2):
                        for ik in range(2):
                            b_i_val = b_inner[ij*2+ik]
                            if b_i_val == 0: continue
                            j = oj*2 + ij
                            k = ok*2 + ik
                            V_comp[j][k] = b_val * b_i_val
            
            # Construct W_comp (4x4)
            W_comp = [[0]*4 for _ in range(4)]
            # c_outer tells which 2x2 block of C this outer term affects
            # c_inner tells which 2x2 sub-block of that block this inner term affects
            for oc in range(4):
                c_o_val = c_outer[oc]
                if c_o_val == 0: continue
                ci_block, cj_block = divmod(oc, 2) # row, col of 2x2 block
                for ic in range(4):
                    c_i_val = c_inner[ic]
                    if c_i_val == 0: continue
                    ii_block, ij_block = divmod(ic, 2) # row, col of 2x2 sub-block
                    i = ci_block*2 + ii_block
                    j = cj_block*2 + ij_block
                    W_comp[i][j] = c_o_val * c_i_val
            
            # Check if all are zero
            if all(all(U_comp[i][j]==0 for j in range(4)) for i in range(4)):
                continue
            if all(all(V_comp[i][j]==0 for j in range(4)) for i in range(4)):
                continue
            if all(all(W_comp[i][j]==0 for j in range(4)) for i in range(4)):
                continue
                
            components.append((U_comp, V_comp, W_comp))
            
    return components

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    def check_time():
        return time.time() - t0 < DEADLINE

    # Get 18 components for 4x4 * 4x4
    comps_4x4 = generate_strassen_4x4_components()
    # There should be 18 components
    
    all_U = []
    all_V = []
    all_W = []
    
    # Part 1: B cols 0-3
    for U, V, W in comps_4x4:
        # U is 4x4 (i,j)
        # V is 4x4 (j,k) -> map k to 0-3
        # W is 4x4 (k,i) -> map k to 0-3
        U_flat = [U[i][j] for i in range(4) for j in range(4)]
        V_flat = [V[j][k] for j in range(4) for k in range(4)]
        W_flat = [W[k][i] for k in range(4) for i in range(4)]
        all_U.append(U_flat)
        all_V.append(V_flat)
        all_W.append(W_flat)
        
    # Part 2: B cols 4-7
    for U, V, W in comps_4x4:
        U_flat = [U[i][j] for i in range(4) for j in range(4)]
        # V: j in 0..3, k in 0..3 -> k_new = k+4
        V_flat = [0]*36
        for j in range(4):
            for k in range(4):
                V_flat[j*9 + (k+4)] = V[j][k]
        # W: k in 0..3, i in 0..3 -> k_new = k+4
        W_flat = [0]*36
        for k in range(4):
            for i in range(4):
                W_flat[(k+4)*4 + i] = W[k][i]
        all_U.append(U_flat)
        all_V.append(V_flat)
        all_W.append(W_flat)
        
    # Part 3: B col 8 (Rank 1)
    # U[i,j] = 1
    # V[j,8] = 1
    # W[8,i] = 1
    U_flat = [1]*16
    V_flat = [0]*36
    for j in range(4):
        V_flat[j*9 + 8] = 1
    W_flat = [0]*36
    for i in range(4):
        W_flat[8*4 + i] = 1
    all_U.append(U_flat)
    all_V.append(V_flat)
    all_W.append(W_flat)
    
    r = len(all_U)
    
    # Build final matrices
    U = [u for u in all_U]
    V = [v for v in all_V]
    W = [w for w in all_W]
    
    # Transpose to match required shapes
    # U: (n*m, r) -> (16, r)
    # V: (m*p, r) -> (36, r)
    # W: (p*n, r) -> (36, r)
    
    U_final = [[0]*r for _ in range(n*m)]
    for c in range(r):
        for idx in range(n*m):
            U_final[idx][c] = all_U[c][idx]
            
    V_final = [[0]*r for _ in range(m*p)]
    for c in range(r):
        for idx in range(m*p):
            V_final[idx][c] = all_V[c][idx]
            
    W_final = [[0]*r for _ in range(p*n)]
    for c in range(r):
        for idx in range(p*n):
            W_final[idx][c] = all_W[c][idx]
            
    return {"U": U_final, "V": V_final, "W": W_final}
