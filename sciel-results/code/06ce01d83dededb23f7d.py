"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2x2
    # u[l][i,j], v[l][j,k], w[l][k,i]
    # Indices: i,j in 0..1; k in 0..1
    # Flattened U index: i*2+j
    # Flattened V local index: j*2+k
    # Flattened W local index: k*2+i
    
    u_strassen = [
        [1, 0, 0, 1], # M1: a00+a11
        [0, 0, 1, 1], # M2: a10+a11
        [1, 0, 0, 0], # M3: a00
        [0, 0, 0, 1], # M4: a11
        [1, 1, 0, 0], # M5: a00+a01
        [-1, 0, 1, 0],# M6: a10-a00
        [0, 1, 0, -1] # M7: a01-a11
    ]
    
    v_strassen = [
        [1, 0, 0, 1], # M1: b00+b11
        [1, 0, 0, 0], # M2: b00
        [0, 0, 1, -1],# M3: b10-b11
        [0, 1, 0, 1], # M4: b01+b11
        [0, 0, 0, 1], # M5: b11
        [1, 1, 0, 0], # M6: b00+b01
        [0, 0, 1, 1]  # M7: b10+b11
    ]
    
    w_strassen = [
        [1, 0, 0, 1], # M1: c00+1, c11+1
        [0, 1, 0, -1],# M2: c10+1, c11-1
        [0, 0, 1, 1], # M3: c01+1, c11+1
        [1, 0, 0, 0], # M4: c00+1
        [-1, 0, 1, 0],# M5: c00-1, c01+1
        [0, 0, 0, 1], # M6: c11+1
        [-1, 1, 0, 0] # M7: c00-1, c10+1
    ]

    # Schoolbook coefficients for 2x2x1
    u_school = [
        [1, 0, 0, 0], # a00
        [0, 1, 0, 0], # a01
        [0, 0, 1, 0], # a10
        [0, 0, 0, 1]  # a11
    ]
    
    v_school = [
        [1, 0], # b0
        [0, 1], # b1
        [1, 0], # b0
        [0, 1]  # b1
    ]
    
    w_school = [
        [1, 0], # c0
        [1, 0], # c0
        [0, 1], # c1
        [0, 1]  # c1
    ]

    # Fill Strassen blocks
    for b in range(6):
        col0 = 2 * b
        col1 = 2 * b + 1
        offset = b * 7
        for l in range(7):
            L = offset + l
            # Fill U
            for idx in range(4):
                U[idx][L] = u_strassen[l][idx]
            
            # Fill V
            # Local j in 0..1, k_local in 0..1
            for j in range(2):
                for k_local in range(2):
                    k_global = col0 + k_local
                    v_idx = j * p + k_global
                    val = v_strassen[l][j * 2 + k_local]
                    V[v_idx][L] += val
            
            # Fill W
            # Local k_local in 0..1, i in 0..1
            for k_local in range(2):
                k_global = col0 + k_local
                for i in range(2):
                    w_idx = k_global * n + i
                    val = w_strassen[l][k_local * 2 + i]
                    W[w_idx][L] += val

    # Fill Schoolbook block
    col0 = 12
    offset = 42
    for l in range(4):
        L = offset + l
        # Fill U
        for idx in range(4):
            U[idx][L] = u_school[l][idx]
        
        # Fill V
        # Local j in 0..1, k_local in 0..1 (but only one col)
        for j in range(2):
            for k_local in range(2):
                # v_school[l] is length 2, corresponding to b0, b1
                # But here B is 2x1, so b0 is row 0, b1 is row 1?
                # Wait, in 2x2x1, B is 2x1.
                # My v_school above:
                # l=0: [1, 0] -> b0=1 (row 0), b1=0 (row 1)?
                # In 2x2x1, indices are j (row of B) and k (col of B).
                # k is fixed to 0 (local).
                # So v[j, 0].
                # v_school[0] = [1, 0] means v[0,0]=1, v[1,0]=0.
                # This corresponds to b0 (row 0) = 1.
                # Correct.
                k_global = col0
                v_idx = j * p + k_global
                val = v_school[l][j] # Since k_local is always 0 in 2x1 context?
                # Wait, v_strassen used j*2+k_local.
                # For 2x1, k_local is 0.
                # So v_school[l][j] is correct if v_school is stored as [v00, v10].
                # My v_school definition:
                # [1, 0] -> v[0,0]=1, v[1,0]=0.
                # [0, 1] -> v[0,0]=0, v[1,0]=1.
                # Yes.
                V[v_idx][L] += val
        
        # Fill W
        # Local k_local is 0. i in 0..1.
        for i in range(2):
            k_global = col0
            w_idx = k_global * n + i
            val = w_school[l][i] # w[0, i]
            W[w_idx][L] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
