"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 Strassen terms = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definition for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Coefficients for U (A entries: a,b,c,d at indices 0,1,2,3)
    # Coefficients for V (B entries: e,f,g,h at local indices 0,1,2,3)
    # Coefficients for W (C entries: local 0,1,2,3 corresponding to C00, C01, C10, C11)
    
    # We map local B indices (e,f,g,h) to global V indices.
    # Block r uses columns 2r and 2r+1.
    # e = B[0, 2r] -> V index 0*p + 2r
    # f = B[0, 2r+1] -> V index 0*p + 2r+1
    # g = B[1, 2r] -> V index 1*p + 2r
    # h = B[1, 2r+1] -> V index 1*p + 2r+1
    
    # We map local C indices (C00, C01, C10, C11) to global W indices.
    # C00 is row 0, col 2r. W index = (2r)*n + 0
    # C01 is row 0, col 2r+1. W index = (2r+1)*n + 0
    # C10 is row 1, col 2r. W index = (2r)*n + 1
    # C11 is row 1, col 2r+1. W index = (2r+1)*n + 1
    
    terms = []
    
    # Term 1: M1 = (a+d)(e+h)
    terms.append({
        'u': {0: 1, 3: 1}, # a, d
        'v': {0: 1, 3: 1}, # e, h (local)
        'w': {0: 1, 3: 1}  # C00, C11 (local)
    })
    # Term 2: M2 = (c+d)e
    terms.append({
        'u': {2: 1, 3: 1}, # c, d
        'v': {0: 1},       # e
        'w': {2: 1, 3: -1} # C10, -C11
    })
    # Term 3: M3 = a(f-h)
    terms.append({
        'u': {0: 1},       # a
        'v': {1: 1, 3: -1},# f, -h
        'w': {1: 1, 3: 1}  # C01, C11
    })
    # Term 4: M4 = d(g-e)
    terms.append({
        'u': {3: 1},       # d
        'v': {2: 1, 0: -1},# g, -e
        'w': {0: 1, 2: 1}  # C00, C10
    })
    # Term 5: M5 = (a+b)h
    terms.append({
        'u': {0: 1, 1: 1}, # a, b
        'v': {3: 1},       # h
        'w': {0: -1, 1: 1} # -C00, C01
    })
    # Term 6: M6 = (c-a)(e+f)
    terms.append({
        'u': {2: 1, 0: -1},# c, -a
        'v': {0: 1, 1: 1}, # e, f
        'w': {3: 1}        # C11
    })
    # Term 7: M7 = (b-d)(g+h)
    terms.append({
        'u': {1: 1, 3: -1},# b, -d
        'v': {2: 1, 3: 1}, # g, h
        'w': {0: 1}        # C00
    })
    
    col = 0
    for r in range(4):
        # Global indices for B in this block
        # e: V[0*p + 2r]
        # f: V[0*p + 2r+1]
        # g: V[1*p + 2r]
        # h: V[1*p + 2r+1]
        v_idx_map = {
            0: 0 * p + 2 * r,
            1: 0 * p + 2 * r + 1,
            2: 1 * p + 2 * r,
            3: 1 * p + 2 * r + 1
        }
        
        # Global indices for C in this block
        # C00: W[(2r)*n + 0]
        # C01: W[(2r+1)*n + 0]
        # C10: W[(2r)*n + 1]
        # C11: W[(2r+1)*n + 1]
        w_idx_map = {
            0: (2 * r) * n + 0,
            1: (2 * r + 1) * n + 0,
            2: (2 * r) * n + 1,
            3: (2 * r + 1) * n + 1
        }
        
        for t in terms:
            # Update U
            for u_local_idx, val in t['u'].items():
                U[u_local_idx][col] += val
            
            # Update V
            for v_local_idx, val in t['v'].items():
                v_global_idx = v_idx_map[v_local_idx]
                V[v_global_idx][col] += val
            
            # Update W
            for w_local_idx, val in t['w'].items():
                w_global_idx = w_idx_map[w_local_idx]
                W[w_global_idx][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
