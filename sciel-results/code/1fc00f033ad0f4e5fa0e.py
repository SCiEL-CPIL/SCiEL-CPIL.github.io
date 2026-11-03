import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We will build lists of (u, v, w) tuples
    terms = []
    
    # Helper to add a Strassen block
    # A_block: (2, 2) indices (i, j_local) -> global (i, j_global)
    # B_block: (2, 2) indices (j_local, k_local) -> global (j_global, k_global)
    # C_block: (2, 2) indices (i, k_local) -> global (i, k_global)
    # offsets: (i_off, j_off, k_off) for global indices
    # Note: A is 2x4, B is 4x4, C is 2x4
    # A indices: i in 0..1, j in 0..3
    # B indices: j in 0..3, k in 0..3
    # C indices: i in 0..1, k in 0..3
    
    # Strassen coefficients for the 7 multiplications
    # M1: (X00+X11)(Y00+Y11)
    # M2: (X10+X11)Y00
    # M3: X00(Y01-Y11)
    # M4: X11(Y10-Y00)
    # M5: (X00+X01)Y11
    # M6: (X10-X00)(Y00+Y01)
    # M7: (X01-X11)(Y10+Y11)
    
    # Output reconstruction:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(A_i_off, A_j_off, B_j_off, B_k_off, C_i_off, C_k_off):
        # A is 2x4. Local A is 2x2 at (i, j)
        # B is 4x4. Local B is 2x2 at (j, k)
        # C is 2x4. Local C is 2x2 at (i, k)
        
        # Indices in local 2x2 space: 0, 1
        # Global indices:
        # A: i + A_i_off, j + A_j_off
        # B: j + B_j_off, k + B_k_off
        # C: i + C_i_off, k + C_k_off
        
        # We need to define the linear forms for each of the 7 M's
        # Each M contributes to specific Z entries.
        
        # We will create 7 rank-1 terms.
        # For each term, we define u (for A), v (for B), w (for C).
        # u has size 8 (2*4), v has size 16 (4*4), w has size 8 (4*2).
        # Wait, W shape is (p*n, rank) = (4*2, rank) = (8, rank).
        # W index is k*n + i.
        
        # Let's represent u, v, w as dictionaries or lists of (index, coeff)
        
        # M1
        # A: (0,0) and (1,1) in local -> global (A_i_off+0, A_j_off+0) and (A_i_off+1, A_j_off+1)
        # B: (0,0) and (1,1) in local -> global (B_j_off+0, B_k_off+0) and (B_j_off+1, B_k_off+1)
        # C: 
        # Z00 += 1, Z11 += 1
        # Local Z00 -> Global (C_i_off+0, C_k_off+0) -> W index (C_k_off+0)*2 + (C_i_off+0)
        # Local Z11 -> Global (C_i_off+1, C_k_off+1) -> W index (C_k_off+1)*2 + (C_i_off+1)
        
        # We will store terms as (u_dict, v_dict, w_dict)
        
        def make_vec(size, items):
            d = {}
            for idx, coeff in items:
                d[idx] = d.get(idx, 0) + coeff
            return d
        
        # M1
        u1 = make_vec(n*m, [
            ((A_i_off+0)*m + (A_j_off+0), 1),
            ((A_i_off+1)*m + (A_j_off+1), 1)
        ])
        v1 = make_vec(m*p, [
            ((B_j_off+0)*p + (B_k_off+0), 1),
            ((B_j_off+1)*p + (B_k_off+1), 1)
        ])
        w1 = make_vec(p*n, [
            ((C_k_off+0)*n + (C_i_off+0), 1), # Z00
            ((C_k_off+1)*n + (C_i_off+1), 1)  # Z11
        ])
        terms.append((u1, v1, w1))
        
        # M2: (X10+X11)Y00
        # A: (1,0) and (1,1)
        # B: (0,0)
        # C: Z10 += 1
        u2 = make_vec(n*m, [
            ((A_i_off+1)*m + (A_j_off+0), 1),
            ((A_i_off+1)*m + (A_j_off+1), 1)
        ])
        v2 = make_vec(m*p, [
            ((B_j_off+0)*p + (B_k_off+0), 1)
        ])
        w2 = make_vec(p*n, [
            ((C_k_off+0)*n + (C_i_off+1), 1) # Z10
        ])
        terms.append((u2, v2, w2))
        
        # M3: X00(Y01-Y11)
        # A: (0,0)
        # B: (0,1) and (1,1) with -1
        # C: Z01 += 1
        u3 = make_vec(n*m, [
            ((A_i_off+0)*m + (A_j_off+0), 1)
        ])
        v3 = make_vec(m*p, [
            ((B_j_off+0)*p + (B_k_off+1), 1),
            ((B_j_off+1)*p + (B_k_off+1), -1)
        ])
        w3 = make_vec(p*n, [
            ((C_k_off+1)*n + (C_i_off+0), 1) # Z01
        ])
        terms.append((u3, v3, w3))
        
        # M4: X11(Y10-Y00)
        # A: (1,1)
        # B: (1,0) and (0,0) with -1
        # C: Z00 += 1, Z10 += 1
        u4 = make_vec(n*m, [
            ((A_i_off+1)*m + (A_j_off+1), 1)
        ])
        v4 = make_vec(m*p, [
            ((B_j_off+1)*p + (B_k_off+0), 1),
            ((B_j_off+0)*p + (B_k_off+0), -1)
        ])
        w4 = make_vec(p*n, [
            ((C_k_off+0)*n + (C_i_off+0), 1), # Z00
            ((C_k_off+0)*n + (C_i_off+1), 1)  # Z10
        ])
        terms.append((u4, v4, w4))
        
        # M5: (X00+X01)Y11
        # A: (0,0) and (0,1)
        # B: (1,1)
        # C: Z00 -= 1, Z01 += 1
        u5 = make_vec(n*m, [
            ((A_i_off+0)*m + (A_j_off+0), 1),
            ((A_i_off+0)*m + (A_j_off+1), 1)
        ])
        v5 = make_vec(m*p, [
            ((B_j_off+1)*p + (B_k_off+1), 1)
        ])
        w5 = make_vec(p*n, [
            ((C_k_off+0)*n + (C_i_off+0), -1), # Z00
            ((C_k_off+1)*n + (C_i_off+0), 1)  # Z01
        ])
        terms.append((u5, v5, w5))
        
        # M6: (X10-X00)(Y00+Y01)
        # A: (1,0) and (0,0) with -1
        # B: (0,0) and (0,1)
        # C: Z11 += 1
        u6 = make_vec(n*m, [
            ((A_i_off+1)*m + (A_j_off+0), 1),
            ((A_i_off+0)*m + (A_j_off+0), -1)
        ])
        v6 = make_vec(m*p, [
            ((B_j_off+0)*p + (B_k_off+0), 1),
            ((B_j_off+0)*p + (B_k_off+1), 1)
        ])
        w6 = make_vec(p*n, [
            ((C_k_off+1)*n + (C_i_off+1), 1) # Z11
        ])
        terms.append((u6, v6, w6))
        
        # M7: (X01-X11)(Y10+Y11)
        # A: (0,1) and (1,1) with -1
        # B: (1,0) and (1,1)
        # C: Z00 += 1
        u7 = make_vec(n*m, [
            ((A_i_off+0)*m + (A_j_off+1), 1),
            ((A_i_off+1)*m + (A_j_off+1), -1)
        ])
        v7 = make_vec(m*p, [
            ((B_j_off+1)*p + (B_k_off+0), 1),
            ((B_j_off+1)*p + (B_k_off+1), 1)
        ])
        w7 = make_vec(p*n, [
            ((C_k_off+0)*n + (C_i_off+0), 1) # Z00
        ])
        terms.append((u7, v7, w7))

    # Call for the 4 blocks
    # A_L: i=0..1, j=0..1. Offsets: i=0, j=0
    # B_LL: j=0..1, k=0..1. Offsets: j=0, k=0
    # C_L: i=0..1, k=0..1. Offsets: i=0, k=0
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # A_R: i=0..1, j=2..3. Offsets: i=0, j=2
    # B_RL: j=2..3, k=0..1. Offsets: j=2, k=0
    # C_L: i=0..1, k=0..1. Offsets: i=0, k=0
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # A_L: i=0..1, j=0..1. Offsets: i=0, j=0
    # B_LR: j=0..1, k=2..3. Offsets: j=0, k=2
    # C_R: i=0..1, k=2..3. Offsets: i=0, k=2
    add_strassen_block(0, 0, 0, 2, 0, 2)
    
    # A_R: i=0..1, j=2..3. Offsets: i=0, j=2
    # B_RR: j=2..3, k=2..3. Offsets: j=2, k=2
    # C_R: i=0..1, k=2..3. Offsets: i=0, k=2
    add_strassen_block(0, 2, 2, 2, 0, 2)

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][col] = val
        for idx, val in v.items():
            V[idx][col] = val
        for idx, val in w.items():
            W[idx][col] = val
            
    return {"U": U, "V": V, "W": W}
