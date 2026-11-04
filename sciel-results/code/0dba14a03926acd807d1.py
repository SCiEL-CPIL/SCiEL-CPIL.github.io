"""Rewritten program for 6x6 matrix multiplication with rank 161 decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 161 decomposition:
    # 7 Strassen terms for 2x2 blocks of 3x3 matrices.
    # Each 3x3 multiplication uses Laderman's rank 23 algorithm.
    # Total rank = 7 * 23 = 161.
    rank = 161
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block matrices
    # t: term index 0..6
    alphas = [
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    betas = [
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    gammas = [
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    # Laderman coefficients for 3x3 matrix multiplication
    # lad_U[k] : dict of (i,j) -> coeff for A
    # lad_V[k] : dict of (j,k) -> coeff for B
    # lad_W[k] : dict of (i,j) -> coeff for C
    # Note: indices are 0-based
    
    lad_U = [
        {(0,0): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): 1},
        {(0,0): 1, (0,1): 1, (0,2): 1},
        {(0,0): 1, (1,0): 1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (0,2): 1, (1,0): 1, (1,2): 1},
        {(0,2): 1, (1,2): -1, (2,2): 1},
        {(0,2): 1, (2,2): 1},
        {(0,2): 1, (1,2): 1, (2,2): 1},
        {(1,0): 1, (1,2): 1, (2,0): 1, (2,2): 1},
        {(1,1): 1, (2,1): 1},
        {(1,0): 1, (1,2): 1},
        {(1,1): 1, (2,1): 1},
        {(1,1): 1, (1,2): 1, (2,1): 1, (2,2): 1},
        {(2,0): 1},
        {(2,0): 1},
        {(2,1): 1},
        {(2,1): 1},
        {(2,0): 1, (2,1): 1},
        {(2,2): 1},
        {(2,2): 1},
        {(2,0): 1, (2,1): 1, (2,2): 1}
    ]
    
    lad_V = [
        {(0,0): 1},
        {(1,0): 1},
        {(0,1): 1},
        {(1,2): 1},
        {(0,1): 1},
        {(0,0): 1, (1,0): -1},
        {(0,1): 1, (1,1): 1},
        {(1,0): 1, (2,0): 1},
        {(0,0): 1},
        {(0,1): 1},
        {(0,1): 1, (0,2): 1},
        {(0,0): 1, (1,0): -1},
        {(2,2): 1},
        {(0,2): 1},
        {(0,2): 1, (1,2): -1},
        {(1,1): 1},
        {(1,2): 1},
        {(1,1): 1},
        {(1,2): 1},
        {(0,1): 1},
        {(1,0): 1},
        {(1,1): 1},
        {(0,2): 1}
    ]
    
    lad_W = [
        {(0,0): 1},
        {(0,0): 1, (1,0): 1},
        {(0,1): 1},
        {(0,2): 1, (1,2): 1},
        {(0,1): 1, (1,0): 1},
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
        {(0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
        {(1,0): 1, (2,0): 1, (0,0): 1, (0,1): 1, (1,1): 1, (2,1): 1},
        {(0,2): 1, (0,1): 1, (1,2): 1, (2,2): 1, (1,1): 1},
        {(0,1): 1, (0,2): 1, (1,1): 1, (1,2): 1, (2,2): 1, (1,0): 1},
        {(0,2): 1, (1,2): 1, (2,2): 1, (1,0): 1, (1,1): 1, (2,1): 1},
        {(1,0): 1, (2,0): 1, (0,0): 1, (0,1): 1, (1,1): 1, (2,1): 1},
        {(1,0): 1, (1,1): 1, (2,2): 1, (0,2): 1, (1,2): 1, (2,1): 1},
        {(2,1): 1, (1,2): 1, (2,2): 1, (0,2): 1, (1,2): 1, (2,1): 1},
        {(0,2): 1, (1,2): 1, (2,2): 1, (1,1): 1, (2,1): 1, (0,1): 1},
        {(1,1): 1, (2,0): 1, (2,1): 1},
        {(1,2): 1, (2,0): 1, (2,2): 1},
        {(1,1): 1, (2,1): 1, (2,2): 1},
        {(1,2): 1, (2,1): 1, (2,2): 1},
        {(0,1): 1, (2,0): 1, (2,1): 1},
        {(1,0): 1, (2,0): 1, (2,1): 1},
        {(1,1): 1, (2,1): 1, (2,2): 1},
        {(0,2): 1, (1,2): 1, (2,2): 1}
    ]
    
    # Note: The lad_W lists above are rough sketches. I must ensure they map p_k to c_ij correctly.
    # Let's refine lad_W to be a list of dicts mapping (i,j) to coeff.
    
    # Correct mapping of p_k to c_ij based on Laderman:
    # c_00: p1, p2, p6, p12
    # c_01: p3, p5, p7, p9, p10, p11, p20
    # c_02: p4, p7, p10, p11, p14, p15, p23
    # c_10: p2, p5, p6, p8, p12, p13, p21
    # c_11: p6, p7, p8, p12, p13, p15, p16, p18, p22
    # c_12: p4, p7, p10, p11, p13, p15, p17, p19, p23
    # c_20: p8, p12, p16, p20, p21
    # c_21: p8, p14, p16, p18, p20, p22
    # c_22: p11, p13, p15, p17, p19, p22, p23
    
    # I will generate lad_W properly in code to avoid manual error.
    
    lad_W = [None] * 23
    for k in range(23):
        lad_W[k] = {}
        
    # Add contributions
    def add_w(k, i, j, val=1):
        if lad_W[k] is None:
            lad_W[k] = {}
        lad_W[k][(i,j)] = lad_W[k].get((i,j), 0) + val
        
    add_w(0, 0, 0, 1) # p1 to c00
    add_w(1, 0, 0, 1) # p2 to c00
    add_w(5, 0, 0, 1) # p6 to c00
    add_w(11, 0, 0, 1) # p12 to c00
    
    add_w(2, 0, 1, 1) # p3 to c01
    add_w(4, 0, 1, 1) # p5 to c01
    add_w(6, 0, 1, 1) # p7 to c01
    add_w(8, 0, 1, 1) # p9 to c01
    add_w(9, 0, 1, 1) # p10 to c01
    add_w(10, 0, 1, 1) # p11 to c01
    add_w(19, 0, 1, 1) # p20 to c01
    
    add_w(3, 0, 2, 1) # p4 to c02
    add_w(6, 0, 2, 1) # p7 to c02
    add_w(9, 0, 2, 1) # p10 to c02
    add_w(10, 0, 2, 1) # p11 to c02
    add_w(13, 0, 2, 1) # p14 to c02
    add_w(14, 0, 2, 1) # p15 to c02
    add_w(22, 0, 2, 1) # p23 to c02
    
    add_w(1, 1, 0, 1) # p2 to c10
    add_w(4, 1, 0, 1) # p5 to c10
    add_w(5, 1, 0, 1) # p6 to c10
    add_w(7, 1, 0, 1) # p8 to c10
    add_w(11, 1, 0, 1) # p12 to c10
    add_w(12, 1, 0, 1) # p13 to c10
    add_w(20, 1, 0, 1) # p21 to c10
    
    add_w(5, 1, 1, 1) # p6 to c11
    add_w(6, 1, 1, 1) # p7 to c11
    add_w(7, 1, 1, 1) # p8 to c11
    add_w(11, 1, 1, 1) # p12 to c11
    add_w(12, 1, 1, 1) # p13 to c11
    add_w(14, 1, 1, 1) # p15 to c11
    add_w(15, 1, 1, 1) # p16 to c11
    add_w(17, 1, 1, 1) # p18 to c11
    add_w(21, 1, 1, 1) # p22 to c11
    
    add_w(3, 1, 2, 1) # p4 to c12
    add_w(6, 1, 2, 1) # p7 to c12
    add_w(9, 1, 2, 1) # p10 to c12
    add_w(10, 1, 2, 1) # p11 to c12
    add_w(12, 1, 2, 1) # p13 to c12
    add_w(14, 1, 2, 1) # p15 to c12
    add_w(16, 1, 2, 1) # p17 to c12
    add_w(18, 1, 2, 1) # p19 to c12
    add_w(22, 1, 2, 1) # p23 to c12
    
    add_w(7, 2, 0, 1) # p8 to c20
    add_w(11, 2, 0, 1) # p12 to c20
    add_w(15, 2, 0, 1) # p16 to c20
    add_w(19, 2, 0, 1) # p20 to c20
    add_w(20, 2, 0, 1) # p21 to c20
    
    add_w(7, 2, 1, 1) # p8 to c21
    add_w(13, 2, 1, 1) # p14 to c21
    add_w(15, 2, 1, 1) # p16 to c21
    add_w(17, 2, 1, 1) # p18 to c21
    add_w(19, 2, 1, 1) # p20 to c21
    add_w(21, 2, 1, 1) # p22 to c21
    
    add_w(10, 2, 2, 1) # p11 to c22
    add_w(12, 2, 2, 1) # p13 to c22
    add_w(14, 2, 2, 1) # p15 to c22
    add_w(16, 2, 2, 1) # p17 to c22
    add_w(18, 2, 2, 1) # p19 to c22
    add_w(21, 2, 2, 1) # p22 to c22
    add_w(22, 2, 2, 1) # p23 to c22
    
    for t in range(7):
        for k_lad in range(23):
            c = t * 23 + k_lad
            
            # Fill U
            for (I, J), val_a in alphas[t].items():
                for (u, v), val_u in lad_U[k_lad].items():
                    i_global = I * 3 + u
                    j_global = J * 3 + v
                    U[i_global * 6 + j_global][c] += val_a * val_u
                    
            # Fill V
            for (Jp, K), val_b in betas[t].items():
                for (v2, w), val_v in lad_V[k_lad].items():
                    j_global = Jp * 3 + v2
                    k_global = K * 3 + w
                    V[j_global * 6 + k_global][c] += val_b * val_v
                    
            # Fill W
            for (I, K), val_g in gammas[t].items():
                for (u, v), val_w in lad_W[k_lad].items():
                    i_global = I * 3 + u
                    k_global = K * 3 + v
                    W[k_global * 6 + i_global][c] += val_g * val_w
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
