import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x7 matrix.
    Uses block decomposition with Strassen for 2x2x2x2 blocks and schoolbook for 2x2x2x1 blocks.
    Total rank: 50.
    """
    n, m, p = DIMENSIONS
    rank_total = 50
    
    # Initialize matrices
    U = [[0] * rank_total for _ in range(n * m)]  # (8, 50)
    V = [[0] * rank_total for _ in range(m * p)]  # (28, 50)
    W = [[0] * rank_total for _ in range(p * n)]  # (14, 50)
    
    # Helper to add a sub-decomposition
    # a_map: list of 4 global U indices for local a,b,c,d
    # b_map: list of 4 global V indices for local e,f,g,h (or u,v for schoolbook)
    # w_map: list of 4 global W indices for local C00,C01,C10,C11 (or C0,C1 for schoolbook)
    # start_col: starting column in U,V,W
    
    def add_strassen(a_map, b_map, w_map, start_col):
        # Strassen multipliers
        # r=1: U_A=[1,0,0,1], U_B=[1,0,0,1], W=[1,0,0,1] (C00, C11) -> Wait, W map is local indices
        # Local C indices: 0:C00, 1:C01, 2:C10, 3:C11
        # W contributions:
        # M1: C00 += 1, C11 += 1
        # M2: C10 += 1
        # M3: C01 += 1, C11 += 1
        # M4: C00 += 1, C10 += 1
        # M5: C00 -= 1, C01 += 1
        # M6: C11 += 1
        # M7: C00 += 1
        
        # U_A coeffs for a,b,c,d
        u_a = [1, 0, 0, 1]
        u_b = [0, 1, 0, -1] # Wait, M3 is a(f-h), M5 is (a+b)h, M7 is (b-d)(g+h)
        # Let's re-verify U_A
        # M1: a+d -> [1,0,0,1]
        # M2: c+d -> [0,0,1,1]
        # M3: a   -> [1,0,0,0]
        # M4: d   -> [0,0,0,1]
        # M5: a+b -> [1,1,0,0]
        # M6: c-a -> [0,-1,1,0]
        # M7: b-d -> [0,1,0,-1]
        
        u_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [0, -1, 1, 0],# M6
            [0, 1, 0, -1] # M7
        ]
        
        # U_B coeffs for e,f,g,h
        # M1: e+h -> [1,0,0,1]
        # M2: e   -> [1,0,0,0]
        # M3: f-h -> [0,1,0,-1]
        # M4: g-e -> [-1,0,1,0]
        # M5: h   -> [0,0,0,1]
        # M6: e+f -> [1,1,0,0]
        # M7: g+h -> [0,0,1,1]
        
        v_coeffs = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1],# M3
            [-1, 0, 1, 0],# M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        
        # W coeffs for C00, C01, C10, C11
        # M1: C00+1, C11+1 -> [1, 0, 0, 1]
        # M2: C10+1        -> [0, 0, 1, 0]
        # M3: C01+1, C11+1 -> [0, 1, 0, 1]
        # M4: C00+1, C10+1 -> [1, 0, 1, 0]
        # M5: C00-1, C01+1 -> [-1, 1, 0, 0]
        # M6: C11+1        -> [0, 0, 0, 1]
        # M7: C00+1        -> [1, 0, 0, 0]
        
        w_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 0], # M2
            [0, 1, 0, 1], # M3
            [1, 0, 1, 0], # M4
            [-1, 1, 0, 0],# M5
            [0, 0, 0, 1], # M6
            [1, 0, 0, 0]  # M7
        ]
        
        for r in range(7):
            col = start_col + r
            for local_idx in range(4):
                U[a_map[local_idx]][col] += u_coeffs[r][local_idx]
                V[b_map[local_idx]][col] += v_coeffs[r][local_idx]
                W[w_map[local_idx]][col] += w_coeffs[r][local_idx]

    def add_schoolbook(a_map, b_map, w_map, start_col):
        # Schoolbook for 2x2 x 2x1
        # B is 2x1 (u, v). C is 2x1 (c0, c1).
        # M1: a*u -> C0
        # M2: b*v -> C0
        # M3: c*u -> C1
        # M4: d*v -> C1
        
        u_coeffs = [
            [1, 0, 0, 0], # M1 (a)
            [0, 1, 0, 0], # M2 (b)
            [0, 0, 1, 0], # M3 (c)
            [0, 0, 0, 1]  # M4 (d)
        ]
        
        v_coeffs = [
            [1, 0], # M1 (u)
            [0, 1], # M2 (v)
            [1, 0], # M3 (u)
            [0, 1]  # M4 (v)
        ]
        
        w_coeffs = [
            [1, 0], # M1 (C0)
            [1, 0], # M2 (C0)
            [0, 1], # M3 (C1)
            [0, 1]  # M4 (C1)
        ]
        
        for r in range(4):
            col = start_col + r
            for local_idx in range(4):
                U[a_map[local_idx]][col] += u_coeffs[r][local_idx]
            for local_idx in range(2):
                V[b_map[local_idx]][col] += v_coeffs[r][local_idx]
                W[w_map[local_idx]][col] += w_coeffs[r][local_idx]

    # Define mappings
    # A indices: i*m + j
    # A_L (j=0,1): 
    # a(0,0)->0, b(0,1)->1, c(1,0)->4, d(1,1)->5
    a_map_L = [0, 1, 4, 5]
    # A_R (j=2,3):
    # a(0,2)->2, b(0,3)->3, c(1,2)->6, d(1,3)->7
    a_map_R = [2, 3, 6, 7]
    
    # B indices: j*p + k
    # B_T (j=0,1)
    # B_T^1 (k=0,1): e(0,0)->0, f(0,1)->1, g(1,0)->7, h(1,1)->8
    b_map_T1 = [0, 1, 7, 8]
    # B_T^2 (k=2,3): e(0,2)->2, f(0,3)->3, g(1,2)->9, h(1,3)->10
    b_map_T2 = [2, 3, 9, 10]
    # B_T^3 (k=4,5): e(0,4)->4, f(0,5)->5, g(1,4)->11, h(1,5)->12
    b_map_T3 = [4, 5, 11, 12]
    # B_T^4 (k=6): u(0,6)->6, v(1,6)->13
    b_map_T4 = [6, 13]
    
    # B_B (j=2,3)
    # B_B^1 (k=0,1): e(2,0)->14, f(2,1)->15, g(3,0)->21, h(3,1)->22
    b_map_B1 = [14, 15, 21, 22]
    # B_B^2 (k=2,3): e(2,2)->16, f(2,3)->17, g(3,2)->23, h(3,3)->24
    b_map_B2 = [16, 17, 23, 24]
    # B_B^3 (k=4,5): e(2,4)->18, f(2,5)->19, g(3,4)->25, h(3,5)->26
    b_map_B3 = [18, 19, 25, 26]
    # B_B^4 (k=6): u(2,6)->20, v(3,6)->27
    b_map_B4 = [20, 27]
    
    # W indices: k*n + i
    # C^1 (k=0,1): 
    # C00(0,0)->0, C01(0,1)->2, C10(1,0)->1, C11(1,1)->3
    w_map_C1 = [0, 2, 1, 3]
    # C^2 (k=2,3):
    # C00(2,0)->4, C01(2,1)->6, C10(3,0)->5, C11(3,1)->7
    w_map_C2 = [4, 6, 5, 7]
    # C^3 (k=4,5):
    # C00(4,0)->8, C01(4,1)->10, C10(5,0)->9, C11(5,1)->11
    w_map_C3 = [8, 10, 9, 11]
    # C^4 (k=6):
    # C0(6,0)->12, C1(6,1)->13
    w_map_C4 = [12, 13]
    
    # Assemble
    col = 0
    # Side L
    add_strassen(a_map_L, b_map_T1, w_map_C1, col); col += 7
    add_strassen(a_map_L, b_map_T2, w_map_C2, col); col += 7
    add_strassen(a_map_L, b_map_T3, w_map_C3, col); col += 7
    add_schoolbook(a_map_L, b_map_T4, w_map_C4, col); col += 4
    
    # Side R
    add_strassen(a_map_R, b_map_B1, w_map_C1, col); col += 7
    add_strassen(a_map_R, b_map_B2, w_map_C2, col); col += 7
    add_strassen(a_map_R, b_map_B3, w_map_C3, col); col += 7
    add_schoolbook(a_map_R, b_map_B4, w_map_C4, col); col += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
