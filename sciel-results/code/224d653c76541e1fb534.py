import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # C11 (2x2) from A1(2x5)B1(5x2): 
    #   A1_top(2x2)B1_top(2x2): 7
    #   A1_bot(2x3)B1_bot(3x2): 8
    #   Total 15
    # C12 (2x3) from A1(2x5)B2(5x3):
    #   A1_top(2x2)B2_top(2x3): 9
    #   A1_bot(2x3)B2_bot(3x3): 11
    #   Total 20
    # C21 (2x2) from A2(2x5)B1(5x2): 15
    # C22 (2x3) from A2(2x5)B2(5x3): 20
    # Total Rank = 15 + 20 + 15 + 20 = 70
    
    rank = 70
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a factor
    def add_factor(u_map, v_map, w_map, u_val, v_val, w_val):
        nonlocal col
        for (ui, uj), u in u_map.items():
            U[ui * m + uj][col] += u * u_val
        for (vj, vk), v in v_map.items():
            V[vj * p + vk][col] += v * v_val
        for (wk, wi), w in w_map.items():
            W[wk * n + wi][col] += w * w_val
        col += 1
        
    # We need to define the sub-decompositions.
    # It is easier to just hardcode the contributions for each sub-block type.
    
    # Type 1: 2x2x2 (Strassen)
    # U: 2x2, V: 2x2, W: 2x2
    # Strassen factors (U, V, W) are 4x7 each.
    # We will generate them on the fly or hardcode.
    
    # Let's create a function to get the local factors for a specific sub-problem.
    
    def get_strassen_2x2():
        # Returns lists of (u_idx, val), (v_idx, val), (w_idx, val) for each of the 7 factors
        # u_idx: (i, j), v_idx: (j, k), w_idx: (k, i)
        factors = []
        # F1 = (A11+A22)(B11+B22)
        factors.append([
            [(0,0), 1], [(1,1), 1],
            [(0,0), 1], [(1,1), 1],
            [(0,0), 1], [(0,1), 1], [(1,0), 1], [(1,1), 1]
        ])
        # F2 = (A21+A22)B11
        factors.append([
            [(1,0), 1], [(1,1), 1],
            [(0,0), 1], [(1,0), 1],
            [(0,0), 1], [(0,1), 1]
        ])
        # F3 = A11(B21-B22)
        factors.append([
            [(0,0), 1],
            [(0,1), 1], [(1,1), -1],
            [(0,0), 1], [(1,0), -1]
        ])
        # F4 = A22(B12-B11)
        factors.append([
            [(1,1), 1],
            [(0,0), -1], [(1,0), 1],
            [(0,1), 1], [(1,1), -1]
        ])
        # F5 = (A11+A12)B22
        factors.append([
            [(0,0), 1], [(0,1), 1],
            [(1,1), 1], [(0,1), 1],
            [(1,1), 1], [(1,0), 1]
        ])
        # F6 = (A21-A11)(B11+B12)
        factors.append([
            [(1,0), 1], [(0,0), -1],
            [(0,0), 1], [(0,1), 1], [(1,0), 1], [(1,1), 1],
            [(0,0), 1], [(0,1), 1]
        ])
        # F7 = (A12-A22)(B21+B22)
        factors.append([
            [(0,1), 1], [(1,1), -1],
            [(0,1), 1], [(1,1), 1], [(1,0), 1], [(0,0), 1],
            [(1,0), 1], [(1,1), 1]
        ])
        return factors

    def get_2x2x1():
        # C_i1 = A_i1 B_11 + A_i2 B_21
        # Factor 1: A_i1 B_11
        # Factor 2: A_i2 B_21
        return [
            [
                [(0,0), 1], [(1,0), 1],
                [(0,0), 1],
                [(0,0), 1]
            ],
            [
                [(0,1), 1], [(1,1), 1],
                [(1,0), 1],
                [(0,1), 1]
            ]
        ]

    def get_2x1x2():
        # C_ik = A_i1 B_1k
        # Rank 1
        return [
            [
                [(0,0), 1], [(1,0), 1],
                [(0,0), 1],
                [(0,0), 1], [(0,1), 1]
            ]
        ]

    def get_2x1x1():
        # C_i1 = A_i1 B_11
        # Rank 1
        return [
            [
                [(0,0), 1], [(1,0), 1],
                [(0,0), 1],
                [(0,0), 1]
            ]
        ]

    def get_2x2x3():
        # A(2x2) B(2x3)
        # B = [B_L(2x2), B_R(2x1)]
        # A B_L: 2x2x2 (Strassen)
        # A B_R: 2x2x1
        factors = get_strassen_2x2()
        factors_2x2x1 = get_2x2x1()
        # Map indices for B_R part
        # B_R is at cols 1 (local) which is global col 1 of the 2x3 block.
        # Wait, local indices for 2x2x3:
        # U: (i, j) j in 0,1
        # V: (j, k) j in 0,1, k in 0,1,2
        # W: (k, i) k in 0,1,2, i in 0,1
        
        # For Strassen part (B_L, k in 0,1):
        # V index (j, k) stays same.
        # W index (k, i) stays same.
        
        # For 2x2x1 part (B_R, k=2 in global 2x3? No, k is local to B).
        # In 2x2x3, B is 2x3.
        # B_L is cols 0,1. B_R is col 2.
        # So for 2x2x1 factors, we need to map V's k=0 to global k=2.
        
        mapped_factors = []
        for f in factors:
            mapped_factors.append(f)
            
        for f in factors_2x2x1:
            u_part, v_part, w_part = f
            # v_part has (j, k) with k=0. Map to (j, 2).
            v_new = [(j, 2) for (j, k) in v_part]
            # w_part has (k, i) with k=0. Map to (2, i).
            w_new = [(2, i) for (k, i) in w_part]
            mapped_factors.append([u_part, v_new, w_new])
            
        return mapped_factors

    def get_2x3x2():
        # A(2x3) B(3x2)
        # A = [A_L(2x2), A_R(2x1)]
        # B = [B_L(2x2); B_R(1x2)]
        # A_L B_L: 2x2x2 (Strassen)
        # A_R B_R: 2x1x2
        
        factors = get_strassen_2x2()
        factors_2x1x2 = get_2x1x2()
        
        # Local indices for 2x3x2:
        # U: (i, j) j in 0,1,2
        # V: (j, k) j in 0,1,2, k in 0,1
        # W: (k, i) k in 0,1, i in 0,1
        
        # Strassen part uses A_L (j=0,1) and B_L (j=0,1).
        # Indices match.
        
        # 2x1x2 part uses A_R (j=2) and B_R (j=2).
        # In get_2x1x2, U is (i, 0), V is (0, k), W is (k, i).
        # We need to map U's j=0 to global j=2.
        # V's j=0 to global j=2.
        
        mapped_factors = []
        for f in factors:
            mapped_factors.append(f)
            
        for f in factors_2x1x2:
            u_part, v_part, w_part = f
            u_new = [(i, 2) for (i, j) in u_part]
            v_new = [(2, k) for (j, k) in v_part]
            mapped_factors.append([u_new, v_new, w_part])
            
        return mapped_factors

    def get_2x3x3():
        # A(2x3) B(3x3)
        # A = [A_L(2x2), A_R(2x1)]
        # B = [[B_LL(2x2), B_LR(2x1)], [B_RL(1x2), B_RR(1x1)]]
        
        # A_L B_LL: 2x2x2 (Strassen)
        # A_R B_RL: 2x1x2
        # A_L B_LR: 2x2x1
        # A_R B_RR: 2x1x1
        
        factors_strassen = get_strassen_2x2()
        factors_2x1x2 = get_2x1x2()
        factors_2x2x1 = get_2x2x1()
        factors_2x1x1 = get_2x1x1()
        
        # Local indices for 2x3x3:
        # U: (i, j) j in 0,1,2
        # V: (j, k) j in 0,1,2, k in 0,1,2
        # W: (k, i) k in 0,1,2, i in 0,1
        
        mapped_factors = []
        
        # 1. A_L B_LL (j=0,1 in A, j=0,1 in B, k=0,1 in B)
        for f in factors_strassen:
            mapped_factors.append(f)
            
        # 2. A_R B_RL (j=2 in A, j=2 in B, k=0,1 in B)
        for f in factors_2x1x2:
            u_part, v_part, w_part = f
            u_new = [(i, 2) for (i, j) in u_part]
            v_new = [(2, k) for (j, k) in v_part]
            mapped_factors.append([u_new, v_new, w_part])
            
        # 3. A_L B_LR (j=0,1 in A, j=0,1 in B, k=2 in B)
        for f in factors_2x2x1:
            u_part, v_part, w_part = f
            v_new = [(j, 2) for (j, k) in v_part]
            w_new = [(2, i) for (k, i) in w_part]
            mapped_factors.append([u_part, v_new, w_new])
            
        # 4. A_R B_RR (j=2 in A, j=2 in B, k=2 in B)
        for f in factors_2x1x1:
            u_part, v_part, w_part = f
            u_new = [(i, 2) for (i, j) in u_part]
            v_new = [(2, 2) for (j, k) in v_part]
            w_new = [(2, i) for (k, i) in w_part]
            mapped_factors.append([u_new, v_new, w_new])
            
        return mapped_factors

    # Mapping global blocks to local factors
    
    # Block 1: A1 (rows 0,1) B1 (cols 0,1 of B, which are 5x2)
    # B1 is 5x2. Rows 0,1 (2x2) and 2,3,4 (3x2).
    # A1 is 2x5. Cols 0,1 (2x2) and 2,3,4 (2x3).
    
    # Contribution 1.1: A1_top (2x2) B1_top (2x2) -> 2x2x2
    # Global U: i in 0,1, j in 0,1
    # Global V: j in 0,1, k in 0,1
    # Global W: k in 0,1, i in 0,1
    for f in get_strassen_2x2():
        u_part, v_part, w_part = f
        u_map = { (ui, uj): val for (ui, uj), val in u_part }
        v_map = { (vj, vk): val for (vj, vk), val in v_part }
        w_map = { (wk, wi): val for (wk, wi), val in w_part }
        add_factor(u_map, v_map, w_map, 1, 1, 1)

    # Contribution 1.2: A1_bot (2x3) B1_bot (3x2) -> 2x3x2
    # Global U: i in 0,1, j in 2,3,4
    # Global V: j in 2,3,4, k in 0,1
    # Global W: k in 0,1, i in 0,1
    for f in get_2x3x2():
        u_part, v_part, w_part = f
        # Local j in 0,1,2 -> Global j in 2,3,4
        u_map = { (ui, uj+2): val for (ui, uj), val in u_part }
        v_map = { (vj+2, vk): val for (vj, vk), val in v_part }
        w_map = { (wk, wi): val for (wk, wi), val in w_part }
        add_factor(u_map, v_map, w_map, 1, 1, 1)

    # Block 2: A1 (rows 0,1) B2 (cols 2,3,4 of B, which are 5x3)
    # B2 is 5x3. Rows 0,1 (2x3) and 2,3,4 (3x3).
    # A1 is 2x5. Cols 0,1 (2x2) and 2,3,4 (2x3).
    
    # Contribution 2.1: A1_top (2x2) B2_top (2x3) -> 2x2x3
    # Global U: i in 0,1, j in 0,1
    # Global V: j in 0,1, k in 2,3,4
    # Global W: k in 2,3,4, i in 0,1
    for f in get_2x2x3():
        u_part, v_part, w_part = f
        u_map = { (ui, uj): val for (ui, uj), val in u_part }
        v_map = { (vj, vk+2): val for (vj, vk), val in v_part }
        w_map = { (wk+2, wi): val for (wk, wi), val in w_part }
        add_factor(u_map, v_map, w_map, 1, 1, 1)

    # Contribution 2.2: A1_bot (2x3) B2_bot (3x3) -> 2x3x3
    # Global U: i in 0,1, j in 2,3,4
    # Global V: j in 2,3,4, k in 2,3,4
    # Global W: k in 2,3,4, i in 0,1
    for f in get_2x3x3():
        u_part, v_part, w_part = f
        u_map = { (ui, uj+2): val for (ui, uj), val in u_part }
        v_map = { (vj+2, vk+2): val for (vj, vk), val in v_part }
        w_map = { (wk+2, wi): val for (wk, wi), val in w_part }
        add_factor(u_map, v_map, w_map, 1, 1, 1)

    # Block 3: A2 (rows 2,3) B1 (cols 0,1 of B)
    # Same as Block 1, but U rows offset by 2, W rows offset by 2.
    
    # Contribution 3.1: A2_top (2x2) B1_top (2x2) -> 2x2x2
    for f in get_strassen_2x2():
        u_part, v_part, w_part = f
        u_map = { (ui+2, uj): val for (ui, uj), val in u_part }
        v_map = { (vj, vk): val for (vj, vk), val in v_part }
        w_map = { (wk, wi+2): val for (wk, wi), val in w_part }
        add_factor(u_map, v_map, w_map, 1, 1, 1)

    # Contribution 3.2: A2_bot (2x3) B1_bot (3x2) -> 2x3x2
    for f in get_2x3x2():
        u_part, v_part, w_part = f
        u_map = { (ui+2, uj+2): val for (ui, uj), val in u_part }
        v_map = { (vj+2, vk): val for (vj, vk), val in v_part }
        w_map = { (wk, wi+2): val for (wk, wi), val in w_part }
        add_factor(u_map, v_map, w_map, 1, 1, 1)

    # Block 4: A2 (rows 2,3) B2 (cols 2,3,4 of B)
    # Same as Block 2, but U rows offset by 2, W rows offset by 2.
    
    # Contribution 4.1: A2_top (2x2) B2_top (2x3) -> 2x2x3
    for f in get_2x2x3():
        u_part, v_part, w_part = f
        u_map = { (ui+2, uj): val for (ui, uj), val in u_part }
        v_map = { (vj, vk+2): val for (vj, vk), val in v_part }
        w_map = { (wk+2, wi+2): val for (wk, wi), val in w_part }
        add_factor(u_map, v_map, w_map, 1, 1, 1)

    # Contribution 4.2: A2_bot (2x3) B2_bot (3x3) -> 2x3x3
    for f in get_2x3x3():
        u_part, v_part, w_part = f
        u_map = { (ui+2, uj+2): val for (ui, uj), val in u_part }
        v_map = { (vj+2, vk+2): val for (vj, vk), val in v_part }
        w_map = { (wk+2, wi+2): val for (wk, wi), val in w_part }
        add_factor(u_map, v_map, w_map, 1, 1, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
