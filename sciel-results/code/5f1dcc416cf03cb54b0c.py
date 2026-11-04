import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try Winograd F(3) based approach first, fall back to Strassen if it fails
    try:
        result = _winograd_construct()
        if result is not None:
            return result
    except Exception:
        pass
    
    # Fallback to Strassen approach (rank 77)
    result = _strassen_construct()
    if result is None:
        # Emergency fallback: return schoolbook decomposition (rank 96)
        return _schoolbook_construct()
    return result

def _schoolbook_construct():
    """Fallback schoolbook decomposition with rank 96."""
    n, m, p = 4, 4, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _winograd_construct():
    """Winograd F(3) based decomposition for 4x4 * 4x6.
    
    Uses overlapping 3x3 Winograd F(3) blocks to reduce rank below 77.
    """
    n, m, p = 4, 4, 6
    
    # Winograd F(3) for 3x3*3x3 has rank 23.
    # We'll use 4 overlapping 3x3 windows for A (rows/cols 0-2 and 1-3)
    # and for B, we'll use windows that cover the 4x6 shape.
    
    # For A (4x4), 3x3 windows:
    # A_w0: rows 0-2, cols 0-2
    # A_w1: rows 0-2, cols 1-3
    # A_w2: rows 1-3, cols 0-2
    # A_w3: rows 1-3, cols 1-3
    
    # For B (4x6), 3x3 windows (need to cover cols 0-5):
    # B_w0: rows 0-2, cols 0-2
    # B_w1: rows 0-2, cols 1-3
    # B_w2: rows 0-2, cols 2-4
    # B_w3: rows 0-2, cols 3-5
    # B_w4: rows 1-3, cols 0-2
    # B_w5: rows 1-3, cols 1-3
    # B_w6: rows 1-3, cols 2-4
    # B_w7: rows 1-3, cols 3-5
    
    # For C (4x6), 3x3 windows:
    # C_w0: rows 0-2, cols 0-2
    # C_w1: rows 0-2, cols 1-3
    # C_w2: rows 0-2, cols 2-4
    # C_w3: rows 0-2, cols 3-5
    # C_w4: rows 1-3, cols 0-2
    # C_w5: rows 1-3, cols 1-3
    # C_w6: rows 1-3, cols 2-4
    # C_w7: rows 1-3, cols 3-5
    
    # This approach is too complex and likely won't reduce rank significantly
    # due to the overlapping nature. Let me return None to use Strassen.
    return None

def _strassen_construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Rank is 7 * (7 + 4) = 77
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for top level
    # M_s = A_s B_s
    # A_s = sum c_a A_a
    # B_s = sum d_b B_b
    # a, b in {11, 12, 21, 22}
    
    # Mapping of block names to indices
    # A blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B blocks:
    # B11: rows 0-1, cols 0-2
    # B12: rows 0-1, cols 3-5
    # B21: rows 2-3, cols 0-2
    # B22: rows 2-3, cols 3-5
    
    # C blocks:
    # C11: rows 0-1, cols 0-2
    # C12: rows 0-1, cols 3-5
    # C21: rows 2-3, cols 0-2
    # C22: rows 2-3, cols 3-5
    
    # Try to reduce rank by using Winograd F(3) for 3x3 subproblems
    # within the Strassen framework. For each Strassen product, instead of
    # using 7 mults for 2x2*2x2, we could potentially use fewer if we
    # batch operations.
    
    # Actually, let's try a different approach: use the fact that B is 4x6
    # and decompose it as 4x3 + 4x3, then use Winograd F(3) for each 4x4*4x3.
    
    # For 4x4 * 4x3:
    # Split A into 2x2 blocks (4 blocks), B into 2x2 blocks (2 blocks of 2x2) 
    # Wait, 4x3 can be split as 2x3 + 2x3, and each 2x3 as 2x2 + 2x1.
    # This is getting complicated. Let me just keep the current approach.
    
    # Helper to get global index for A
    def idx_A(i, j):
        return i * m + j
        
    # Helper to get global index for B
    def idx_B(j, k):
        return j * p + k
        
    # Helper to get global index for C
    def idx_C(i, k):
        return k * n + i
        
    # Define the 7 Strassen products for top level
    # Each product s has:
    # A_coeffs: dict {block_name: coeff}
    # B_coeffs: dict {block_name: coeff}
    
    strassen_defs = [
        # s=1
        ({"11": 1, "22": 1}, {"11": 1, "22": 1}),
        # s=2
        ({"21": 1, "22": 1}, {"11": 1}),
        # s=3
        ({"11": 1}, {"12": 1, "22": -1}),
        # s=4
        ({"22": 1}, {"21": 1, "11": -1}),
        # s=5
        ({"11": 1, "12": 1}, {"22": 1}),
        # s=6
        ({"21": 1, "11": -1}, {"11": 1, "12": 1}),
        # s=7
        ({"12": 1, "22": -1}, {"21": 1, "22": 1}),
    ]
    
    # For each s, we compute the local A_s (2x2) and B_s (2x3)
    # A_s[i][j] is sum of c_a * A_a[i][j]
    # B_s[i][k] is sum of d_b * B_b[i][k]
    
    # We need to generate terms for A_s * B_s
    # B_s is split into B_left (2x2) and B_right (2x1)
    # B_left[i][k] for k in 0,1
    # B_right[i] for i in 0,1 (corresponds to k=2 in B_s)
    
    for s_idx, (a_coeffs, b_coeffs) in enumerate(strassen_defs):
        if time.time() > deadline:
            break
        s = s_idx + 1
        
        # Construct local A_s entries as linear combos of global A
        # A_s[i][j] where i,j in 0..1
        # A_a[i][j] corresponds to global A at (row_offset + i, col_offset + j)
        # Block 11: offset (0,0)
        # Block 12: offset (0,2)
        # Block 21: offset (2,0)
        # Block 22: offset (2,2)
        
        a_local = [[None for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                # a_local[i][j] is a dict of global_A_index -> coeff
                a_local[i][j] = {}
                for block, c in a_coeffs.items():
                    if block == "11":
                        r, c_off = 0, 0
                    elif block == "12":
                        r, c_off = 0, 2
                    elif block == "21":
                        r, c_off = 2, 0
                    else: # 22
                        r, c_off = 2, 2
                    gi = r + i
                    gj = c_off + j
                    g_idx = idx_A(gi, gj)
                    a_local[i][j][g_idx] = a_local[i][j].get(g_idx, 0) + c
                    
        # Construct local B_s entries
        # B_s[i][k] where i in 0..1, k in 0..2
        # Block 11: rows 0-1, cols 0-2
        # Block 12: rows 0-1, cols 3-5
        # Block 21: rows 2-3, cols 0-2
        # Block 22: rows 2-3, cols 3-5
        
        b_local = [[None for _ in range(3)] for _ in range(2)]
        for i in range(2):
            for k in range(3):
                b_local[i][k] = {}
                for block, d in b_coeffs.items():
                    if block == "11":
                        r, c_off = 0, 0
                    elif block == "12":
                        r, c_off = 0, 3
                    elif block == "21":
                        r, c_off = 2, 0
                    else: # 22
                        r, c_off = 2, 3
                    gj = r + i
                    gk = c_off + k
                    g_idx = idx_B(gj, gk)
                    b_local[i][k][g_idx] = b_local[i][k].get(g_idx, 0) + d
                    
        # Now decompose A_s * B_s
        # B_s = [B_left (2x2), B_right (2x1)]
        # B_left[i][k] = b_local[i][k] for k=0,1
        # B_right[i] = b_local[i][2]
        
        # Part 1: A_s * B_left (2x2 * 2x2) -> 7 Strassen terms
        # A_s is 2x2.
        # u11 = a_local[0][0], u12 = a_local[0][1]
        # u21 = a_local[1][0], u22 = a_local[1][1]
        # v11 = b_local[0][0], v12 = b_local[0][1]
        # v21 = b_local[1][0], v22 = b_local[1][1]
        
        # Strassen terms for 2x2:
        # P1 = (u11+u22)(v11+v22) -> C11 += P1, C22 += P1
        # P2 = (u21+u22)v11 -> C21 += P2, C22 -= P2
        # P3 = u11(v12-v22) -> C12 += P3, C22 += P3
        # P4 = u22(v21-v11) -> C11 += P4, C21 += P4
        # P5 = (u11+u12)v22 -> C11 -= P5, C12 += P5
        # P6 = (u21-u11)(v11+v12) -> C22 += P6
        # P7 = (u12-u22)(v21+v22) -> C11 += P7
        
        # We need to add terms to the global U, V, W.
        # Each term is a product of a linear combo of A and a linear combo of B.
        # The result contributes to specific entries of M_s (which is 2x3).
        # M_s[i][k] for k=0,1 comes from this part.
        # M_s[0][0] (C11 local), M_s[0][1] (C12 local)
        # M_s[1][0] (C21 local), M_s[1][1] (C22 local)
        
        # Helper to add a term defined by linear combos
        def add_term_from_combos(u_combo, v_combo, w_targets):
            # u_combo: dict global_A_idx -> coeff
            # v_combo: dict global_B_idx -> coeff
            # w_targets: list of (global_C_idx, coeff)
            nonlocal col
            # Check if u_combo or v_combo is empty
            if not u_combo or not v_combo:
                return
            
            # This term is rank 1 in the tensor sense?
            # Yes, it's sum_{a,b} u_a v_b C_{ab}
            # Wait, the tensor decomposition is sum_r U_r V_r W_r
            # If U_r has multiple non-zeros and V_r has multiple non-zeros, it's still a valid term
            # as long as it corresponds to a single multiplication of linear combos.
            
            for u_idx, u_val in u_combo.items():
                if u_val != 0:
                    U[u_idx][col] += u_val
            for v_idx, v_val in v_combo.items():
                if v_val != 0:
                    V[v_idx][col] += v_val
            for w_idx, w_val in w_targets:
                if w_val != 0:
                    W[w_idx][col] += w_val
            col += 1

        # We need to map local M_s[i][k] to global C indices.
        # M_s is the s-th Strassen product.
        # We determined earlier how M_s maps to C blocks.
        # Let's define a function to get global C indices for local M_s[i][k] with coefficient alpha.
        
        def get_w_targets(local_i, local_k, alpha):
            # local_i in 0..1, local_k in 0..2
            # Returns list of (global_C_idx, coeff)
            targets = []
            
            # C block mapping:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # M_s contributes to C blocks based on s.
            # For each block C_xy, if M_s is in the sum, add contribution.
            
            # Determine C block offsets
            # C11: row_off 0, col_off 0
            # C12: row_off 0, col_off 3
            # C21: row_off 2, col_off 0
            # C22: row_off 2, col_off 3
            
            # For a given local_i, local_k, the global C index is:
            # C_block[i + row_off][k + col_off]
            
            # We need to know which blocks M_s contributes to.
            blocks = []
            if s == 1:
                blocks = [("11", 1), ("22", 1)]
            elif s == 2:
                blocks = [("21", 1), ("22", -1)]
            elif s == 3:
                blocks = [("12", 1), ("22", 1)]
            elif s == 4:
                blocks = [("11", 1), ("21", 1)]
            elif s == 5:
                blocks = [("11", -1), ("12", 1)]
            elif s == 6:
                blocks = [("22", 1)]
            elif s == 7:
                blocks = [("11", 1)]
            
            for block_name, coeff in blocks:
                if block_name == "11":
                    r_off, c_off = 0, 0
                elif block_name == "12":
                    r_off, c_off = 0, 3
                elif block_name == "21":
                    r_off, c_off = 2, 0
                else:
                    r_off, c_off = 2, 3
                
                g_i = local_i + r_off
                g_k = local_k + c_off
                g_idx = idx_C(g_i, g_k)
                targets.append((g_idx, alpha * coeff))
                
            return targets

        # Define Strassen terms for 2x2
        # u11, u12, u21, u22 are dicts
        u11 = a_local[0][0]
        u12 = a_local[0][1]
        u21 = a_local[1][0]
        u22 = a_local[1][1]
        
        v11 = b_local[0][0]
        v12 = b_local[0][1]
        v21 = b_local[1][0]
        v22 = b_local[1][1]
        
        # P1
        u_p1 = {k: v for k, v in u11.items()}
        for k, v in u22.items(): u_p1[k] = u_p1.get(k, 0) + v
        v_p1 = {k: v for k, v in v11.items()}
        for k, v in v22.items(): v_p1[k] = v_p1.get(k, 0) + v
        # Contributes to M_s[0][0] and M_s[1][1]
        w1 = get_w_targets(0, 0, 1)
        w2 = get_w_targets(1, 1, 1)
        # Combine w targets
        w_targets = w1 + w2
        # Merge w_targets if indices overlap
        merged_w = {}
        for idx, val in w_targets:
            merged_w[idx] = merged_w.get(idx, 0) + val
        add_term_from_combos(u_p1, v_p1, list(merged_w.items()))
        
        # P2
        u_p2 = {k: v for k, v in u21.items()}
        for k, v in u22.items(): u_p2[k] = u_p2.get(k, 0) + v
        v_p2 = v11
        # Contributes to M_s[1][0] and M_s[1][1] with -1
        w1 = get_w_targets(1, 0, 1)
        w2 = get_w_targets(1, 1, -1)
        w_targets = w1 + w2
        merged_w = {}
        for idx, val in w_targets:
            merged_w[idx] = merged_w.get(idx, 0) + val
        add_term_from_combos(u_p2, v_p2, list(merged_w.items()))
        
        # P3
        u_p3 = u11
        v_p3 = {k: v for k, v in v12.items()}
        for k, v in v22.items(): v_p3[k] = v_p3.get(k, 0) - v
        # Contributes to M_s[0][1] and M_s[1][1]
        w1 = get_w_targets(0, 1, 1)
        w2 = get_w_targets(1, 1, 1)
        w_targets = w1 + w2
        merged_w = {}
        for idx, val in w_targets:
            merged_w[idx] = merged_w.get(idx, 0) + val
        add_term_from_combos(u_p3, v_p3, list(merged_w.items()))
        
        # P4
        u_p4 = u22
        v_p4 = {k: v for k, v in v21.items()}
        for k, v in v11.items(): v_p4[k] = v_p4.get(k, 0) - v
        # Contributes to M_s[0][0] and M_s[1][0]
        w1 = get_w_targets(0, 0, 1)
        w2 = get_w_targets(1, 0, 1)
        w_targets = w1 + w2
        merged_w = {}
        for idx, val in w_targets:
            merged_w[idx] = merged_w.get(idx, 0) + val
        add_term_from_combos(u_p4, v_p4, list(merged_w.items()))
        
        # P5
        u_p5 = {k: v for k, v in u11.items()}
        for k, v in u12.items(): u_p5[k] = u_p5.get(k, 0) + v
        v_p5 = v22
        # Contributes to M_s[0][0] with -1 and M_s[0][1] with 1
        w1 = get_w_targets(0, 0, -1)
        w2 = get_w_targets(0, 1, 1)
        w_targets = w1 + w2
        merged_w = {}
        for idx, val in w_targets:
            merged_w[idx] = merged_w.get(idx, 0) + val
        add_term_from_combos(u_p5, v_p5, list(merged_w.items()))
        
        # P6
        u_p6 = {k: v for k, v in u21.items()}
        for k, v in u11.items(): u_p6[k] = u_p6.get(k, 0) - v
        v_p6 = {k: v for k, v in v11.items()}
        for k, v in v12.items(): v_p6[k] = v_p6.get(k, 0) + v
        # Contributes to M_s[1][1]
        w1 = get_w_targets(1, 1, 1)
        w_targets = w1
        merged_w = {}
        for idx, val in w_targets:
            merged_w[idx] = merged_w.get(idx, 0) + val
        add_term_from_combos(u_p6, v_p6, list(merged_w.items()))
        
        # P7
        u_p7 = {k: v for k, v in u12.items()}
        for k, v in u22.items(): u_p7[k] = u_p7.get(k, 0) - v
        v_p7 = {k: v for k, v in v21.items()}
        for k, v in v22.items(): v_p7[k] = v_p7.get(k, 0) + v
        # Contributes to M_s[0][0]
        w1 = get_w_targets(0, 0, 1)
        w_targets = w1
        merged_w = {}
        for idx, val in w_targets:
            merged_w[idx] = merged_w.get(idx, 0) + val
        add_term_from_combos(u_p7, v_p7, list(merged_w.items()))
        
        # Part 2: A_s * B_right (2x2 * 2x1) -> 4 Schoolbook terms
        # B_right is column k=2 of B_s
        # b_r0 = b_local[0][2]
        # b_r1 = b_local[1][2]
        
        # Q1 = u11 * b_r0 -> M_s[0][2]
        # Q2 = u12 * b_r1 -> M_s[0][2]
        # Q3 = u21 * b_r0 -> M_s[1][2]
        # Q4 = u22 * b_r1 -> M_s[1][2]
        
        br0 = b_local[0][2]
        br1 = b_local[1][2]
        
        # Q1
        w1 = get_w_targets(0, 2, 1)
        merged_w = {}
        for idx, val in w1:
            merged_w[idx] = merged_w.get(idx, 0) + val
        add_term_from_combos(u11, br0, list(merged_w.items()))
        
        # Q2
        w1 = get_w_targets(0, 2, 1)
        merged_w = {}
        for idx, val in w1:
            merged_w[idx] = merged_w.get(idx, 0) + val
        add_term_from_combos(u12, br1, list(merged_w.items()))
        
        # Q3
        w1 = get_w_targets(1, 2, 1)
        merged_w = {}
        for idx, val in w1:
            merged_w[idx] = merged_w.get(idx, 0) + val
        add_term_from_combos(u21, br0, list(merged_w.items()))
        
        # Q4
        w1 = get_w_targets(1, 2, 1)
        merged_w = {}
        for idx, val in w1:
            merged_w[idx] = merged_w.get(idx, 0) + val
        add_term_from_combos(u22, br1, list(merged_w.items()))

    # If we didn't complete all Strassen products, the result is invalid
    if col < rank:
        return None
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
