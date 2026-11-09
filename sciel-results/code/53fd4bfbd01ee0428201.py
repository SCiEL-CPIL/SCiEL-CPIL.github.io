"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS # n=2, m=4, p=5
    
    # We will construct a rank 36 decomposition.
    # M_{2,4,5} = M_{2,2,5} (for A cols 0,1 and B rows 0,1) + M_{2,2,5} (for A cols 2,3 and B rows 2,3)
    # M_{2,2,5} = M_{2,2,2} (cols 0,1) + M_{2,2,2} (cols 2,3) + M_{2,2,1} (col 4)
    # Rank M_{2,2,2} = 7 (Strassen)
    # Rank M_{2,2,1} = 4 (Schoolbook)
    # Rank M_{2,2,5} = 7 + 7 + 4 = 18
    # Total Rank = 18 + 18 = 36
    
    rank_222 = 7
    rank_221 = 4
    rank_225 = rank_222 + rank_222 + rank_221 # 18
    total_rank = 2 * rank_225 # 36
    
    # Initialize matrices
    U = [[0.0] * total_rank for _ in range(n * m)] # 2*4 = 8 rows
    V = [[0.0] * total_rank for _ in range(m * p)] # 4*5 = 20 rows
    W = [[0.0] * total_rank for _ in range(p * n)] # 5*2 = 10 rows
    
    # Helper to generate Strassen factors for M_{2,2,2}
    # Local indices: U row 0..3 (A00,A01,A10,A11), V row 0..3 (B00,B01,B10,B11), W row 0..3 (C00,C01,C10,C11)
    # C = A B.
    # M0 = A00 * B00 + A11 * B11
    # M1 = (A10 + A00) * B00
    # M2 = (A01 + A11) * B11
    # M3 = A00 * (B10 + B00)
    # M4 = A11 * (B01 + B11)
    # M5 = (A10 + A11) * (B11 + B01)
    # M6 = (A00 + A01) * (B01 + B00)
    # C00 = M0 + M1 + M2 - M4 - M5
    # C01 = M3 + M2 - M4 - M6
    # C10 = M5 + M1 - M3 - M6
    # C11 = M0 + M6 - M1 - M2
    
    def get_strassen_factors():
        # Returns U, V, W parts of shape (4, 7), (4, 7), (4, 7) for local M_{2,2,2}
        u_part = [[0.0] * 7 for _ in range(4)]
        v_part = [[0.0] * 7 for _ in range(4)]
        w_part = [[0.0] * 7 for _ in range(4)]
        
        # Col 0: M0 = A00*B00 + A11*B11
        u_part[0][0] = 1; v_part[0][0] = 1; w_part[0][0] = 1
        u_part[3][0] = 1; v_part[3][0] = 1; w_part[3][0] = 1 # A11, B11 -> C11
        
        # Wait, Strassen coefficients for C:
        # C00: +M0, +M1, +M2, -M4, -M5
        # C01: +M3, +M2, -M4, -M6
        # C10: +M5, +M1, -M3, -M6
        # C11: +M0, +M6, -M1, -M2
        
        # Let's re-verify the standard Strassen scheme.
        # M1 = (A00 + A11) B00 -> contributes to C00?
        # Standard:
        # M1 = (A00 + A11)B00
        # M2 = A11(B01 - B00)
        # M3 = (A00 + A01)B11
        # M4 = (A10 + A11)B01
        # M5 = (A00 - A10)(B00 + B01)
        # M6 = (A01 - A11)(B10 + B11)
        # M7 = (A00 + A01)(B00 - B10)
        # C00 = M1 - M5 + M7
        # C01 = M3 + M6
        # C10 = M2 + M5
        # C11 = M1 - M3 + M4
        # This is one variant. Let's use a consistent one.
        
        # Simpler variant often used:
        # M0 = A00B00
        # M1 = (A10+A00)B00
        # M2 = (A01+A11)B11
        # M3 = A00(B10+B00)
        # M4 = A11(B01+B11)
        # M5 = (A10+A11)(B11+B01)
        # M6 = (A00+A01)(B01+B00)
        
        # Recalculating W parts for this variant:
        # C00 = M0 + M1 + M2 - M4 - M5  ?? 
        # Let's check C00 = A00B00 + A01B10.
        # My previous derivation was messy. Let's use a verified set.
        
        # Verified Strassen:
        # p1 = a(g-e)
        # p2 = (a+b)e
        # p3 = (c+d)f
        # p4 = d(h-f)
        # p5 = (a+d)(e+h)
        # p6 = (b-d)(f+h)
        # p7 = (a-c)(e+f)
        # c = p5+p4-p6-p7
        # d = p1+p7
        # e = p1+p3
        # f = p2+p3-p5-p1
        
        # Mapping to our indices A00,A01,A10,A11 and B00,B01,B10,B11:
        # a=A00, b=A01, c=A10, d=A11
        # e=B00, f=B01, g=B10, h=B11
        
        # p1 = A00 * (B10 - B00)
        # p2 = (A00 + A01) * B00
        # p3 = (A10 + A11) * B01
        # p4 = A11 * (B11 - B01)
        # p5 = (A00 + A11) * (B00 + B11)
        # p6 = (A01 - A11) * (B01 + B11)
        # p7 = (A00 - A10) * (B00 + B01)
        
        # C00 (c) = p5 + p4 - p6 - p7
        # C01 (d) = p1 + p7
        # C10 (e) = p1 + p3
        # C11 (f) = p2 + p3 - p5 - p1
        
        u_part = [
            [0, 1, 1, 0, 1, 0, 1], # A00: p2(1), p5(1), p7(1) -- wait, p7 has (A00-A10), so A00 coeff is 1.
            [0, 0, 0, 0, 0, 0, 0], # A01: Not in p1? p2 has A01. 
            # Let's fill carefully.
        ]
        
        # p1: A00 * 1, B10 * 1, B00 * -1
        # p2: A00 * 1, A01 * 1, B00 * 1
        # p3: A10 * 1, A11 * 1, B01 * 1
        # p4: A11 * 1, B11 * 1, B01 * -1
        # p5: A00 * 1, A11 * 1, B00 * 1, B11 * 1
        # p6: A01 * 1, A11 * -1, B01 * 1, B11 * 1
        # p7: A00 * 1, A10 * -1, B00 * 1, B01 * 1
        
        u_s = [
            [0, 1, 0, 0, 1, 0, 1], # A00
            [0, 1, 0, 0, 0, 1, 0], # A01
            [0, 0, 1, 0, 0, 0, -1], # A10
            [0, 0, 1, 1, 1, -1, 0]  # A11
        ]
        v_s = [
            [1, -1, 0, 0, 1, 0, 1], # B00
            [0, 0, 1, -1, 0, 1, 1], # B01
            [1, 0, 0, 0, 0, 0, 0], # B10
            [0, 0, 0, 1, 1, 1, 0]  # B11
        ]
        w_s = [
            [0, 0, 0, 0, 1, -1, -1], # C00 (p5+p4-p6-p7)
            [1, 0, 0, 0, 0, 0, 1],   # C01 (p1+p7)
            [1, 0, 1, 0, 0, 0, 0],   # C10 (p1+p3)
            [0, 1, 1, 0, -1, -1, 0]  # C11 (p2+p3-p5-p1)
        ]
        return u_s, v_s, w_s

    # Helper to generate Schoolbook factors for M_{2,2,1}
    # A(2x2) * B(2x1) -> C(2x1)
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Rank 4
    def get_schoolbook_221_factors():
        u_b = [
            [1, 0, 0, 0], # A00
            [0, 1, 0, 0], # A01
            [0, 0, 1, 0], # A10
            [0, 0, 0, 1]  # A11
        ]
        v_b = [
            [1, 0, 1, 0], # B0 (in term 1 and 3)
            [0, 1, 0, 1]  # B1 (in term 2 and 4)
        ]
        w_b = [
            [1, 1, 0, 0], # C0 (terms 1, 2)
            [0, 0, 1, 1]  # C1 (terms 3, 4)
        ]
        return u_b, v_b, w_b

    # Build M_{2,2,5} factors (Rank 18)
    # Local U: 4 rows (A)
    # Local V: 10 rows (B: 2 rows, 5 cols)
    # Local W: 10 rows (C: 5 cols, 2 rows) -> Index k*2+i
    
    u_225 = [[0.0] * 18 for _ in range(4)]
    v_225 = [[0.0] * 18 for _ in range(10)]
    w_225 = [[0.0] * 18 for _ in range(10)]
    
    col_offset = 0
    u_s, v_s, w_s = get_strassen_factors()
    
    # Block 1: B cols 0,1
    for r in range(4):
        for c in range(7):
            u_225[r][col_offset + c] = u_s[r][c]
    # Map V local (4 rows: B00,B01,B10,B11) to global V (10 rows: j*5+k)
    # Local 0 (B00) -> Global 0 (0*5+0)
    # Local 1 (B01) -> Global 1 (0*5+1)
    # Local 2 (B10) -> Global 5 (1*5+0)
    # Local 3 (B11) -> Global 6 (1*5+1)
    v_map_1 = [0, 1, 5, 6]
    for r in range(4):
        for c in range(7):
            v_225[v_map_1[r]][col_offset + c] = v_s[r][c]
    # Map W local (4 rows: C00,C01,C10,C11) to global W (10 rows: k*2+i)
    # Local 0 (C00) -> Global 0 (0*2+0)
    # Local 1 (C01) -> Global 1 (0*2+1)
    # Local 2 (C10) -> Global 2 (1*2+0)
    # Local 3 (C11) -> Global 3 (1*2+1)
    w_map_1 = [0, 1, 2, 3]
    for r in range(4):
        for c in range(7):
            w_225[w_map_1[r]][col_offset + c] = w_s[r][c]
            
    col_offset += 7
    
    # Block 2: B cols 2,3
    for r in range(4):
        for c in range(7):
            u_225[r][col_offset + c] = u_s[r][c]
    # Local B indices map to global cols 2,3
    # Local 0 (B00) -> Global B0,2 -> 0*5+2 = 2
    # Local 1 (B01) -> Global B0,3 -> 0*5+3 = 3
    # Local 2 (B10) -> Global B1,2 -> 1*5+2 = 7
    # Local 3 (B11) -> Global B1,3 -> 1*5+3 = 8
    v_map_2 = [2, 3, 7, 8]
    for r in range(4):
        for c in range(7):
            v_225[v_map_2[r]][col_offset + c] = v_s[r][c]
    # W local C indices k=2,3
    # Local 0 (C00 of block) -> Global C0,2 -> 2*2+0 = 4
    # Local 1 (C01 of block) -> Global C0,3 -> 3*2+0 = 6? No, W index is k*n+i.
    # Global W row index = k_global * 2 + i_global.
    # Block 2 produces C for k=2,3.
    # Local C00 (i=0, k_local=0) -> Global C0,2 (i=0, k=2) -> 2*2+0 = 4
    # Local C01 (i=1, k_local=0) -> Global C1,2 (i=1, k=2) -> 2*2+1 = 5
    # Local C10 (i=0, k_local=1) -> Global C0,3 (i=0, k=3) -> 3*2+0 = 6
    # Local C11 (i=1, k_local=1) -> Global C1,3 (i=1, k=3) -> 3*2+1 = 7
    w_map_2 = [4, 5, 6, 7]
    for r in range(4):
        for c in range(7):
            w_225[w_map_2[r]][col_offset + c] = w_s[r][c]
            
    col_offset += 7
    
    # Block 3: B col 4 (M_{2,2,1})
    u_b, v_b, w_b = get_schoolbook_221_factors()
    for r in range(4):
        for c in range(4):
            u_225[r][col_offset + c] = u_b[r][c]
    # Local B indices (2 rows: B0, B1) map to global B col 4
    # Local 0 (B0) -> Global B0,4 -> 0*5+4 = 4
    # Local 1 (B1) -> Global B1,4 -> 1*5+4 = 9
    v_map_3 = [4, 9]
    for r in range(2):
        for c in range(4):
            v_225[v_map_3[r]][col_offset + c] = v_b[r][c]
    # Local C indices (2 rows: C0, C1) map to global C col 4
    # Local 0 (C0) -> Global C0,4 (i=0, k=4) -> 4*2+0 = 8
    # Local 1 (C1) -> Global C1,4 (i=1, k=4) -> 4*2+1 = 9
    w_map_3 = [8, 9]
    for r in range(2):
        for c in range(4):
            w_225[w_map_3[r]][col_offset + c] = w_b[r][c]
            
    # Now assemble M_{2,4,5} from two M_{2,2,5} blocks
    # Block A: A cols 0,1, B rows 0,1
    # Block B: A cols 2,3, B rows 2,3
    
    # Block A uses u_225, v_225, w_225
    # Global U index: i*4+j
    # Block A U rows 0..3 map to Global U rows 0,1,4,5
    u_map_A = [0, 1, 4, 5]
    for r in range(4):
        for c in range(18):
            U[u_map_A[r]][c] = u_225[r][c]
            
    # Block A V rows 0..9 map to Global V rows 0..9 (B rows 0,1)
    # Global V index j*5+k
    # Block A B rows 0,1.
    # Local V row l (0..9) corresponds to B_{local_row, local_col}.
    # Global V row is global_j * 5 + global_k.
    # Since Block A uses B rows 0,1 and cols 0..4,
    # Local V index is just the same as Global V index for j=0,1.
    # Local V row 0 (B0,0) -> Global 0.
    # Local V row 4 (B0,4) -> Global 4.
    # Local V row 5 (B1,0) -> Global 5.
    # So V map is identity for rows 0..9.
    for r in range(10):
        for c in range(18):
            V[r][c] = v_225[r][c]
            
    # Block A W rows 0..9 map to Global W rows 0..9
    # Global W index k*2+i.
    # Block A produces C for all i,k.
    # So W map is identity.
    for r in range(10):
        for c in range(18):
            W[r][c] = w_225[r][c]
            
    # Block B uses u_225, v_225, w_225 but shifted.
    col_offset_B = 18
    # Global U index: i*4+j
    # Block B U rows 0..3 map to A cols 2,3.
    # A_{0,2} (i=0,j=2) -> Global U 0*4+2 = 2
    # A_{0,3} (i=0,j=3) -> Global U 3
    # A_{1,2} (i=1,j=2) -> Global U 6
    # A_{1,3} (i=1,j=3) -> Global U 7
    u_map_B = [2, 3, 6, 7]
    for r in range(4):
        for c in range(18):
            U[u_map_B[r]][col_offset_B + c] = u_225[r][c]
            
    # Block B V rows 0..9 map to Global V rows 10..19 (B rows 2,3)
    # Local V row l (0..9) corresponds to B_{local_j, local_k}.
    # Global V row is global_j * 5 + global_k.
    # local_j 0 -> global_j 2. local_j 1 -> global_j 3.
    # Global row = (local_j + 2) * 5 + local_k = local_j*5 + local_k + 10.
    # So V map is offset by 10.
    for r in range(10):
        for c in range(18):
            V[r + 10][col_offset_B + c] = v_225[r][c]
            
    # Block B W rows 0..9 map to Global W rows 0..9
    # The output C is the same, so we add to existing W.
    for r in range(10):
        for c in range(18):
            W[r][col_offset_B + c] = w_225[r][c]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
