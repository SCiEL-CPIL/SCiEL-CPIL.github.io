import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank
    # R(2,2,2) = 7
    # R(2,2,1) = 4
    # R(2,2,3) = 7 + 4 = 11
    # R(2,2,7) = 7 + 7 + 11 = 25
    # R(2,4,7) = 2 * 25 = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # a_indices: list of (idx, coeff) for U
    # b_indices: list of (idx, coeff) for V
    # c_indices: list of (idx, coeff) for W
    def add_term(col, a_terms, b_terms, c_terms):
        for idx, coeff in a_terms:
            U[idx][col] += coeff
        for idx, coeff in b_terms:
            V[idx][col] += coeff
        for idx, coeff in c_terms:
            W[idx][col] += coeff

    # Strassen 2x2x2
    # A: 0,1,2,3 (00, 01, 10, 11)
    # B: 0,1,2,3 (00, 01, 10, 11)
    # C: 0,1,2,3 (00, 01, 10, 11)
    # Terms:
    # 1: (A00+A11)(B00+B11) -> C00, C01, C10, C11
    # 2: (A10+A11)B11
    # 3: A00(B01-B11)
    # 4: A11(B10-B00)
    # 5: (A00+A01)B00
    # 6: (A10+A00)(B11-B01)
    # 7: (A11-A01)(B00+B10)
    
    def add_strassen_222(base_u, base_v, base_w, col_offset):
        # base_u: index offset for A (0..3)
        # base_v: index offset for B (0..3)
        # base_w: index offset for C (0..3)
        
        # Term 1
        add_term(col_offset, 
                 [(base_u+0, 1), (base_u+3, 1)],
                 [(base_v+0, 1), (base_v+3, 1)],
                 [(base_w+0, 1), (base_w+1, 1), (base_w+2, 1), (base_w+3, 1)])
        # Term 2
        add_term(col_offset+1,
                 [(base_u+2, 1), (base_u+3, 1)],
                 [(base_v+3, 1)],
                 [(base_w+0, -1), (base_w+2, 1), (base_w+3, -1)])
        # Term 3
        add_term(col_offset+2,
                 [(base_u+0, 1)],
                 [(base_v+1, 1), (base_v+3, -1)],
                 [(base_w+1, 1), (base_w+3, -1)])
        # Term 4
        add_term(col_offset+3,
                 [(base_u+3, 1)],
                 [(base_v+2, 1), (base_v+0, -1)],
                 [(base_w+0, 1), (base_w+3, -1)])
        # Term 5
        add_term(col_offset+4,
                 [(base_u+0, 1), (base_u+1, 1)],
                 [(base_v+0, 1)],
                 [(base_w+1, -1), (base_w+3, 1)])
        # Term 6
        add_term(col_offset+5,
                 [(base_u+2, 1), (base_u+0, 1)],
                 [(base_v+3, 1), (base_v+1, -1)],
                 [(base_w+0, 1), (base_w+2, -1)])
        # Term 7
        add_term(col_offset+6,
                 [(base_u+3, 1), (base_u+1, -1)],
                 [(base_v+0, 1), (base_v+2, 1)],
                 [(base_w+1, 1), (base_w+2, 1)])

    # Schoolbook 2x2x1
    # A: 0..3, B: 0..1, C: 0..1
    def add_schoolbook_221(base_u, base_v, base_w, col_offset):
        # C0 = A00 B0 + A01 B1
        add_term(col_offset,
                 [(base_u+0, 1)],
                 [(base_v+0, 1)],
                 [(base_w+0, 1)])
        add_term(col_offset+1,
                 [(base_u+1, 1)],
                 [(base_v+1, 1)],
                 [(base_w+0, 1)])
        # C1 = A10 B0 + A11 B1
        add_term(col_offset+2,
                 [(base_u+2, 1)],
                 [(base_v+0, 1)],
                 [(base_w+1, 1)])
        add_term(col_offset+3,
                 [(base_u+3, 1)],
                 [(base_v+1, 1)],
                 [(base_w+1, 1)])

    # Process Block 1: A1 (2x2) * B1 (2x7)
    # A1 indices in U: i in 0..1, j in 0..1.
    # U index: i*4 + j.
    # A1[0,0] -> U[0], A1[0,1] -> U[1], A1[1,0] -> U[4], A1[1,1] -> U[5].
    # Wait, U index is i*m + j. m=4.
    # i=0, j=0 -> 0. i=0, j=1 -> 1.
    # i=1, j=0 -> 4. i=1, j=1 -> 5.
    # So base_u for A1 is 0, but indices are 0,1,4,5.
    # My add_strassen expects contiguous 0,1,2,3.
    # I need to map local indices 0..3 to global indices.
    
    # Let's create a mapping function for U indices for a 2x2 block at (i0, j0)
    def u_idx(i, j):
        return i * m + j

    def v_idx(i, j):
        return i * p + j

    def w_idx(i, j):
        return i * n + j

    # Block 1: A1 (rows 0-1, cols 0-1 of A), B1 (rows 0-1 of B, cols 0-6)
    # A1 local (i,j) -> global (i, j)
    # B1 local (i,j) -> global (i, j)
    # C1 local (i,j) -> global (i, j)
    
    col = 0
    # Sub-block 1a: B1 cols 0-1 (2x2)
    # A1: u_idx(0,0)=0, u_idx(0,1)=1, u_idx(1,0)=4, u_idx(1,1)=5
    # B1: v_idx(0,0)=0, v_idx(0,1)=1, v_idx(1,0)=7, v_idx(1,1)=8
    # C1: w_idx(0,0)=0, w_idx(0,1)=1, w_idx(1,0)=2, w_idx(1,1)=3
    # Note: Strassen assumes A indices 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1).
    # So I need to pass the actual global indices.
    
    # Let's modify add_strassen to take lists of indices.
    # Or just map inside.
    
    # I will refactor to pass index maps.
    
    def add_strassen_222_mapped(u_map, v_map, w_map, col_offset):
        # u_map: [u00, u01, u10, u11]
        # v_map: [v00, v01, v10, v11]
        # w_map: [w00, w01, w10, w11]
        
        add_term(col_offset, 
                 [(u_map[0], 1), (u_map[3], 1)],
                 [(v_map[0], 1), (v_map[3], 1)],
                 [(w_map[0], 1), (w_map[1], 1), (w_map[2], 1), (w_map[3], 1)])
        add_term(col_offset+1,
                 [(u_map[2], 1), (u_map[3], 1)],
                 [(v_map[3], 1)],
                 [(w_map[0], -1), (w_map[2], 1), (w_map[3], -1)])
        add_term(col_offset+2,
                 [(u_map[0], 1)],
                 [(v_map[1], 1), (v_map[3], -1)],
                 [(w_map[1], 1), (w_map[3], -1)])
        add_term(col_offset+3,
                 [(u_map[3], 1)],
                 [(v_map[2], 1), (v_map[0], -1)],
                 [(w_map[0], 1), (w_map[3], -1)])
        add_term(col_offset+4,
                 [(u_map[0], 1), (u_map[1], 1)],
                 [(v_map[0], 1)],
                 [(w_map[1], -1), (w_map[3], 1)])
        add_term(col_offset+5,
                 [(u_map[2], 1), (u_map[0], 1)],
                 [(v_map[3], 1), (v_map[1], -1)],
                 [(w_map[0], 1), (w_map[2], -1)])
        add_term(col_offset+6,
                 [(u_map[3], 1), (u_map[1], -1)],
                 [(v_map[0], 1), (v_map[2], 1)],
                 [(w_map[1], 1), (w_map[2], 1)])

    def add_schoolbook_221_mapped(u_map, v_map, w_map, col_offset):
        # v_map: [v0, v1] for 2x1
        # w_map: [w0, w1] for 2x1
        add_term(col_offset,
                 [(u_map[0], 1)],
                 [(v_map[0], 1)],
                 [(w_map[0], 1)])
        add_term(col_offset+1,
                 [(u_map[1], 1)],
                 [(v_map[1], 1)],
                 [(w_map[0], 1)])
        add_term(col_offset+2,
                 [(u_map[2], 1)],
                 [(v_map[0], 1)],
                 [(w_map[1], 1)])
        add_term(col_offset+3,
                 [(u_map[3], 1)],
                 [(v_map[1], 1)],
                 [(w_map[1], 1)])

    # Block 1: A1 (i=0,1; j=0,1)
    u_map_1 = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    
    # B1a: j=0,1; k=0,1
    v_map_1a = [v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1)]
    w_map_1a = [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)]
    add_strassen_222_mapped(u_map_1, v_map_1a, w_map_1a, col)
    col += 7
    
    # B1b: j=0,1; k=2,3
    v_map_1b = [v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3)]
    w_map_1b = [w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1)]
    add_strassen_222_mapped(u_map_1, v_map_1b, w_map_1b, col)
    col += 7
    
    # B1c: j=0,1; k=4,5,6
    # Split into B1c1 (2x2) and B1c2 (2x1)
    # B1c1: k=4,5
    v_map_1c1 = [v_idx(0,4), v_idx(0,5), v_idx(1,4), v_idx(1,5)]
    w_map_1c1 = [w_idx(4,0), w_idx(4,1), w_idx(5,0), w_idx(5,1)]
    add_strassen_222_mapped(u_map_1, v_map_1c1, w_map_1c1, col)
    col += 7
    
    # B1c2: k=6
    v_map_1c2 = [v_idx(0,6), v_idx(1,6)]
    w_map_1c2 = [w_idx(6,0), w_idx(6,1)]
    add_schoolbook_221_mapped(u_map_1, v_map_1c2, w_map_1c2, col)
    col += 4
    
    # Block 2: A2 (i=0,1; j=2,3)
    u_map_2 = [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)]
    
    # B2a: j=2,3; k=0,1
    v_map_2a = [v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1)]
    w_map_2a = [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)]
    add_strassen_222_mapped(u_map_2, v_map_2a, w_map_2a, col)
    col += 7
    
    # B2b: j=2,3; k=2,3
    v_map_2b = [v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3)]
    w_map_2b = [w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1)]
    add_strassen_222_mapped(u_map_2, v_map_2b, w_map_2b, col)
    col += 7
    
    # B2c: j=2,3; k=4,5,6
    v_map_2c1 = [v_idx(2,4), v_idx(2,5), v_idx(3,4), v_idx(3,5)]
    w_map_2c1 = [w_idx(4,0), w_idx(4,1), w_idx(5,0), w_idx(5,1)]
    add_strassen_222_mapped(u_map_2, v_map_2c1, w_map_2c1, col)
    col += 7
    
    v_map_2c2 = [v_idx(2,6), v_idx(3,6)]
    w_map_2c2 = [w_idx(6,0), w_idx(6,1)]
    add_schoolbook_221_mapped(u_map_2, v_map_2c2, w_map_2c2, col)
    col += 4
    
    return {"U": U, "V": V, "W": W}
