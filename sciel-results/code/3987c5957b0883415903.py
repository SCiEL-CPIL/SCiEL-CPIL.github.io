import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank will be sum of sub-ranks
    # We will build U, V, W as lists of lists, then convert to numpy if needed, 
    # but the interface expects lists of lists or similar.
    # The seed uses lists of lists.
    
    # Helper to add a term
    # u: list of length n*m
    # v: list of length m*p
    # w: list of length p*n
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)

    # --- Subproblem 1: 2x3 x 3x5 ---
    # Split into 2x3 x 3x3 and 2x3 x 3x2
    
    # 1a: 2x3 x 3x3 (Rank 18 Schoolbook)
    # A is 2x3, B is 3x3.
    # A indices: 0..5 (2*3)
    # B indices: 0..8 (3*3)
    # C indices: 0..5 (3*2) -> k*n+i. k in 0..2, i in 0..1.
    for i in range(2):
        for j in range(3):
            for k in range(3):
                u = [0]*10
                v = [0]*15
                w = [0]*10
                u[i*3 + j] = 1
                v[j*3 + k] = 1
                w[k*2 + i] = 1
                add_term(u, v, w)
                
    # 1b: 2x3 x 3x2 (Rank 12 Schoolbook)
    # A is 2x3 (cols 3,4,5 of original A? No, A is split into A3 and A2.
    # Original A is 2x5. A3 is cols 0,1,2. A2 is cols 3,4.
    # Original B is 5x5. B3 is rows 0,1,2. B2 is rows 3,4.
    # So for 1b, A is still A3 (cols 0,1,2). B is B32 (rows 0,1,2, cols 3,4 of B? No.
    # B3 is 3x5. B32 is the right part of B3, so 3x2 (cols 3,4 of B).
    # So B indices for this subproblem are local 0..5 (3*2).
    # But we need to map to global B indices.
    # Global B index for row r, col c is r*5+c.
    # For B32, r in 0..2, c in 3..4.
    
    for i in range(2):
        for j in range(3):
            for k in range(2):
                u = [0]*10
                v = [0]*25
                w = [0]*10
                u[i*3 + j] = 1
                # v index: row j, col (3+k)
                v[j*5 + (3+k)] = 1
                # w index: output col (3+k), row i
                # Output C is 2x5. Col index is 3+k.
                # w index: k_out * n + i.
                w[(3+k)*2 + i] = 1
                add_term(u, v, w)

    # --- Subproblem 2: 2x2 x 2x5 ---
    # A2 is cols 3,4 of A.
    # B2 is rows 3,4 of B.
    
    # 2a: 2x2 x 2x2 (Rank 7 Strassen)
    # A2 indices: 6,7,8,9 (2x2).
    # B22 indices: rows 3,4, cols 3,4.
    # Output C22: rows 0,1, cols 3,4.
    
    # Strassen for 2x2x2
    # Local indices:
    # A: 0,1,2,3 -> a,b,c,d
    # B: 0,1,2,3 -> e,f,g,h
    # C: 0,1,2,3 -> c00,c01,c10,c11
    
    # We will map local to global.
    # Local A idx la -> Global A idx 6+la
    # Local B idx lb -> Global B idx 15+lb (3*5+3 = 18? No. 3*5+3=18. 3*5+4=19. 4*5+3=23. 4*5+4=24)
    # Wait, B22 is rows 3,4 cols 3,4.
    # Global B index: r*5+c.
    # (3,3)->18, (3,4)->19, (4,3)->23, (4,4)->24.
    
    # Local C idx lc -> Global C index.
    # C22 is rows 0,1 cols 3,4.
    # (0,3)->3, (0,4)->4, (1,3)->5, (1,4)->6.
    
    def add_strassen_222(global_a_offset, global_b_map, global_c_map):
        # global_a_offset: base index for A
        # global_b_map: list of 4 global indices for B
        # global_c_map: list of 4 global indices for C
        
        terms = [
            # (u, v, w)
            ([1,0,0,1], [1,1,0,0], [1,0,0,-1]), # M1
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),  # M2
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]), # M3
            ([0,0,0,1], [0,-1,1,0], [0,0,1,0]), # M4
            ([1,1,0,0], [0,0,0,1], [0,-1,0,0]), # M5
            ([1,-1,0,0], [1,1,0,0], [0,0,0,1]), # M6
            ([0,1,0,-1], [0,0,1,1], [0,0,0,1])  # M7
        ]
        
        for u_loc, v_loc, w_loc in terms:
            u = [0]*10
            v = [0]*25
            w = [0]*10
            for i in range(4):
                if u_loc[i] != 0:
                    u[global_a_offset + i] = u_loc[i]
                if v_loc[i] != 0:
                    v[global_b_map[i]] = v_loc[i]
                if w_loc[i] != 0:
                    w[global_c_map[i]] = w_loc[i]
            add_term(u, v, w)

    # Map for B22
    b_map_222 = [18, 19, 23, 24]
    # Map for C22
    c_map_222 = [3, 4, 5, 6]
    add_strassen_222(6, b_map_222, c_map_222)

    # 2b: 2x2 x 2x3 (Rank 12 Schoolbook)
    # A2 is cols 3,4.
    # B23 is rows 3,4, cols 0,1,2.
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u = [0]*10
                v = [0]*25
                w = [0]*10
                # A2 local j -> global 3+j
                u[6 + i*2 + j] = 1
                # B23 local (j, k) -> global (3+j, k)
                v[(3+j)*5 + k] = 1
                # C local (k, i) -> global (k, i)
                w[k*2 + i] = 1
                add_term(u, v, w)

    # Convert to lists of lists as required by interface
    # The seed returns {'U': U, 'V': V, 'W': W} where U is list of lists.
    # My U is list of lists (each term is a list).
    # But the interface expects U to be shape (n*m, rank).
    # So U[i][r] is the value.
    # My current U is [term_0, term_1, ...] where term_i is a vector.
    # I need to transpose.
    
    rank = len(U)
    U_out = [[0]*rank for _ in range(n*m)]
    V_out = [[0]*rank for _ in range(m*p)]
    W_out = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            U_out[i][r] = U[r][i]
        for i in range(m*p):
            V_out[i][r] = V[r][i]
        for i in range(p*n):
            W_out[i][r] = W[r][i]
            
    return {"U": U_out, "V": V_out, "W": W_out}

