import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank calculated as 39
    rank = 39
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1

    # Block 1: M_{2,1,7}
    # j is fixed to 2
    # i in 0..1, k in 0..6
    for i in range(n):
        for k in range(p):
            a = i * m + 2
            b = 2 * p + k
            c = k * n + i
            add_term(a, 1, b, 1, c, 1)
            
    # Block 2: M_{2,2,4}
    # j in 0..1, k in 0..3
    # Split into two M_{2,2,2}
    # Sub-block 2a: k in 0..1
    # Sub-block 2b: k in 2..3
    
    # Strassen coefficients for M_{2,2,2}
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen terms (u, v, w) vectors in local 4D space
    # M1 = A00 * (B01 - B11) -> C01
    # M2 = (A00 + A01) * B11 -> C00, C11
    # M3 = (A10 + A11) * B00 -> C10, C11
    # M4 = A11 * (B10 - B00) -> C00, C11
    # M5 = (A00 + A11) * (B00 + B11) -> C00, C01, C10, C11
    # M6 = (A01 - A11) * (B10 + B11) -> C00, C11
    # M7 = (A00 - A10) * (B01 + B10) -> C10, C11
    
    # We define the 7 rank-1 tensors by their non-zero components
    # Format: (list of (idx, val) for U, list of (idx, val) for V, list of (idx, val) for W)
    
    strassen_terms = [
        # M1
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
        # M2
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (3, -1)]), # C00=1, C11=-1? 
        # Let's re-verify Strassen output formulas
        # C00 = M5 + M4 - M2 + M6
        # C01 = M1 + M5
        # C10 = M3 + M7
        # C11 = M5 - M1 + M3 - M7
        
        # So for M2, it contributes +1 to C00 and -1 to C11?
        # C00 has -M2. C11 has no M2?
        # Wait, C11 = M5 - M1 + M3 - M7. No M2.
        # So M2 contributes -1 to C00.
        # My previous line: [(0, 1), (3, -1)] means C00=1, C11=-1. This is wrong.
        # It should be C00=-1.
        
        # Let's list contributions to C indices (0,1,2,3)
        # M1: C01 += 1
        # M2: C00 -= 1
        # M3: C10 += 1
        # M4: C00 += 1
        # M5: C00 += 1, C01 += 1, C10 += 1, C11 += 1
        # M6: C00 += 1
        # M7: C10 += 1, C11 -= 1
        
        # Check C00: -M2 + M4 + M5 + M6.
        # Strassen: C00 = M5 + M4 - M2 + M6. Correct.
        # Check C01: M1 + M5. Correct.
        # Check C10: M3 + M7. Correct.
        # Check C11: M5 - M1 + M3 - M7. Correct.
    ]
    
    # Corrected Strassen Terms (U, V, W)
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    S = [
        # M1: A00 * (B01 - B11) -> C01
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
        # M2: (A00 + A01) * B11 -> -C00
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1)]),
        # M3: (A10 + A11) * B00 -> C10
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # M4: A11 * (B10 - B00) -> C00
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1)]),
        # M5: (A00 + A11) * (B00 + B11) -> C00, C01, C10, C11
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (1, 1), (2, 1), (3, 1)]),
        # M6: (A01 - A11) * (B10 + B11) -> C00
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        # M7: (A00 - A10) * (B01 + B10) -> C10, -C11
        ([(0, 1), (2, -1)], [(1, 1), (2, 1)], [(2, 1), (3, -1)])
    ]
    
    def add_strassen_block(k_offset):
        nonlocal col
        for term in S:
            u_terms, v_terms, w_terms = term
            # U mapping: local a (0..3) to global a
            # local a = i*2 + j. global a = i*3 + j.
            # 0->0, 1->1, 2->3, 3->4
            u_map = {0: 0, 1: 1, 2: 3, 3: 4}
            
            # V mapping: local b (0..3) to global b
            # local b = j*2 + k_loc. global b = j*7 + (k_offset + k_loc).
            # 0 (0,0) -> 0*7 + k_off
            # 1 (0,1) -> 0*7 + k_off + 1
            # 2 (1,0) -> 1*7 + k_off
            # 3 (1,1) -> 1*7 + k_off + 1
            v_map = {0: k_offset, 1: k_offset + 1, 2: 7 + k_offset, 3: 8 + k_offset}
            
            # W mapping: local c (0..3) to global c
            # local c = k_loc*2 + i. global c = (k_offset + k_loc)*2 + i.
            # 0 (0,0) -> k_off*2
            # 1 (0,1) -> k_off*2 + 1
            # 2 (1,0) -> (k_off+1)*2
            # 3 (1,1) -> (k_off+1)*2 + 1
            w_map = {0: k_offset*2, 1: k_offset*2 + 1, 2: (k_offset+1)*2, 3: (k_offset+1)*2 + 1}
            
            # We need to sum the contributions to U, V, W for this column
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            
            for idx, val in u_terms:
                u_vec[u_map[idx]] += val
            for idx, val in v_terms:
                v_vec[v_map[idx]] += val
            for idx, val in w_terms:
                w_vec[w_map[idx]] += val
                
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1

    # Add Block 2a: k in 0..1 (offset 0)
    add_strassen_block(0)
    
    # Add Block 2b: k in 2..3 (offset 2)
    add_strassen_block(2)
    
    # Block 3: M_{2,2,3}
    # j in 0..1, k in 4..6
    # Split into M_{2,2,2} (k in 4..5) and M_{2,2,1} (k in 6)
    
    # Sub-block 3a: M_{2,2,2} for k in 4..5 (offset 4)
    add_strassen_block(4)
    
    # Sub-block 3b: M_{2,2,1} for k in 6
    # Schoolbook for 2x2x1
    for i in range(n):
        for j in range(2): # j in 0,1
            a = i * m + j
            b = j * p + 6
            c = 6 * n + i
            add_term(a, 1, b, 1, c, 1)
            
    # Convert to lists
    U_list = [list(map(int, row)) for row in U]
    V_list = [list(map(int, row)) for row in V]
    W_list = [list(map(int, row)) for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
