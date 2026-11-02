import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _strassen_2x2_terms():
    """
    Returns list of (u, v, w) for 2x2x2 Strassen decomposition.
    u is vector of length 4 (A elements), v is vector of length 4 (B elements), w is vector of length 4 (C elements).
    Indices: A: 00, 01, 10, 11 -> 0, 1, 2, 3
    B: 00, 01, 10, 11 -> 0, 1, 2, 3
    C: 00, 01, 10, 11 -> 0, 1, 2, 3
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C01 -= M3, C10 -= M4, C11 += M1
    # Actually, Strassen formulas:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M7
    
    # We need to add contributions to U, V, W.
    # Each term is a rank-1 tensor.
    # Term k: u_k, v_k, w_k.
    # C = sum u_k v_k w_k^T ? No.
    # C_{ij} = sum_k u_{ik} v_{jk} w_{kl}? No.
    # The tensor is T = sum u ⊗ v ⊗ w.
    # C_{ij} = sum_k u_{i, k} v_{j, k} w_{k, (i,j)}? 
    # No, the standard form is:
    # C_{ij} = sum_k U_{i, k} V_{j, k} W_{k, (i,j)}? 
    # Wait, the task defines W[k*n+i, r].
    # So W is indexed by (output index, term index).
    # Output index is (i, k) for C? No, C is n x p.
    # In 2x2 case, C is 2x2. Index (r, c).
    # W index is c*2 + r.
    
    # Let's define the terms explicitly.
    # M1 = (A00+A11)(B00+B11)
    # u: A00=1, A11=1. Others 0.
    # v: B00=1, B11=1. Others 0.
    # w: Contributes to C00 (+1), C11 (+1).
    # C00 index 0, C11 index 3.
    
    # M2 = (A10+A11)B00
    # u: A10=1, A11=1.
    # v: B00=1.
    # w: Contributes to C10 (+1). Index 2.
    
    # M3 = A00(B10-B11)
    # u: A00=1.
    # v: B10=1, B11=-1.
    # w: Contributes to C01 (+1), C11 (-1). Indices 1, 3.
    
    # M4 = A11(B01-B00)
    # u: A11=1.
    # v: B01=1, B00=-1.
    # w: Contributes to C00 (-1), C10 (+1). Indices 0, 2.
    
    # M5 = (A00+A01)B11
    # u: A00=1, A01=1.
    # v: B11=1.
    # w: Contributes to C01 (+1), C11 (+1). Indices 1, 3.
    
    # M6 = (A10-A00)(B00+B01)
    # u: A10=1, A00=-1.
    # v: B00=1, B01=1.
    # w: Contributes to C00 (+1). Index 0.
    
    # M7 = (A01-A11)(B10+B11)
    # u: A01=1, A11=-1.
    # v: B10=1, B11=1.
    # w: Contributes to C00 (+1), C11 (-1). Indices 0, 3.
    
    # We will store these as sparse lists of (index, value) for each vector.
    
    def make_vec(pairs):
        v = [0.0] * 4
        for idx, val in pairs:
            v[idx] = val
        return v

    terms.append({
        'u': make_vec([(0,1), (3,1)]),
        'v': make_vec([(0,1), (3,1)]),
        'w': make_vec([(0,1), (3,1)]) # C00, C11
    })
    terms.append({
        'u': make_vec([(2,1), (3,1)]),
        'v': make_vec([(0,1)]),
        'w': make_vec([(2,1)]) # C10
    })
    terms.append({
        'u': make_vec([(0,1)]),
        'v': make_vec([(2,1), (3,-1)]),
        'w': make_vec([(1,1), (3,-1)]) # C01, C11
    })
    terms.append({
        'u': make_vec([(3,1)]),
        'v': make_vec([(1,1), (0,-1)]),
        'w': make_vec([(0,-1), (2,1)]) # C00, C10
    })
    terms.append({
        'u': make_vec([(0,1), (1,1)]),
        'v': make_vec([(3,1)]),
        'w': make_vec([(1,1), (3,1)]) # C01, C11
    })
    terms.append({
        'u': make_vec([(2,1), (0,-1)]),
        'v': make_vec([(0,1), (1,1)]),
        'w': make_vec([(0,1)]) # C00
    })
    terms.append({
        'u': make_vec([(1,1), (3,-1)]),
        'v': make_vec([(2,1), (3,1)]),
        'w': make_vec([(0,1), (3,-1)]) # C00, C11
    })
    return terms

def construct():
    n, m, p = DIMENSIONS
    # We decompose into 3 parts: 4x4x4, 4x4x4, 4x4x1
    # Total rank = 49 + 49 + 16 = 114
    
    rank = 114
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block of terms
    def add_block(offset_u, offset_v, offset_w, terms, scale=1.0):
        nonlocal col
        for t in terms:
            u_vec = t['u']
            v_vec = t['v']
            w_vec = t['w']
            for i in range(4):
                if u_vec[i] != 0:
                    U[offset_u + i][col] = u_vec[i] * scale
            for j in range(4):
                if v_vec[j] != 0:
                    V[offset_v + j][col] = v_vec[j] * scale
            for k in range(4):
                if w_vec[k] != 0:
                    W[offset_w + k][col] = w_vec[k] * scale
            col += 1

    # Part 1: C1 = A B1 (cols 0-3 of B)
    # A is 4x4, B1 is 4x4, C1 is 4x4.
    # We use Strassen on 4x4.
    # Split A into 4 blocks of 2x2.
    # A00: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,4,5 in A (row-major 4x4)
    # A01: (0,2),(0,3),(1,2),(1,3) -> indices 2,3,6,7
    # A10: (2,0),(2,1),(3,0),(3,1) -> indices 8,9,12,13
    # A11: (2,2),(2,3),(3,2),(3,3) -> indices 10,11,14,15
    
    # B1 is cols 0-3 of B.
    # B1 is 4x4.
    # B100: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,4,5 in B (row-major 4x9, but we only use first 4 cols)
    # B101: (0,2),(0,3),(1,2),(1,3) -> indices 2,3,6,7
    # B110: (2,0),(2,1),(3,0),(3,1) -> indices 8,9,12,13
    # B111: (2,2),(2,3),(3,2),(3,3) -> indices 10,11,14,15
    
    # C1 is 4x4.
    # C100: indices 0,1,4,5 in W? No, W is p*n = 9*4 = 36.
    # W index is k*n + i.
    # C1 corresponds to k=0..3.
    # C100 (i=0,1; k=0,1) -> W indices 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
    # C101 (i=0,1; k=2,3) -> W indices 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13.
    # C110 (i=2,3; k=0,1) -> W indices 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7.
    # C111 (i=2,3; k=2,3) -> W indices 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15.
    
    # We need to map the 2x2 Strassen terms to the 4x4 blocks.
    # This is complex to hardcode.
    # Instead, we can generate the 4x4 decomposition by combining 2x2 decompositions.
    
    # Let's write a function to generate 4x4x4 terms from 2x2x2 terms.
    def generate_4x4x4_terms():
        # 7 products of 2x2 matrices.
        # Each product is decomposed into 7 rank-1 terms.
        # Total 49 terms.
        all_terms = []
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # ...
        
        # We need to define the linear combinations for A and B blocks.
        # A_blocks: A00, A01, A10, A11
        # B_blocks: B00, B01, B10, B11
        
        # For each P_k, we have a linear combination of A blocks and B blocks.
        # e.g. P1: A_comb = A00 + A11, B_comb = B00 + B11.
        # The rank-1 terms for P1 are the same as Strassen 2x2, but the u and v vectors are mapped to the 4x4 space.
        
        # Let's create a mapping for u and v.
        # u is length 4 (2x2). We need to expand to length 16 (4x4).
        # For A00, the elements are at indices 0,1,4,5.
        # For A01, 2,3,6,7.
        # For A10, 8,9,12,13.
        # For A11, 10,11,14,15.
        
        def expand_u(u_2x2, block_indices):
            # u_2x2 is length 4.
            # block_indices is a list of 4 indices in the 16-dim space corresponding to the 2x2 block.
            u_4x4 = [0.0] * 16
            for i in range(4):
                if u_2x2[i] != 0:
                    u_4x4[block_indices[i]] = u_2x2[i]
            return u_4x4
            
        def expand_v(v_2x2, block_indices):
            v_4x4 = [0.0] * 16
            for j in range(4):
                if v_2x2[j] != 0:
                    v_4x4[block_indices[j]] = v_2x2[j]
            return v_4x4

        def expand_w(w_2x2, block_indices):
            # w is for C. C is 4x4.
            # W index in 4x4 space is k*4 + i.
            # block_indices for C blocks.
            w_4x4 = [0.0] * 16
            for k in range(4):
                if w_2x2[k] != 0:
                    w_4x4[block_indices[k]] = w_2x2[k]
            return w_4x4

        # Block index mappings
        # A: row-major 4x4.
        # A00: 0,1,4,5
        # A01: 2,3,6,7
        # A10: 8,9,12,13
        # A11: 10,11,14,15
        A_blocks = {
            0: [0,1,4,5],
            1: [2,3,6,7],
            2: [8,9,12,13],
            3: [10,11,14,15]
        }
        
        # B: row-major 4x4 (part of 4x9).
        # B00: 0,1,4,5
        # B01: 2,3,6,7
        # B10: 8,9,12,13
        # B11: 10,11,14,15
        B_blocks = A_blocks # Same structure
        
        # C: W index k*n + i. n=4.
        # C00 (i=0,1; k=0,1): 0,1,4,5
        # C01 (i=0,1; k=2,3): 8,9,12,13
        # C10 (i=2,3; k=0,1): 2,3,6,7
        # C11 (i=2,3; k=2,3): 10,11,14,15
        C_blocks = {
            0: [0,1,4,5],
            1: [8,9,12,13],
            2: [2,3,6,7],
            3: [10,11,14,15]
        }
        
        base_terms = _strassen_2x2_terms()
        
        # Define the 7 products
        # P1: A00+A11, B00+B11
        # P2: A10+A11, B00
        # P3: A00, B10-B11
        # P4: A11, B01-B00
        # P5: A00+A01, B11
        # P6: A10-A00, B00+B01
        # P7: A01-A11, B10+B11
        
        # We need to combine the u and v vectors for each product.
        # For P1, u = u_A00 + u_A11.
        # Since the base terms are linear, we can just add the expanded vectors.
        
        products = [
            ([0, 3], [0, 3]), # P1
            ([2, 3], [0]),    # P2
            ([0], [2, 3]),    # P3 (B10-B11 -> [2, -3])
            ([3], [1, 0]),    # P4 (B01-B00 -> [1, -0])
            ([0, 1], [3]),    # P5
            ([2, 0], [0, 1]), # P6 (A10-A00 -> [2, -0])
            ([1, 3], [2, 3])  # P7 (A01-A11 -> [1, -3])
        ]
        
        # Note: The coefficients are implicit in the base terms?
        # No, the base terms are for standard multiplication.
        # Strassen formulas use specific combinations.
        # We must apply the coefficients from Strassen formulas.
        
        # Let's re-verify Strassen formulas.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 - M7
        
        # We will generate terms for each M_k, then combine them for C.
        # But we can just generate the rank-1 terms for each M_k and add them to the global sum with appropriate signs for W.
        
        # For each M_k, we have 7 rank-1 terms.
        # Total 49 terms.
        
        for m_idx in range(7):
            # Determine A and B block combinations and signs
            if m_idx == 0: # M1
                a_blocks = [(0, 1), (3, 1)]
                b_blocks = [(0, 1), (3, 1)]
            elif m_idx == 1: # M2
                a_blocks = [(2, 1), (3, 1)]
                b_blocks = [(0, 1)]
            elif m_idx == 2: # M3
                a_blocks = [(0, 1)]
                b_blocks = [(2, 1), (3, -1)]
            elif m_idx == 3: # M4
                a_blocks = [(3, 1)]
                b_blocks = [(1, 1), (0, -1)]
            elif m_idx == 4: # M5
                a_blocks = [(0, 1), (1, 1)]
                b_blocks = [(3, 1)]
            elif m_idx == 5: # M6
                a_blocks = [(2, 1), (0, -1)]
                b_blocks = [(0, 1), (1, 1)]
            elif m_idx == 6: # M7
                a_blocks = [(1, 1), (3, -1)]
                b_blocks = [(2, 1), (3, 1)]
            
            # For each of the 7 base terms of 2x2 multiplication
            for b_idx in range(7):
                base_u = base_terms[b_idx]['u']
                base_v = base_terms[b_idx]['v']
                base_w = base_terms[b_idx]['w']
                
                # Expand u to 4x4
                u_4x4 = [0.0] * 16
                for block_idx, sign in a_blocks:
                    block_indices = A_blocks[block_idx]
                    for i in range(4):
                        if base_u[i] != 0:
                            u_4x4[block_indices[i]] += sign * base_u[i]
                            
                # Expand v to 4x4
                v_4x4 = [0.0] * 16
                for block_idx, sign in b_blocks:
                    block_indices = B_blocks[block_idx]
                    for j in range(4):
                        if base_v[j] != 0:
                            v_4x4[block_indices[j]] += sign * base_v[j]
                            
                # Expand w to 4x4
                # We need to determine which C blocks this M_k contributes to and with what sign.
                # This depends on m_idx and b_idx (which determines which C element of the 2x2 result).
                # base_w tells us which elements of the 2x2 result (C00, C01, C10, C11) are affected.
                
                w_4x4 = [0.0] * 16
                for c_block_idx in range(4):
                    if base_w[c_block_idx] != 0:
                        # Determine sign from Strassen formulas
                        sign = 0
                        if m_idx == 0: # M1
                            if c_block_idx == 0: sign = 1   # C00
                            elif c_block_idx == 3: sign = 1 # C11
                        elif m_idx == 1: # M2
                            if c_block_idx == 2: sign = 1   # C10
                        elif m_idx == 2: # M3
                            if c_block_idx == 1: sign = 1   # C01
                            elif c_block_idx == 3: sign = -1# C11
                        elif m_idx == 3: # M4
                            if c_block_idx == 0: sign = -1  # C00
                            elif c_block_idx == 2: sign = 1 # C10
                        elif m_idx == 4: # M5
                            if c_block_idx == 1: sign = 1   # C01
                            elif c_block_idx == 3: sign = 1 # C11
                        elif m_idx == 5: # M6
                            if c_block_idx == 0: sign = 1   # C00
                        elif m_idx == 6: # M7
                            if c_block_idx == 0: sign = 1   # C00
                            elif c_block_idx == 3: sign = -1# C11
                            
                        if sign != 0:
                            block_indices = C_blocks[c_block_idx]
                            for k in range(4):
                                if base_w[c_block_idx] != 0:
                                    w_4x4[block_indices[k]] += sign * base_w[c_block_idx]
                
                all_terms.append({
                    'u': u_4x4,
                    'v': v_4x4,
                    'w': w_4x4
                })
                
        return all_terms

    terms_4x4x4 = generate_4x4x4_terms()
    
    # Add Part 1: C1 (cols 0-3)
    # U offset 0, V offset 0, W offset 0
    add_block(0, 0, 0, terms_4x4x4)
    
    # Add Part 2: C2 (cols 4-7)
    # U offset 0, V offset 16 (4 cols * 4 rows), W offset 16 (4 cols * 4 rows)
    add_block(0, 16, 16, terms_4x4x4)
    
    # Part 3: C3 (col 8)
    # 4x4x1 multiplication.
    # C_{i, 0} = sum_j A_{i,j} B_{j, 0}
    # 16 terms.
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1.0
            V[j * 9 + 8][col] = 1.0 # B index j*9 + 8
            W[8 * 4 + i][col] = 1.0 # W index 8*4 + i
            col += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
