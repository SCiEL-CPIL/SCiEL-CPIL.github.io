import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Base terms for M_2,2,2 (Strassen)
    # u, v, w are dicts {local_index: value}
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),      # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),      # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),      # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),      # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),      # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})       # M7
    ]
    
    # Base terms for M_2,2,1 (Schoolbook)
    schoolbook_221_terms = [
        ({0: 1}, {0: 1}, {0: 1}), # A00 B00 -> C00
        ({1: 1}, {1: 1}, {0: 1}), # A01 B10 -> C00
        ({2: 1}, {0: 1}, {1: 1}), # A10 B00 -> C10
        ({3: 1}, {1: 1}, {1: 1})  # A11 B10 -> C10
    ]
    
    # Helper to map local terms to global terms
    def map_terms(local_terms, u_map, v_map, w_map):
        mapped = []
        for u, v, w in local_terms:
            new_u = {u_map[i]: val for i, val in u.items()}
            new_v = {v_map[i]: val for i, val in v.items()}
            new_w = {w_map[i]: val for i, val in w.items()}
            mapped.append((new_u, new_v, new_w))
        return mapped

    # Generate M_2,2,4 terms
    # B is 2x4. Split into B1 (cols 0,1) and B2 (cols 2,3).
    # C is 2x4. C1 (cols 0,1), C2 (cols 2,3).
    # A is 2x2.
    
    # For B1 (M_2,2,2 instance 1):
    # Local B idx 0,1,2,3 -> Global B idx 0,1,2,3?
    # Local B is 2x2. Indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # Global B for B1 is 2x2 at cols 0,1.
    # Global B index: r*4 + c? No, in M_2,2,4, B is 2x4.
    # Let's define local indices for M_2,2,4.
    # A: 0..3.
    # B: 0..7. (r*4+c)
    # C: 0..7. (r*4+c)
    
    # Mapping for B1 (cols 0,1 of M_2,2,4):
    # Local M_2,2,2 B idx 0 (0,0) -> M_2,2,4 B idx 0 (0,0)
    # Local M_2,2,2 B idx 1 (0,1) -> M_2,2,4 B idx 1 (0,1)
    # Local M_2,2,2 B idx 2 (1,0) -> M_2,2,4 B idx 4 (1,0)
    # Local M_2,2,2 B idx 3 (1,1) -> M_2,2,4 B idx 5 (1,1)
    
    # Mapping for C1 (cols 0,1 of M_2,2,4):
    # Local M_2,2,2 C idx 0 (0,0) -> M_2,2,4 C idx 0
    # Local M_2,2,2 C idx 1 (0,1) -> M_2,2,4 C idx 1
    # Local M_2,2,2 C idx 2 (1,0) -> M_2,2,4 C idx 4
    # Local M_2,2,2 C idx 3 (1,1) -> M_2,2,4 C idx 5
    
    u_map_222 = lambda i: i
    v_map_B1 = {0:0, 1:1, 2:4, 3:5}
    w_map_C1 = {0:0, 1:1, 2:4, 3:5}
    
    terms_B1 = map_terms(strassen_terms, u_map_222, v_map_B1, w_map_C1)
    
    # Mapping for B2 (cols 2,3 of M_2,2,4):
    # Local M_2,2,2 B idx 0 (0,0) -> M_2,2,4 B idx 2 (0,2)
    # Local M_2,2,2 B idx 1 (0,1) -> M_2,2,4 B idx 3 (0,3)
    # Local M_2,2,2 B idx 2 (1,0) -> M_2,2,4 B idx 6 (1,2)
    # Local M_2,2,2 B idx 3 (1,1) -> M_2,2,4 B idx 7 (1,3)
    
    # Mapping for C2 (cols 2,3 of M_2,2,4):
    # Local M_2,2,2 C idx 0 (0,0) -> M_2,2,4 C idx 2
    # Local M_2,2,2 C idx 1 (0,1) -> M_2,2,4 C idx 3
    # Local M_2,2,2 C idx 2 (1,0) -> M_2,2,4 C idx 6
    # Local M_2,2,2 C idx 3 (1,1) -> M_2,2,4 C idx 7
    
    v_map_B2 = {0:2, 1:3, 2:6, 3:7}
    w_map_C2 = {0:2, 1:3, 2:6, 3:7}
    
    terms_B2 = map_terms(strassen_terms, u_map_222, v_map_B2, w_map_C2)
    
    terms_224 = terms_B1 + terms_B2
    
    # Generate M_2,2,3 terms
    # B is 2x3. Split into B1 (cols 0,1) and B2 (col 2).
    # C is 2x3. C1 (cols 0,1), C2 (col 2).
    # A is 2x2.
    
    # Local M_2,2,3 indices:
    # A: 0..3
    # B: 0..5 (r*3+c)
    # C: 0..5 (r*3+c)
    
    # Mapping for B1 (M_2,2,2) to M_2,2,3:
    # Local M_2,2,2 B idx 0 (0,0) -> M_2,2,3 B idx 0 (0,0)
    # Local M_2,2,2 B idx 1 (0,1) -> M_2,2,3 B idx 1 (0,1)
    # Local M_2,2,2 B idx 2 (1,0) -> M_2,2,3 B idx 3 (1,0)
    # Local M_2,2,2 B idx 3 (1,1) -> M_2,2,3 B idx 4 (1,1)
    
    # Mapping for C1 (M_2,2,2) to M_2,2,3:
    # Local M_2,2,2 C idx 0 (0,0) -> M_2,2,3 C idx 0
    # Local M_2,2,2 C idx 1 (0,1) -> M_2,2,3 C idx 1
    # Local M_2,2,2 C idx 2 (1,0) -> M_2,2,3 C idx 3
    # Local M_2,2,2 C idx 3 (1,1) -> M_2,2,3 C idx 4
    
    v_map_B1_223 = {0:0, 1:1, 2:3, 3:4}
    w_map_C1_223 = {0:0, 1:1, 2:3, 3:4}
    
    terms_B1_223 = map_terms(strassen_terms, u_map_222, v_map_B1_223, w_map_C1_223)
    
    # Mapping for B2 (M_2,2,1) to M_2,2,3:
    # Local M_2,2,1 B idx 0 (0,0) -> M_2,2,3 B idx 2 (0,2)
    # Local M_2,2,1 B idx 1 (1,0) -> M_2,2,3 B idx 5 (1,2)
    
    # Mapping for C2 (M_2,2,1) to M_2,2,3:
    # Local M_2,2,1 C idx 0 (0,0) -> M_2,2,3 C idx 2
    # Local M_2,2,1 C idx 1 (1,0) -> M_2,2,3 C idx 5
    
    v_map_B2_223 = {0:2, 1:5}
    w_map_C2_223 = {0:2, 1:5}
    
    terms_B2_223 = map_terms(schoolbook_221_terms, u_map_222, v_map_B2_223, w_map_C2_223)
    
    terms_223 = terms_B1_223 + terms_B2_223
    
    # Generate M_2,2,7 terms
    # B is 2x7. Split into B1 (cols 0..3) and B2 (cols 4..6).
    # C is 2x7. C1 (cols 0..3), C2 (cols 4..6).
    # A is 2x2.
    
    # Local M_2,2,7 indices:
    # A: 0..3
    # B: 0..13 (r*7+c)
    # C: 0..13 (r*7+c)
    
    # Mapping for B1 (M_2,2,4) to M_2,2,7:
    # Local M_2,2,4 B idx (r*4+c) -> M_2,2,7 B idx (r*7+c)
    # c in 0..3.
    v_map_B1_227 = {i: (i // 4) * 7 + (i % 4) for i in range(8)}
    w_map_C1_227 = {i: (i // 4) * 7 + (i % 4) for i in range(8)}
    
    terms_B1_227 = map_terms(terms_224, u_map_222, v_map_B1_227, w_map_C1_227)
    
    # Mapping for B2 (M_2,2,3) to M_2,2,7:
    # Local M_2,2,3 B idx (r*3+c) -> M_2,2,7 B idx (r*7 + (4+c))
    # c in 0..2.
    v_map_B2_227 = {i: (i // 3) * 7 + (4 + (i % 3)) for i in range(6)}
    w_map_C2_227 = {i: (i // 3) * 7 + (4 + (i % 3)) for i in range(6)}
    
    terms_B2_227 = map_terms(terms_223, u_map_222, v_map_B2_227, w_map_C2_227)
    
    terms_227 = terms_B1_227 + terms_B2_227
    
    # Generate M_2,4,7 terms
    # A is 2x4. Split into A1 (cols 0,1) and A2 (cols 2,3).
    # B is 4x7. Split into B1 (rows 0,1) and B2 (rows 2,3).
    # C is 2x7. C1 from A1B1, C2 from A2B2.
    
    # Global U size 8. Global V size 28. Global W size 14.
    
    # Mapping for A1 (M_2,2,7) to Global:
    # Local M_2,2,7 A idx (r*2+c) -> Global U idx (r*4+c)
    u_map_A1 = {i: (i // 2) * 4 + (i % 2) for i in range(4)}
    
    # Mapping for B1 (M_2,2,7) to Global V:
    # Local M_2,2,7 B idx (r*7+c) -> Global V idx (r*7+c)
    # Since B1 is rows 0,1 of Global B (4x7).
    v_map_B1_247 = {i: i for i in range(14)}
    
    # Mapping for C1 (M_2,2,7) to Global W:
    # Local M_2,2,7 C idx (r*7+c) -> Global W idx (c*2+r)
    w_map_C1_247 = {i: (i % 7) * 2 + (i // 7) for i in range(14)}
    
    terms_A1B1 = map_terms(terms_227, u_map_A1, v_map_B1_247, w_map_C1_247)
    
    # Mapping for A2 (M_2,2,7) to Global:
    # Local M_2,2,7 A idx (r*2+c) -> Global U idx (r*4 + (2+c))
    u_map_A2 = {i: (i // 2) * 4 + (2 + (i % 2)) for i in range(4)}
    
    # Mapping for B2 (M_2,2,7) to Global V:
    # Local M_2,2,7 B idx (r*7+c) -> Global V idx ((2+r)*7+c)
    v_map_B2_247 = {i: ((i // 7) + 2) * 7 + (i % 7) for i in range(14)}
    
    # Mapping for C2 (M_2,2,7) to Global W:
    # Local M_2,2,7 C idx (r*7+c) -> Global W idx (c*2+r)
    w_map_C2_247 = {i: (i % 7) * 2 + (i // 7) for i in range(14)}
    
    terms_A2B2 = map_terms(terms_227, u_map_A2, v_map_B2_247, w_map_C2_247)
    
    all_terms = terms_A1B1 + terms_A2B2
    
    rank = len(all_terms)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for col, (u, v, w) in enumerate(all_terms):
        for idx, val in u.items():
            U[idx, col] = val
        for idx, val in v.items():
            V[idx, col] = val
        for idx, val in w.items():
            W[idx, col] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
