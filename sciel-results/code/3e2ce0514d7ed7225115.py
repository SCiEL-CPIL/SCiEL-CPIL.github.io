import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    u_forms = [
        [1, 0, 0, 1],  # P1: a+d
        [0, 0, 1, 1],  # P2: c+d
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: a+b
        [-1, 0, 1, 0], # P6: c-a
        [0, 1, 0, -1]  # P7: b-d
    ]
    
    # Strassen linear forms for B (2x2) local
    # B indices: e, f, g, h -> 0, 1, 2, 3
    v_forms_local = [
        [1, 0, 0, 1],  # P1: e+h
        [1, 0, 0, 0],  # P2: e
        [0, 1, 0, -1], # P3: f-h
        [0, 0, 1, -1], # P4: g-e
        [0, 0, 0, 1],  # P5: h
        [1, 1, 0, 0],  # P6: e+f
        [0, 0, 1, 1]   # P7: g+h
    ]
    
    # W contributions for each Pk to C indices (00, 01, 10, 11)
    # C indices local: 0, 1, 2, 3
    w_contribs = [
        [1, 0, 0, 1],  # P1: C00+1, C11+1
        [0, 0, 1, -1], # P2: C10+1, C11-1
        [0, 1, 0, 1],  # P3: C01+1, C11+1
        [1, 0, 1, 0],  # P4: C00+1, C10+1
        [-1, 1, 0, 0], # P5: C00-1, C01+1
        [0, 0, 0, 1],  # P6: C11+1
        [1, 0, 0, 0]   # P7: C00+1
    ]
    
    for i in range(4):
        for k in range(7):
            r = i * 7 + k
            
            # Fill U
            for idx, val in enumerate(u_forms[k]):
                if val != 0:
                    U[idx][r] = val
                    
            # Fill V
            # B global indices for block i
            # e: 2i, f: 2i+1, g: 8+2i, h: 8+2i+1
            v_indices = [2*i, 2*i+1, 8+2*i, 8+2*i+1]
            for idx, val in enumerate(v_forms_local[k]):
                if val != 0:
                    V[v_indices[idx]][r] = val
                    
            # Fill W
            # C global indices for block i
            # C00: 4i, C01: 4i+2, C10: 4i+1, C11: 4i+3
            w_indices = [4*i, 4*i+2, 4*i+1, 4*i+3]
            for idx, val in enumerate(w_contribs[k]):
                if val != 0:
                    W[w_indices[idx]][r] = val

    return {"U": U, "V": V, "W": W}
