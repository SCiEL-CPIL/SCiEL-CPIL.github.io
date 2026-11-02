import time

def construct():
    n, m, p = 2, 2, 14
    num_blocks = p // 2  # 7 blocks of 2 columns each
    rank = 7 * num_blocks  # 7 Strassen terms per block = 49
    
    deadline = time.time() + 110  # safety margin before 120s limit

    # U: shape (n*m, rank) = (4, rank)
    # V: shape (m*p, rank) = (28, rank)  
    # W: shape (p*n, rank) = (28, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2:
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V local indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W local indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    terms = [
        # (U_dict, V_dict, W_dict)
        # M1 = (A00+A11)*(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)*B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00*(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11*(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)*B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)*(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)*(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    col = 0
    for b in range(num_blocks):
        # V global indices for block b:
        # B00 -> row 0, col 2b -> index 0*14 + 2b = 2b
        # B01 -> row 0, col 2b+1 -> index 0*14 + 2b+1 = 2b+1
        # B10 -> row 1, col 2b -> index 1*14 + 2b = 14+2b
        # B11 -> row 1, col 2b+1 -> index 1*14 + 2b+1 = 14+2b+1
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        
        # W global indices for block b (trace convention W[k*n+i]):
        # C00 -> k=2b, i=0 -> index (2b)*2 + 0 = 4b
        # C01 -> k=2b+1, i=0 -> index (2b+1)*2 + 0 = 4b+2
        # C10 -> k=2b, i=1 -> index (2b)*2 + 1 = 4b+1
        # C11 -> k=2b+1, i=1 -> index (2b+1)*2 + 1 = 4b+3
        w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for u_dict, v_dict, w_dict in terms:
            for idx, val in u_dict.items():
                U[idx][col] += val
            for l_idx, val in v_dict.items():
                V[v_offsets[l_idx]][col] += val
            for l_idx, val in w_dict.items():
                W[w_offsets[l_idx]][col] += val
            col += 1
    
    # Verify correctness (optional, quick check)
    # The decomposition is exact by construction using Strassen's algorithm
    # for each independent 2x2 block. Rank = 49 = 7 * 7.
    
    if time.time() > deadline:
        # This should never happen for this fixed construction, but guard anyway
        pass
    
    return {"U": U, "V": V, "W": W}
