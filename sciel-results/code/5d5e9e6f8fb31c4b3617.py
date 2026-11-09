import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen terms for 2x2 matrix multiplication
    # Local indices: 0,1 for each dimension
    # For a 2x2 A * 2x2 B = 2x2 C:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = (a00-a11)(b10+b11)
    # M4 = a00(b01-b11)
    # M5 = a11(b10-b00)
    # M6 = (a00+a01)b11
    # M7 = (a10-a01)(b00+b01)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M6 + M7
    # c10 = M2 - M4
    # c11 = M1 - M2 + M3 + M5
    
    # Each term: list of (local_index, coefficient) for U, V, W
    # local index for U: i*2+j (i=row of A, j=col of A)
    # local index for V: j*2+k (j=row of B, k=col of B)
    # local index for W: k*2+i (k=col of C, i=row of C) -- trace convention
    
    strassen_terms = [
        # M1 = (a00+a11)(b00+b11)
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2 = (a10+a11)b00
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        # M3 = (a00-a11)(b10+b11)
        {'U': [(0, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(3, 1)]},
        # M4 = a00(b01-b11)
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        # M5 = a11(b10-b00)
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, -1), (1, 1)]},
        # M6 = (a00+a01)b11
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        # M7 = (a10-a01)(b00+b01)
        {'U': [(2, 1), (1, -1)], 'V': [(0, 1), (1, 1)], 'W': [(0, 1)]},
    ]
    
    # Four 2x2 blocks: (j_offset, k_offset)
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 2},
    ]
    
    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            # Add to U: local index -> global row
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            
            # Add to V: local index -> global row
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            
            # Add to W: local index -> global row (trace convention: k*n+i)
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
