    # Try 22+22+17 decomposition
    U22, V22, W22 = get_22_rank_block()
    U17, V17, W17 = get_17_rank_block()
    
    rank_total = 22 + 22 + 17
    U = np.zeros((9, rank_total))
    V = np.zeros((24, rank_total))
    W = np.zeros((24, rank_total))
    
    # Block 1: k in {0,1,2}
    U[:, 0:22] = U22
    # Map V22 (9x22) to V (24x22) for k in {0,1,2}
    for c in range(22):
        for j in range(3):
            for k in range(3):
                V[j*8 + k, c] = V22[j*3 + k, c]
    # Map W22 (9x22) to W (24x22) for k in {0,1,2}
    for c in range(22):
        for k in range(3):
            for i in range(3):
                W[k*3 + i, c] = W22[k*3 + i, c]
                
    # Block 2: k in {3,4,5}
    U[:, 22:44] = U22
    for c in range(22):
        for j in range(3):
            for k in range(3):
                V[j*8 + (3+k), 22+c] = V22[j*3 + k, c]
    for c in range(22):
        for k in range(3):
            for i in range(3):
                W[(3+k)*3 + i, 22+c] = W22[k*3 + i, c]
                
    # Block 3: k in {6,7}
    U[:, 44:61] = U17
    for c in range(17):
        for j in range(3):
            for k in range(2):
                V[j*8 + (6+k), 44+c] = V17[j*2 + k, c]
    for c in range(17):
        for k in range(2):
            for i in range(3):
                W[(6+k)*3 + i, 44+c] = W17[k*3 + i, c]
                
    if validate(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to 68
    # ... existing code ...
