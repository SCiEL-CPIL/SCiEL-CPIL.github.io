<<<<<<< SEARCH
    covered = np.zeros(N, dtype=bool)
    cover_count = np.zeros(N, dtype=int)
    C = []
    
    # 1. Greedy Phase
    while covered.sum() < N:
        gains = (~covered[neighbors]).sum(axis=1)
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
            
        C.append(best_idx)
        cover_count[neighbors[best_idx]] += 1
        covered[neighbors[best_idx]] = True
=======
    covered = np.zeros(N, dtype=bool)
    cover_count = np.zeros(N, dtype=int)
    C = []
    
    # 1. Partition-Based Initialization
    # Decompose space into 243 planes (fix first 5 coords, vary last 2)
    # For each plane, select 3 points forming a perfect 1-cover of the plane
    base_step = 3 ** 5 # 243
    step_5 = 3 ** 5
    step_6 = 3 ** 6
    
    # Pattern for 3 points in a 3x3 plane (local coords (u,v) where u is coord 5, v is coord 6)
    # We use pattern: (0,0), (1,2), (2,1)
    # Offsets:
    # (0,0) -> 0
    # (1,2) -> 1*3^5 + 2*3^6
    # (2,1) -> 2*3^5 + 1*3^6
    offset_1 = 0
    offset_2 = 1 * step_5 + 2 * step_6
    offset_3 = 2 * step_5 + 1 * step_6
    
    for base in range(0, N, base_step):
        # The 9 points in this plane are base + k for k in 0..8?
        # No, base has last 2 digits 0.
        # Points are base + u*3^5 + v*3^6
        c1 = base + offset_1
        c2 = base + offset_2
        c3 = base + offset_3
        
        C.append(c1)
        C.append(c2)
        C.append(c3)
        
        # Update coverage
        for c in [c1, c2, c3]:
            nb = neighbors[c]
            cover_count[nb] += 1
            covered[nb] = True
>>>>>>> REPLACE
