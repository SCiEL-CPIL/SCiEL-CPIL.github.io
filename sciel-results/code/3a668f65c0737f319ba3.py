def construct():
    import numpy as np
    
    # Find a Sidon set of size ~7 in range 50
    # Greedy approach for Sidon set
    N = 50
    S = []
    used_sums = set()
    
    # We want to maximize the size of the Sidon set
    # Start with 0
    S.append(0)
    used_sums.add(0) # 0+0
    
    for i in range(1, N):
        ok = True
        # Check sums with existing elements
        # i + i
        if i + i in used_sums:
            ok = False
        else:
            for j in S:
                if i + j in used_sums:
                    ok = False
                    break
        if ok:
            used_sums.add(i + i)
            for j in S:
                used_sums.add(i + j)
            S.append(i)
            
    # Create the sequence
    heights = np.zeros(N)
    for idx in S:
        heights[idx] = 1.0
        
    return heights
