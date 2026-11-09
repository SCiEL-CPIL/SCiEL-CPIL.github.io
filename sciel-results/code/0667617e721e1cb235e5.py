import numpy as np

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    # Build successors list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            
    # Compute Bottom Level (BL)
    # BL[i] = 1 + max(BL[j] for j in successors[i]), or 1 if no successors
    # We can do this with a topological sort or simply DFS with memoization.
    # Since n=30, even O(n^2) is fine.
    
    bl = [0] * n
    def compute_bl(i):
        if bl[i] != 0:
            return bl[i]
        max_succ_bl = 0
        for s in successors[i]:
            s_bl = compute_bl(s)
            if s_bl > max_succ_bl:
                max_succ_bl = s_bl
        bl[i] = 1 + max_succ_bl
        return bl[i]
        
    for i in range(n):
        compute_bl(i)
        
    bl_array = np.array(bl, dtype=float)
    
    # Combine BL and Time
    # Weight BL heavily to ensure structural order is respected
    # If BLs are close, Time breaks the tie.
    # Let's try Priority = BL * 100 + Time
    priority_scores = bl_array * 100 + times
    
    return priority_scores
