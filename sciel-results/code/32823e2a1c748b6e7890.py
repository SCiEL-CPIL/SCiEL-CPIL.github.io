def greedy_search(seed):
    np.random.seed(seed)
    covered = np.zeros(TOTAL, dtype=np.int8)
    code = []
    remaining = TOTAL
    
    while remaining > 0:
        uncovered = 1 - covered
        gains = uncovered[NEIGHBORS].sum(axis=1)
        max_gain = np.max(gains)
        candidates = np.where(gains == max_gain)[0]
        c = np.random.choice(candidates)
        code.append(c)
        
        # Update remaining
        # Count how many of c's neighbors were 0
        new_covered = np.sum(covered[NEIGHBORS[c]] == 0)
        remaining -= new_covered
        
        # Update covered
        covered[NEIGHBORS[c]] = 1
        
    # Delete phase
    for i in range(len(code) - 1, -1, -1):
        c = code[i]
        if np.min(covered[NEIGHBORS[c]]) > 1:
            covered[NEIGHBORS[c]] -= 1
            code.pop(i)
            
    return code
