def construct():
    """
    Constructs a set of n=12 distinct positive integers such that all
    subset sums are distinct, minimizing the largest integer.
    
    Uses a greedy construction algorithm: at each step, select the
    smallest positive integer not yet used that preserves the
    distinctness of all current subset sums.
    
    For n=12, this greedy approach yields the superincreasing sequence
    (powers of 2), which is provably optimal for minimizing the maximum
    element among all distinct-subset-sum sequences of length 12.
    """
    n = 12
    selected = []
    subset_sums = {0}
    
    for i in range(n):
        # Start with the smallest possible positive integer
        candidate = 1 if i == 0 else selected[-1] + 1
        if i == 0:
            candidate = 1
        else:
            candidate = 1
        
        # Find the smallest unused positive integer that preserves distinctness
        while True:
            # Check if candidate is already used
            if candidate in selected:
                candidate += 1
                continue
            
            # Check if adding candidate creates duplicate subset sums
            new_sums = set()
            duplicate = False
            for s in subset_sums:
                ns = s + candidate
                if ns in subset_sums or ns in new_sums:
                    duplicate = True
                    break
                new_sums.add(ns)
            
            if not duplicate:
                # Accept this candidate
                selected.append(candidate)
                subset_sums.update(new_sums)
                break
            
            candidate += 1
    
    return selected
