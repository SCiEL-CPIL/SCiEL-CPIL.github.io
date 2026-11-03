import numpy as np

def construct():
    n = 50
    
    def calc_score(h):
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        c = np.convolve(h, h, mode='full')
        c_max = np.max(c)
        if c_max == 0:
            return 0.0
        return (l2_sq / (l1 * c_max))

    best_h = np.zeros(n)
    best_score = 0.0

    # Try to find a large Sidon set
    # A Sidon set has all pairwise sums (including doubles) distinct.
    
    max_k = 0
    best_set = []
    
    for trial in range(5000):
        # Randomized greedy
        indices = list(range(n))
        np.random.shuffle(indices)
        
        current_set = []
        used_sums = set()
        
        for i in indices:
            # Check if i can be added
            valid = True
            # Check sums with existing elements
            for j in current_set:
                s = i + j
                if s in used_sums:
                    valid = False
                    break
            if valid:
                s_ii = i + i
                if s_ii in used_sums:
                    valid = False
            
            if valid:
                current_set.append(i)
                # Update used_sums
                for j in current_set[:-1]: # j < i in the set? No, just all pairs
                    used_sums.add(i + j)
                used_sums.add(i + i)
                
        if len(current_set) > max_k:
            max_k = len(current_set)
            best_set = current_set.copy()
            
    # If we found a set, use it
    if max_k > 0:
        k = max_k
        h = np.zeros(n)
        for idx in best_set:
            h[idx] = 1.0 / k
        score = calc_score(h)
        if score > best_score:
            best_score = score
            best_h = h
    else:
        # Fallback
        best_h[0] = 1.0
        best_score = 1.0

    return list(best_h)
