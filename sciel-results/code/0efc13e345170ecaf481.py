import random
import time

def construct():
    # Precompute all 4-subsets as bitmasks
    masks = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                for l in range(k+1, 16):
                    masks.append((1 << i) | (1 << j) | (1 << k) | (1 << l))
    
    def check_valid(S, current_set):
        # Group current_set by intersection with S
        # Since S has 4 bits, there are at most 16 possible intersections
        # We can use a list of 16 lists, indexed by the submask value?
        # No, submask value is up to 2^16.
        # But we only care about K = A & S.
        # We can just use a dict or list of lists for the 16 submasks of S.
        
        # To speed up, we can precompute the list of submasks of S?
        # Or just iterate A and use A&S as key.
        
        groups = {}
        for A in current_set:
            K = A & S
            if K not in groups:
                groups[K] = []
            groups[K].append(A)
        
        for K, As in groups.items():
            if len(As) < 2:
                continue
            # Check pairs
            # Optimization: If we find one, return False
            for i in range(len(As)):
                Ai = As[i]
                for j in range(i+1, len(As)):
                    Aj = As[j]
                    if (Ai & Aj) == K:
                        return False
        return True

    best_set = []
    best_size = 0
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial greedy to get a baseline
    # Then randomized restarts
    
    n_masks = len(masks)
    
    while time.time() < deadline:
        # Random shuffle
        shuffled = masks[:]
        random.shuffle(shuffled)
        
        current_set = []
        for S in shuffled:
            if check_valid(S, current_set):
                current_set.append(S)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    # Convert back to list of lists
    result = []
    for m in best_set:
        elems = []
        for i in range(16):
            if m & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result
