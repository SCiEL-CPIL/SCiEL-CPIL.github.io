import time
import random

def construct():
    N = 256
    deadline = time.time() + 55  # 55s safety margin (90s limit)
    
    # Precompute weights
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
    
    # For each x and each a, precompute the partner that would form a 3-AP
    # If x is added, it conflicts with (a, 2x-a) or (a, 2a-x)
    # We'll use bitmasks for fast validity checks
    
    def is_valid_addition(mask, x):
        """Check if adding x to the set represented by mask keeps it 3-AP free."""
        # For each a in the set, check if 2x-a is also in the set (x is middle or endpoint)
        # OR if 2a-x is also in the set (a is middle or endpoint)
        m = mask
        while m:
            low = m & (-m)
            a = low.bit_length()  # a is the index
            # Check if 2x - a is in the set
            partner1 = 2 * x - a
            if 1 <= partner1 <= N and partner1 != a:
                if mask & (1 << (partner1 - 1)):
                    return False
            # Check if 2a - x is in the set
            partner2 = 2 * a - x
            if 1 <= partner2 <= N and partner2 != a:
                if mask & (1 << (partner2 - 1)):
                    return False
            m ^= low
        return True
    
    def score_from_mask(mask):
        s = 0.0
        m = mask
        while m:
            low = m & (-m)
            x = low.bit_length()
            s += weights[x]
            m ^= low
        return s
    
    # Greedy initial solution: add smallest valid number first
    mask = 0
    for i in range(1, N + 1):
        if is_valid_addition(mask, i):
            mask |= (1 << (i - 1))
    
    best_mask = mask
    best_score = score_from_mask(mask)
    
    # Local search
    while time.time() < deadline:
        improved = False
        
        # Try to add any available number, preferring small ones (high weight)
        candidates = [i for i in range(1, N + 1) if not (mask & (1 << (i - 1)))]
        candidates.sort(key=lambda x: -weights[x])
        
        added = False
        for x in candidates:
            if is_valid_addition(mask, x):
                mask |= (1 << (x - 1))
                added = True
                current_score = score_from_mask(mask)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
                break
        
        if not added:
            # No single addition possible. Try swaps.
            # Remove a low-value element y from S and add a high-value element x not in S.
            
            # Get elements in S sorted by weight ascending (largest numbers first)
            in_set = []
            m = mask
            while m:
                low = m & (-m)
                x = low.bit_length()
                in_set.append(x)
                m ^= low
            in_set.sort(key=lambda x: weights[x])  # lowest weight first
            
            # Get candidates sorted by weight descending (smallest numbers first)
            candidates.sort(key=lambda x: -weights[x])
            
            swapped = False
            for y in in_set:
                if swapped:
                    break
                for x in candidates:
                    if x == y:
                        continue
                    # Temporarily remove y
                    mask_without_y = mask & ~(1 << (y - 1))
                    if is_valid_addition(mask_without_y, x):
                        mask = mask_without_y | (1 << (x - 1))
                        swapped = True
                        current_score = score_from_mask(mask)
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = mask
                        break
            
            if not swapped:
                # Stuck. Perturb.
                # Remove 1 to 3 random elements, preferring to remove low-weight (large) elements
                num_remove = random.randint(1, 3)
                if len(in_set) > num_remove:
                    # Weighted random selection: prefer removing large numbers (low weight)
                    to_remove = []
                    available = list(in_set)
                    for _ in range(min(num_remove, len(available))):
                        # Pick randomly from the larger half of available (lower weight)
                        if len(available) > 2:
                            # Bias towards larger numbers
                            idx = random.randint(len(available) // 2, len(available) - 1)
                        else:
                            idx = random.randint(0, len(available) - 1)
                        to_remove.append(available[idx])
                        available.pop(idx)
                    
                    for r in to_remove:
                        mask &= ~(1 << (r - 1))
                    
                    # Try to refill greedily
                    for _ in range(10):
                        candidates = [i for i in range(1, N + 1) if not (mask & (1 << (i - 1)))]
                        candidates.sort(key=lambda x: -weights[x])
                        added = False
                        for x in candidates:
                            if is_valid_addition(mask, x):
                                mask |= (1 << (x - 1))
                                added = True
                                break
                        if not added:
                            break
                    
                    current_score = score_from_mask(mask)
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
                
                # Occasionally do a harder restart
                if random.random() < 0.05:
                    # Hard restart: greedy with random tie-breaking
                    mask = 0
                    for i in range(1, N + 1):
                        if random.random() < 0.5:
                            if is_valid_addition(mask, i):
                                mask |= (1 << (i - 1))
                    current_score = score_from_mask(mask)
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
    
    # Convert best mask to list
    result = []
    m = best_mask
    while m:
        low = m & (-m)
        x = low.bit_length()
        result.append(x)
        m ^= low
    
    return result
