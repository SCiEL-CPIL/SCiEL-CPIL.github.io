import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    N = 128
    
    def can_add(v, mask):
        """
        Check if v can be added to the set represented by mask.
        v is 1-indexed.
        mask is the current set (without v).
        Returns True if v can be added without creating a forbidden triple.
        """
        # Find all u in mask such that u+v is also in mask.
        # u is 1-indexed.
        # u in mask => bit (u-1) is set.
        # u+v in mask => bit (u+v-1) is set.
        
        # We need to check if there exist distinct u, w in mask 
        # such that:
        # 1. u+v in mask
        # 2. w+v in mask
        # 3. u+w in mask OR u+w == v
        
        # Collect candidates u
        candidates = []
        # u must be at least 1 and u+v <= 128
        max_u = 128 - v
        for u in range(1, max_u + 1):
            # Check if u is in mask
            if (mask >> (u - 1)) & 1:
                # Check if u+v is in mask
                if (mask >> (u + v - 1)) & 1:
                    candidates.append(u)
                    
        # Check pairs of candidates
        L = len(candidates)
        for i in range(L):
            u = candidates[i]
            for j in range(i + 1, L):
                w = candidates[j]
                s = u + w
                # Check if s is in mask or s == v
                if s == v:
                    return False
                if 1 <= s <= 128:
                    if (mask >> (s - 1)) & 1:
                        return False
        return True

    def greedy_random():
        mask = 0
        # Random permutation of 1..128
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        
        for v in candidates:
            if can_add(v, mask):
                mask |= (1 << (v - 1))
        return mask

    best_mask = 0
    best_count = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        mask = greedy_random()
        count = bin(mask).count('1')
        if count > best_count:
            best_count = count
            best_mask = mask
            
    # Convert best_mask to list of ints
    result = []
    for i in range(128):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
