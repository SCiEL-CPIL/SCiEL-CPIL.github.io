import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 512
    
    # Precompute all 4-APs
    # For each x, store a list of masks representing the other 3 elements of APs containing x
    # Mask is an integer with bits set at positions of the other elements
    conflict_triples = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs (a, a+d, a+2d, a+3d)
    for a in range(1, n + 1):
        # Max d such that a+3d <= n
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            term4 = a + 3 * d
            ap = (a, b, c, term4)
            
            # For each term in the AP, the other 3 form a conflict mask
            # For term a, others are b, c, term4
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[a].append(mask_a)
            
            # For term b, others are a, c, term4
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[b].append(mask_b)
            
            # For term c, others are a, b, term4
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (term4 - 1))
            conflict_triples[c].append(mask_c)
            
            # For term4, others are a, b, c
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflict_triples[term4].append(mask_d)
            
    def can_add(x, S_mask):
        """Check if x can be added to S (represented as bitmask) without forming a 4-AP."""
        for mask in conflict_triples[x]:
            # Check if all bits in mask are set in S_mask
            if (mask & S_mask) == mask:
                return False
        return True

    best_mask = 0
    best_size = 0
    
    # Helper to get size of mask
    def popcount(x):
        return bin(x).count("1")
        
    # Multi-start local search
    while time.time() < deadline:
        # 1. Greedy Construction
        S_mask = 0
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S_mask):
                S_mask |= (1 << (x - 1))
                
        current_size = popcount(S_mask)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
            
        # 2. Local Search (Swaps)
        # We try to improve by swapping elements or adding new ones.
        # To be efficient, we limit the number of iterations per restart.
        # Since can_add is fast, we can do many.
        
        # First, try to add any element not in S
        # We can just iterate through all not-in-S elements once or twice
        # Then do random swaps
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add
            for x in range(1, n + 1):
                if not (S_mask & (1 << (x - 1))):
                    if can_add(x, S_mask):
                        S_mask |= (1 << (x - 1))
                        current_size += 1
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = S_mask
            if not improved:
                break
                
        # Now try random swaps
        # Randomly pick x not in S, y in S.
        # Check if S \ {y} U {x} is valid.
        # i.e., can_add(x, S_mask without y)
        
        max_swaps = 5000
        for _ in range(max_swaps):
            if time.time() >= deadline:
                break
                
            x = random.randint(1, n)
            if S_mask & (1 << (x - 1)):
                continue # x already in S
                
            # Pick y in S
            # To pick y uniformly from set bits:
            # Generate random number, mask with S, find set bit?
            # Or just maintain a list of elements in S? 
            # For N=512, list is fine.
            s_list = [i+1 for i in range(n) if S_mask & (1 << i)]
            if not s_list:
                break
            y = random.choice(s_list)
            
            # Try swap
            # S_new = S_mask ^ (1 << (y-1))
            S_mask_without_y = S_mask & ~(1 << (y - 1))
            
            if can_add(x, S_mask_without_y):
                # Accept swap
                S_mask = S_mask_without_y | (1 << (x - 1))
                # Size is same, but configuration improved potentially
                # We don't update best_size unless size increased (it didn't)
                # But this might allow future additions?
                # We don't re-run the "add" phase immediately, just continue swapping.
                # Or we could break and re-run add phase? 
                # Given time, continuing swaps is fine.
                
    # Convert best_mask to sorted list
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    if not result:
        # Fallback: return a known valid small set if something went wrong
        # E.g., {1, 2} is 4-free
        return [1, 2]
        
    return result
