import time
import random

def construct():
    n = 512
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    # Precompute conflicts
    # conflicts[x] is a list of bitmasks. If (S & mask) == mask, x cannot be added.
    # x is 1-indexed in problem, 0-indexed in bit operations.
    # Bit i corresponds to element i+1.
    
    conflicts = [[] for _ in range(n)]
    
    for x in range(1, n + 1):
        i = x - 1
        # Case 1: x is 1st term: x, x+d, x+2d, x+3d
        # Need x+d, x+2d, x+3d in S
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            y1 = x + d
            y2 = x + 2 * d
            y3 = x + 3 * d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 2: x is 2nd term: x-d, x, x+d, x+2d
        # Need x-d, x+d, x+2d in S
        max_d2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d2 + 1):
            y1 = x - d
            y2 = x + d
            y3 = x + 2 * d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 3: x is 3rd term: x-2d, x-d, x, x+d
        # Need x-2d, x-d, x+d in S
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            y1 = x - 2 * d
            y2 = x - d
            y3 = x + d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 4: x is 4th term: x-3d, x-2d, x-d, x
        # Need x-3d, x-2d, x-d in S
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            y1 = x - 3 * d
            y2 = x - 2 * d
            y3 = x - d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
    def can_add(x_idx, S):
        """Check if element at index x_idx (0-based) can be added to bitset S."""
        for mask in conflicts[x_idx]:
            if (S & mask) == mask:
                return False
        return True

    def greedy_construct(seed_order):
        """Generate a greedy set given a specific order."""
        S = 0
        for i in seed_order:
            if can_add(i, S):
                S |= (1 << i)
        return S

    def local_search(S):
        """Improve S using local search with kicks."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any element
            # We iterate through all not-in-S elements
            for i in range(n):
                if not (S & (1 << i)):
                    if can_add(i, S):
                        S |= (1 << i)
                        improved = True
            
            if improved:
                continue
                
            # If no direct addition, try a kick (remove 1 or 2 elements)
            set_bits = []
            temp_S = S
            for i in range(n):
                if temp_S & (1):
                    set_bits.append(i)
                temp_S >>= 1
            
            if not set_bits:
                break
                
            # Try removing 1 random element
            rem_idx = random.choice(set_bits)
            S_new = S & ~(1 << rem_idx)
            
            # Try to fill S_new greedily
            S_candidate = S_new
            # Randomize order for filling to escape local optima?
            # Or just sequential? Sequential is faster.
            for i in range(n):
                if not (S_candidate & (1 << i)):
                    if can_add(i, S_candidate):
                        S_candidate |= (1 << i)
            
            if S_candidate.bit_count() > S.bit_count():
                S = S_candidate
                improved = True
            else:
                # Try removing 2 random elements
                if len(set_bits) >= 2:
                    rem1, rem2 = random.sample(set_bits, 2)
                    S_new = S & ~(1 << rem1) & ~(1 << rem2)
                    S_candidate = S_new
                    for i in range(n):
                        if not (S_candidate & (1 << i)):
                            if can_add(i, S_candidate):
                                S_candidate |= (1 << i)
                    
                    if S_candidate.bit_count() > S.bit_count():
                        S = S_candidate
                        improved = True
                    else:
                        # If we couldn't improve with kicks, we might be stuck.
                        # To avoid infinite loop, we break or do a more drastic restart.
                        # Since we are in a while loop, if improved is False, we break.
                        break
                        
        return S

    best_S = 0
    best_size = 0
    
    # Multiple restarts with different random orders
    while time.time() < deadline:
        order = list(range(n))
        random.shuffle(order)
        
        S = greedy_construct(order)
        S = local_search(S)
        
        if S.bit_count() > best_size:
            best_S = S
            best_size = S.bit_count()
            
        # If we have found a very large set, we can spend more time refining it?
        # Or just keep restarting.
        
    # Convert bitset to sorted list
    result = []
    for i in range(n):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
