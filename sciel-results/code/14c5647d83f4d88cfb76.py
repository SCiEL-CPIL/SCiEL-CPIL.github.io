import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 512
    
    # Precompute all 4-APs
    # conflict_triples[x] is a list of masks. 
    # If x is in S, and (mask & S_mask) == mask, then a 4-AP exists.
    # The mask contains the other 3 elements.
    conflict_triples = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs (a, a+d, a+2d, a+3d)
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            term4 = a + 3 * d
            
            # For each element in the AP, the other 3 form the conflict mask
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[a].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[b].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (term4 - 1))
            conflict_triples[c].append(mask_c)
            
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflict_triples[term4].append(mask_d)
            
    # Convert to tuples for faster iteration
    conflict_triples = [tuple(lst) for lst in conflict_triples]
    
    bit_positions = {i: 1 << (i - 1) for i in range(1, n + 1)}
    
    def can_add(x, S_mask):
        """Check if x can be added to S without forming a 4-AP."""
        for mask in conflict_triples[x]:
            if (mask & S_mask) == mask:
                return False
        return True

    def greedy_fill(S_mask, S_list, order):
        """Try to add all possible elements in the given order."""
        added_any = False
        for x in order:
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask):
                    S_mask |= bit_positions[x]
                    S_list.append(x)
                    added_any = True
        return S_mask, S_list, added_any

    best_mask = 0
    best_size = 0
    all_elements = list(range(1, n + 1))
    
    # Run multiple restarts to find a good solution
    # Each restart: Greedy + Local Search
    while time.time() < deadline:
        # 1. Greedy Construction with random shuffle
        S_mask = 0
        S_list = []
        candidates = all_elements[:]
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S_mask):
                S_mask |= bit_positions[x]
                S_list.append(x)
        
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
            
        # 2. Local Search Phase
        # We will try to improve the set by swapping elements.
        # Strategy: 
        #   1. Try to add any element.
        #   2. If stuck, perform a random swap (remove y, add x) if valid.
        #   3. Repeat.
        
        # Precompute list of elements NOT in S for faster random selection?
        # Or just pick random x and check.
        
        # To speed up "pick random x not in S", we can maintain a list of candidates
        # But rebuilding it is expensive. Just picking random 1..n and checking is O(1).
        
        no_improvement_count = 0
        max_no_improvement = 20000 # Tune this
        
        while time.time() < deadline:
            # Try to add an element
            added = False
            # Try a few random additions
            for _ in range(10):
                x = random.randint(1, n)
                if not (S_mask & bit_positions[x]):
                    if can_add(x, S_mask):
                        S_mask |= bit_positions[x]
                        S_list.append(x)
                        current_size += 1
                        added = True
                        no_improvement_count = 0
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = S_mask
                        break # Restart loop if added
            
            if added:
                continue
                
            # If no element could be added, try a swap
            # Pick random y in S to remove
            y_idx = random.randint(0, len(S_list) - 1)
            y = S_list[y_idx]
            
            # Pick random x not in S to add
            # We need to ensure x is not in S
            # Try a few times to find a valid x
            swap_made = False
            for _ in range(5):
                x = random.randint(1, n)
                if S_mask & bit_positions[x]:
                    continue
                
                # Check if x can be added to S \ {y}
                S_mask_without_y = S_mask & ~bit_positions[y]
                if can_add(x, S_mask_without_y):
                    # Perform swap
                    S_mask = S_mask_without_y | bit_positions[x]
                    S_list[y_idx] = x
                    swap_made = True
                    no_improvement_count = 0
                    break
            
            if not swap_made:
                no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    break # Stuck, break inner loop and restart outer loop
            
        # Final add pass after local search
        S_mask, S_list, _ = greedy_fill(S_mask, S_list, all_elements)
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask

    # Final polish: try to add any remaining elements to the best solution
    final_list = []
    for i in range(n):
        if best_mask & (1 << i):
            final_list.append(i + 1)
    
    if not final_list:
        return [1, 2]
        
    return final_list
