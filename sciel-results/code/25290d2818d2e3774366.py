import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 512
    
    # Precompute all 4-APs with bitmask conflict representation
    # conflict_triples[x] = list of masks; each mask has bits set for the other 3 elements
    # If (mask & S_mask) == mask, then adding x would complete a 4-AP
    conflict_triples = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            term4 = a + 3 * d
            
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[a].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[b].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (term4 - 1))
            conflict_triples[c].append(mask_c)
            
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflict_triples[term4].append(mask_d)
    
    conflict_triples = [tuple(lst) for lst in conflict_triples]
    
    def can_add(x, S_mask):
        """Check if x can be added to S without forming a 4-AP using bitmask."""
        for mask in conflict_triples[x]:
            if (mask & S_mask) == mask:
                return False
        return True
    
    def count_conflicts(x, S_mask):
        """Count how many 4-APs would be completed if x were added."""
        count = 0
        for mask in conflict_triples[x]:
            if (mask & S_mask) == mask:
                count += 1
        return count
    
    def try_add_all(S_mask, S_list):
        """Try to add all possible elements. Returns (new_mask, new_list, added_count)."""
        added = 0
        for x in range(1, n + 1):
            if not (S_mask & (1 << (x - 1))):
                if can_add(x, S_mask):
                    S_mask |= (1 << (x - 1))
                    S_list.append(x)
                    added += 1
        return S_mask, S_list, added
    
    def greedy_removal():
        """
        Hitting set removal approach:
        Start with all elements, greedily remove to break all 4-APs,
        then add back elements that can be safely included.
        """
        # Start with all elements
        S_mask = (1 << n) - 2  # bits 1..n set (bit 0 unused)
        S_list = list(range(1, n + 1))
        
        # Compute degree: for each element, how many 4-APs it participates in
        # We need to track which APs are still "active" (all 4 elements present)
        # Instead, we'll use a simpler approach: repeatedly find an element 
        # that is in the most active APs and remove it.
        
        # Precompute for each element, the list of AP indices it's in
        # and for each AP, the 4 elements
        ap_elements = []  # list of (a,b,c,d)
        elem_to_aps = [[] for _ in range(n + 1)]
        
        idx = 0
        for a in range(1, n + 1):
            max_d = (n - a) // 3
            for d in range(1, max_d + 1):
                b = a + d
                c = a + 2 * d
                term4 = a + 3 * d
                ap_elements.append((a, b, c, term4))
                elem_to_aps[a].append(idx)
                elem_to_aps[b].append(idx)
                elem_to_aps[c].append(idx)
                elem_to_aps[term4].append(idx)
                idx += 1
        
        num_aps = len(ap_elements)
        
        # Track which APs are still active (all 4 elements in S)
        # An AP is active if all 4 elements are present
        # We'll track for each element how many active APs it's in
        deg = [0] * (n + 1)
        
        # Initialize degrees
        for i in range(num_aps):
            a, b, c, d = ap_elements[i]
            deg[a] += 1
            deg[b] += 1
            deg[c] += 1
            deg[d] += 1
        
        # Greedily remove elements with highest degree
        # Use a priority approach: repeatedly pick element with max degree
        in_set = [True] * (n + 1)
        
        # To speed up, we'll do multiple passes
        while time.time() < deadline:
            # Find element with max degree among those in set
            max_deg = -1
            best_elem = -1
            for i in range(1, n + 1):
                if in_set[i] and deg[i] > max_deg:
                    max_deg = deg[i]
                    best_elem = i
            
            if max_deg == 0 or best_elem == -1:
                break
            
            # Remove best_elem
            in_set[best_elem] = False
            S_mask &= ~(1 << (best_elem - 1))
            
            # Update degrees: for each AP containing best_elem, 
            # if it was active, decrement degree of other elements
            for ap_idx in elem_to_aps[best_elem]:
                a, b, c, d = ap_elements[ap_idx]
                # Check if this AP was active (all 4 elements in set before removal)
                # Since we're removing best_elem, it was in set
                # The AP was active if the other 3 were also in set
                others = [x for x in (a, b, c, d) if x != best_elem]
                if all(in_set[x] for x in others):
                    # This AP was active, now it's broken
                    for x in others:
                        deg[x] -= 1
            
            # Check if we're done (no active APs)
            # Quick check: if max degree is 0, we're done
            has_active = False
            for i in range(1, n + 1):
                if in_set[i] and deg[i] > 0:
                    has_active = True
                    break
            if not has_active:
                break
        
        # Build S_list from in_set
        S_list = [i for i in range(1, n + 1) if in_set[i]]
        
        # Add-back phase: try to add back removed elements
        removed = [i for i in range(1, n + 1) if not in_set[i]]
        random.shuffle(removed)
        
        for x in removed:
            if can_add(x, S_mask):
                S_mask |= (1 << (x - 1))
                S_list.append(x)
                in_set[x] = True
        
        return S_mask, S_list
    
    def simulated_annealing(S_mask, S_list, max_iters=20000):
        """
        Simulated annealing polish:
        - Try adding random elements not in S
        - Try removing random elements from S (with temperature-based acceptance)
        - Try swaps
        - Accept worse solutions with probability exp(-delta/T)
        """
        T = 5.0  # Initial temperature
        T_min = 0.1
        cooling = 0.9995
        
        best_mask = S_mask
        best_size = len(S_list)
        current_mask = S_mask
        current_size = len(S_list)
        current_list = list(S_list)
        
        for it in range(max_iters):
            if time.time() >= deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random element not in S
                x = random.randint(1, n)
                if current_mask & (1 << (x - 1)):
                    continue
                if can_add(x, current_mask):
                    new_mask = current_mask | (1 << (x - 1))
                    new_size = current_size + 1
                    # Always accept improvements
                    current_mask = new_mask
                    current_size = new_size
                    current_list.append(x)
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                else:
                    # Try swap: remove a conflicting element
                    # Find a conflicting AP
                    conflicting_y = -1
                    for mask in conflict_triples[x]:
                        if (mask & current_mask) == mask:
                            # Find one of the elements in mask that's in current_list
                            for y in current_list:
                                if current_mask & (1 << (y - 1)):
                                    # Check if y is in this particular mask
                                    if mask & (1 << (y - 1)):
                                        conflicting_y = y
                                        break
                            if conflicting_y != -1:
                                break
                    
                    if conflicting_y != -1:
                        # Try swap: remove conflicting_y, add x
                        new_mask = (current_mask & ~(1 << (conflicting_y - 1))) | (1 << (x - 1))
                        # Check if this is valid (x can be added after removing conflicting_y)
                        if can_add(x, new_mask):
                            delta = 0  # Same size
                            # Accept with some probability even if same size
                            if random.random() < 0.5:
                                current_mask = new_mask
                                # Update list
                                current_list[current_list.index(conflicting_y)] = x
                                if current_size > best_size:
                                    best_size = current_size
                                    best_mask = current_mask
                        
            elif move_type < 0.7:
                # Try to remove a random element from S
                if not current_list:
                    continue
                y = random.choice(current_list)
                new_mask = current_mask & ~(1 << (y - 1))
                new_size = current_size - 1
                delta = new_size - current_size  # = -1
                
                # Accept with probability exp(delta/T) = exp(-1/T)
                if random.random() < min(1.0, pow(2.718281828, delta / T)):
                    current_mask = new_mask
                    current_size = new_size
                    current_list.remove(y)
                    
                    # After removal, try to add back more elements
                    added = 0
                    for x in range(1, n + 1):
                        if not (current_mask & (1 << (x - 1))):
                            if can_add(x, current_mask):
                                current_mask |= (1 << (x - 1))
                                current_list.append(x)
                                current_size += 1
                                added += 1
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                        
            else:
                # Try swap: remove random y from S, add random x not in S
                if not current_list:
                    continue
                x = random.randint(1, n)
                if current_mask & (1 << (x - 1)):
                    continue
                y = random.choice(current_list)
                
                new_mask = (current_mask & ~(1 << (y - 1))) | (1 << (x - 1))
                if can_add(x, new_mask):
                    delta = 0  # Same size
                    if random.random() < 0.3:
                        current_mask = new_mask
                        current_list[current_list.index(y)] = x
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
            
            # Cool down
            T = max(T_min, T * cooling)
        
        return best_mask, best_size
    
    best_mask = 0
    best_size = 0
    
    while time.time() < deadline:
        # 1. Hitting set removal construction
        S_mask, S_list = greedy_removal()
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
        
        # 2. Exhaustive add pass
        S_mask, S_list, _ = try_add_all(S_mask, S_list)
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
        
        # 3. Simulated annealing polish
        if time.time() < deadline - 10:  # Reserve 10s for final cleanup
            S_mask, current_size = simulated_annealing(S_mask, S_list, max_iters=15000)
            if current_size > best_size:
                best_size = current_size
                best_mask = S_mask
        
        # 4. Final exhaustive add
        S_list = []
        for i in range(n):
            if best_mask & (1 << i):
                S_list.append(i + 1)
        S_mask, S_list, _ = try_add_all(best_mask, S_list)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_mask = S_mask
    
    # Convert best_mask to sorted list
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i + 1)
    
    if not result:
        return [1, 2]
        
    return result
