import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    all_subsets = list(range(1 << n))
    
    def is_comparable(a, b):
        if a == b:
            return False
        return (a & b) == a or (a & b) == b
    
    def count_comparables(mask, others):
        count = 0
        for o in others:
            if is_comparable(mask, o):
                count += 1
        return count
    
    def build_initial_structure():
        """Build a good initial structure using multiple chains."""
        masks = []
        mask_set = set()
        
        def add(mask):
            if mask not in mask_set:
                masks.append(mask)
                mask_set.add(mask)
        
        # Build a long chain: empty -> {0} -> {0,1} -> {0,1,2} -> ... -> full
        chain1 = [0]
        cur = 0
        for i in range(n):
            cur |= (1 << i)
            chain1.append(cur)
        for c in chain1:
            add(c)
        
        # Build a second chain that shares the empty and full sets
        # Chain 2: {1} -> {1,2} -> {1,2,3} -> ... -> {1,2,...,11}
        # But we need to avoid duplicates. Let's build chains that are "parallel"
        # Chain 2: start from {0,1}, add elements in different order
        # Actually, let's build chains that maximize cross-comparability
        
        # Strategy: Build sets by size layers, preferring sets that are comparable to many existing sets
        # Start with empty, full, and build up
        
        # Add all singletons
        for i in range(n):
            add(1 << i)
        
        # Add all size-11 sets
        for i in range(n):
            add(full_mask ^ (1 << i))
        
        # Now greedily add remaining sets
        while len(masks) < m:
            if time.time() > deadline:
                break
            best_gain = -1
            best_mask = None
            for c in all_subsets:
                if c in mask_set:
                    continue
                gain = count_comparables(c, masks)
                if gain > best_gain:
                    best_gain = gain
                    best_mask = c
            if best_mask is None or best_gain <= 0:
                break
            add(best_mask)
        
        # Fill with random if needed
        while len(masks) < m:
            c = random.choice(all_subsets)
            if c not in mask_set:
                add(c)
        
        return masks
    
    # Try multiple initial structures and keep the best
    best_masks = None
    best_score = -1
    
    num_starts = 5
    for start_idx in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        if start_idx == 0:
            masks = build_initial_structure()
        else:
            # Perturbed start: take the best so far and randomly swap some
            if best_masks is None:
                masks = build_initial_structure()
            else:
                masks = best_masks[:]
                # Randomly swap 5-10 sets
                num_swaps = random.randint(5, 10)
                mask_set = set(masks)
                for _ in range(num_swaps):
                    idx = random.randint(0, m - 1)
                    old = masks[idx]
                    new = random.choice(all_subsets)
                    if new not in mask_set:
                        mask_set.remove(old)
                        mask_set.add(new)
                        masks[idx] = new
        
        # Compute initial score
        score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if is_comparable(masks[i], masks[j]):
                    score += 1
        
        if score > best_score:
            best_score = score
            best_masks = masks[:]
        
        # Local search on this structure
        current_masks = masks[:]
        current_set = set(current_masks)
        current_score = score
        
        # Build comparability matrix
        comp = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                if is_comparable(current_masks[i], current_masks[j]):
                    comp[i][j] = 1
                    comp[j][i] = 1
        
        # Simulated annealing local search
        temperature = 2.0
        cooling = 0.9995
        min_temp = 0.01
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 5:
            if no_improve >= max_no_improve:
                if temperature > min_temp:
                    temperature = max(min_temp, temperature * 0.5)
                    no_improve = 0
                    max_no_improve = int(max_no_improve * 1.5)
                else:
                    break
            
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            new_mask = random.choice(all_subsets)
            
            if new_mask == old_mask or new_mask in current_set:
                continue
            
            # Compute delta
            old_contrib = 0
            for j in range(m):
                if j != idx:
                    old_contrib += comp[idx][j]
            
            new_contrib = 0
            for j in range(m):
                if j != idx:
                    if is_comparable(new_mask, current_masks[j]):
                        new_contrib += 1
            
            delta = new_contrib - old_contrib
            
            if delta > 0:
                # Accept
                for j in range(m):
                    if j != idx:
                        comp[idx][j] = 0
                        comp[j][idx] = 0
                        if is_comparable(new_mask, current_masks[j]):
                            comp[idx][j] = 1
                            comp[j][idx] = 1
                
                current_masks[idx] = new_mask
                current_set.remove(old_mask)
                current_set.add(new_mask)
                current_score += delta
                no_improve = 0
                temperature = max(min_temp, temperature * cooling)
            else:
                # Simulated annealing: accept with probability exp(delta/T)
                if temperature > min_temp:
                    import math
                    prob = math.exp(delta / temperature)
                    if random.random() < prob:
                        for j in range(m):
                            if j != idx:
                                comp[idx][j] = 0
                                comp[j][idx] = 0
                                if is_comparable(new_mask, current_masks[j]):
                                    comp[idx][j] = 1
                                    comp[j][idx] = 1
                        
                        current_masks[idx] = new_mask
                        current_set.remove(old_mask)
                        current_set.add(new_mask)
                        current_score += delta
                        no_improve = 0
                        temperature = max(min_temp, temperature * cooling)
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            if current_score > best_score:
                best_score = current_score
                best_masks = current_masks[:]
    
    # Convert best masks to lists
    result = []
    for mask in best_masks:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i)
        result.append(s)
    
    return result
