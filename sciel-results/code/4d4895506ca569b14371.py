import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    total_masks = 1 << n
    
    # Precompute popcount for all masks
    popcount = [0] * total_masks
    for mask in range(1, total_masks):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    # Precompute subset relationships: for each mask, list of all its subsets
    # and list of all its supersets (excluding itself)
    # Actually, let's precompute comparability more efficiently
    # For each pair (a, b), comparable if a&b == a or a&b == b and a != b
    
    # Better approach: precompute for each mask, which other masks are comparable
    # This is a 4096x4096 matrix which is ~16M entries - too much memory potentially
    # Let's use a different approach
    
    # Precompute: for each mask, the list of masks that are subsets of it (proper subsets)
    # and supersets of it (proper supersets)
    # Actually, let's just use bit operations in the inner loop but optimize
    
    def is_comparable(a, b):
        if a == b:
            return False
        return (a & b) == a or (a & b) == b
    
    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if is_comparable(masks[i], masks[j]):
                    s += 1
        return s
    
    def greedy_chain_init(seed_val):
        """Initialize with a chain structure, then greedily add sets that maximize comparability."""
        rng = random.Random(seed_val)
        elem_order = list(range(n))
        rng.shuffle(elem_order)
        
        # Build chain: empty set, then add elements one by one
        chain = [0]
        cur = 0
        for e in elem_order:
            cur |= (1 << e)
            chain.append(cur)
        
        current_masks = list(chain)  # 13 sets
        used_masks = set(current_masks)
        
        # Greedily add sets that maximize immediate comparability
        for _ in range(m - len(current_masks)):
            best_mask = -1
            best_score = -1
            # Sample a subset of candidates to speed up
            candidates = []
            # Try masks near existing ones
            for cm in current_masks:
                for bit in range(n):
                    if (cm >> bit) & 1:
                        candidates.append(cm ^ (1 << bit))
                    else:
                        candidates.append(cm | (1 << bit))
                # Also try random masks
                for _ in range(5):
                    candidates.append(random.randint(0, full_mask))
            
            for mask in set(candidates):
                if mask in used_masks:
                    continue
                score = 0
                for cm in current_masks:
                    if is_comparable(mask, cm):
                        score += 1
                if score > best_score:
                    best_score = score
                    best_mask = mask
            
            if best_mask == -1:
                # Fallback: find any unused mask
                for mask in range(total_masks):
                    if mask not in used_masks:
                        best_mask = mask
                        break
            
            current_masks.append(best_mask)
            used_masks.add(best_mask)
        
        return current_masks
    
    random.seed(42)
    
    # Precompute bit positions for each mask for faster moves
    bit_sets = {}
    for mask in range(total_masks):
        ones = [i for i in range(n) if (mask >> i) & 1]
        zeros = [i for i in range(n) if not ((mask >> i) & 1)]
        bit_sets[mask] = (ones, zeros)
    
    best_global_score = -1
    best_global_solution = None
    
    # Try multiple initializations
    num_restarts = 15
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Greedy chain initialization with different random seeds
        current_masks = greedy_chain_init(42 + r)
        used_masks = set(current_masks)
        
        cur_score = calc_score(current_masks)
        
        if cur_score > best_global_score:
            best_global_score = cur_score
            best_global_solution = list(current_masks)
        
        # Local Search: Hill Climbing with simulated annealing-like acceptance
        stuck = 0
        max_steps_per_restart = 50000
        temperature = 1.0
        cooling_rate = 0.9999
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 3.0:
                break
            
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate a new mask
            old_ones, old_zeros = bit_sets[old_mask]
            new_mask = old_mask
            
            op = random.random()
            if op < 0.30:
                # Add bit
                if old_zeros:
                    bit = random.choice(old_zeros)
                    new_mask = old_mask | (1 << bit)
            elif op < 0.60:
                # Remove bit
                if old_ones:
                    bit = random.choice(old_ones)
                    new_mask = old_mask ^ (1 << bit)
            elif op < 0.80:
                # Swap bit
                if old_ones and old_zeros:
                    bo = random.choice(old_ones)
                    bz = random.choice(old_zeros)
                    new_mask = (old_mask ^ (1 << bo)) | (1 << bz)
            elif op < 0.90:
                # Move to a comparable set: pick a random existing set and try to be comparable
                ref_mask = random.choice(current_masks)
                # Try subsets/supersets of ref_mask
                if random.random() < 0.5:
                    # Try subset
                    new_mask = ref_mask
                    for bit in range(n):
                        if (ref_mask >> bit) & 1 and random.random() < 0.5:
                            new_mask ^= (1 << bit)
                else:
                    # Try superset
                    new_mask = ref_mask
                    for bit in range(n):
                        if not ((ref_mask >> bit) & 1) and random.random() < 0.5:
                            new_mask |= (1 << bit)
            else:
                # Random jump
                new_mask = random.randint(0, full_mask)
                
            if new_mask == old_mask:
                continue
                
            if new_mask in used_masks:
                continue
                
            # Calculate Delta efficiently
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                mask_j = current_masks[j]
                old_comp = is_comparable(old_mask, mask_j)
                new_comp = is_comparable(new_mask, mask_j)
                delta += (1 if new_comp else 0) - (1 if old_comp else 0)
                
            # Accept if improvement, or with probability exp(-delta/temperature) if worsening
            if delta > 0:
                accept = True
            else:
                if temperature > 0.01 and delta > -10:
                    accept = random.random() < pow(2.718281828, -delta / max(temperature, 0.01))
                else:
                    accept = False
            
            if accept:
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                cur_score += delta
                stuck = 0
                if cur_score > best_global_score:
                    best_global_score = cur_score
                    best_global_solution = list(current_masks)
            else:
                stuck += 1
            
            # Cool down
            temperature *= cooling_rate
            
            if stuck > 100:
                # Random kick: replace 1-2 random elements
                kick_count = random.randint(1, 2)
                for _ in range(kick_count):
                    kidx = random.randint(0, m - 1)
                    old_k = current_masks[kidx]
                    while True:
                        new_k = random.randint(0, full_mask)
                        if new_k not in used_masks:
                            break
                    used_masks.remove(old_k)
                    used_masks.add(new_k)
                    current_masks[kidx] = new_k
                cur_score = calc_score(current_masks)
                stuck = 0
                temperature = max(temperature, 0.5)
        
        # Final polish: try all single-element moves until no improvement
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for idx in range(m):
                old_mask = current_masks[idx]
                old_ones, old_zeros = bit_sets[old_mask]
                # Try all neighbors
                neighbors = []
                for bit in old_zeros:
                    neighbors.append(old_mask | (1 << bit))
                for bit in old_ones:
                    neighbors.append(old_mask ^ (1 << bit))
                
                best_neighbor = -1
                best_delta = 0
                for nm in neighbors:
                    if nm in used_masks:
                        continue
                    delta = 0
                    for j in range(m):
                        if j == idx:
                            continue
                        mask_j = current_masks[j]
                        old_comp = is_comparable(old_mask, mask_j)
                        new_comp = is_comparable(nm, mask_j)
                        delta += (1 if new_comp else 0) - (1 if old_comp else 0)
                    if delta > best_delta:
                        best_delta = delta
                        best_neighbor = nm
                
                if best_neighbor != -1 and best_delta > 0:
                    used_masks.remove(old_mask)
                    used_masks.add(best_neighbor)
                    current_masks[idx] = best_neighbor
                    cur_score += best_delta
                    improved = True
                    if cur_score > best_global_score:
                        best_global_score = cur_score
                        best_global_solution = list(current_masks)
    
    # Convert best solution to list of lists
    result = []
    for mask in best_global_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
    
    return result
