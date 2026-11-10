import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = 12
    m = 40
    full_mask = (1 << n) - 1

    # Helper to check strict subset
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    # Precompute popcount for all masks
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Group masks by size for efficient candidate generation
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(1 << n):
        masks_by_size[popcount[mask]].append(mask)

    # Precompute comparability matrix for all pairs of masks
    # This is a 4096x4096 boolean matrix, which is 16M booleans = ~2MB if using bytearrays
    # Let's use a list of bytearrays for memory efficiency and speed
    comp_matrix = [bytearray(1 << n) for _ in range(1 << n)]
    for a in range(1 << n):
        for b in range(1 << n):
            if a != b and is_comparable(a, b):
                comp_matrix[a][b] = 1

    # Helper to calculate score of a set of masks
    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if comp_matrix[masks[i]][masks[j]]:
                    s += 1
        return s

    # Better greedy initialization: build a chain-based solution
    def greedy_init():
        current = []
        used = set()
        
        # Strategy: Build multiple chains that share endpoints
        # Start with empty set and full set
        current.append(0)
        used.add(0)
        current.append(full_mask)
        used.add(full_mask)
        
        # Add all singletons (comparable to empty set)
        for i in range(n):
            mask = 1 << i
            if mask not in used:
                current.append(mask)
                used.add(mask)
        
        # Add all co-singletons (comparable to full set)
        for i in range(n):
            mask = full_mask ^ (1 << i)
            if mask not in used:
                current.append(mask)
                used.add(mask)
        
        # Now greedily add masks that maximize comparability with existing set
        while len(current) < m:
            best_mask = -1
            best_count = -1
            
            # Try masks of various sizes, biased towards middle sizes
            for size in range(1, n):
                for mask in masks_by_size[size]:
                    if mask in used:
                        continue
                    count = 0
                    for cm in current:
                        if comp_matrix[mask][cm]:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_mask = mask
            
            if best_mask == -1:
                break
            
            current.append(best_mask)
            used.add(best_mask)
        
        return current

    random.seed(42)

    # Multiple restarts with different initialization strategies
    best_score = -1
    best_solution = None

    num_restarts = 8

    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Greedy initialization
            current_masks = greedy_init()
        elif restart == 1:
            # B_6 sublattice: vary bits 0-5
            base_masks = list(range(1 << 6))
            random.shuffle(base_masks)
            current_masks = base_masks[:m]
        elif restart == 2:
            # B_7 sublattice: vary bits 0-6, take first 40
            base_masks = list(range(1 << 7))
            random.shuffle(base_masks)
            current_masks = base_masks[:m]
        elif restart == 3:
            # B_5 sublattice (32 masks) + 8 from complementary region
            current_masks = list(range(1 << 5))  # bits 0-4
            used = set(current_masks)
            # Add 8 masks using bits 5-10
            for mask in range(1 << 6):
                full_m = mask << 5
                if full_m not in used and len(current_masks) < m:
                    current_masks.append(full_m)
                    used.add(full_m)
                if len(current_masks) >= m:
                    break
        elif restart == 4:
            # Two interleaved chains (bits 0-5 and bits 6-11)
            current_masks = []
            used = set()
            # Chain 1: bits 0-5
            current_set = 0
            for i in range(6):
                current_set |= (1 << i)
                if current_set not in used:
                    current_masks.append(current_set)
                    used.add(current_set)
            # Chain 2: bits 6-11
            current_set = 0
            for i in range(6, 12):
                current_set |= (1 << i)
                if current_set not in used:
                    current_masks.append(current_set)
                    used.add(current_set)
            # Fill with random
            while len(current_masks) < m:
                mask = random.randint(0, full_mask)
                if mask not in used:
                    current_masks.append(mask)
                    used.add(mask)
        elif restart == 5:
            # Star-like: one central set, many comparable to it
            center = 0
            for i in range(6):
                center |= (1 << i)
            current_masks = [center]
            used = {center}
            # Add subsets of center
            for mask in range(1 << 6):
                if mask != center and mask not in used and len(current_masks) < m:
                    current_masks.append(mask)
                    used.add(mask)
            # Add supersets of center
            for mask in range(1 << n):
                if (mask & center) == center and mask != center and mask not in used and len(current_masks) < m:
                    current_masks.append(mask)
                    used.add(mask)
            while len(current_masks) < m:
                mask = random.randint(0, full_mask)
                if mask not in used:
                    current_masks.append(mask)
                    used.add(mask)
        elif restart == 6:
            # B_6 with shuffled selection
            base_masks = list(range(1 << 6))
            random.shuffle(base_masks)
            current_masks = base_masks[:m]
        else:
            # Pure random
            current_masks = []
            used = set()
            while len(current_masks) < m:
                mask = random.randint(0, full_mask)
                if mask not in used:
                    current_masks.append(mask)
                    used.add(mask)
        
        used_masks = set(current_masks)
        score = calc_score(current_masks)
        
        # Simulated annealing
        T = 8.0
        T_min = 0.1
        cooling_rate = 0.9998
        no_improve_count = 0
        max_no_improve = 1000
        
        # Precompute for faster delta calculation
        # For each mask in current, store its comparability count with others
        comp_counts = [0] * m
        for i in range(m):
            for j in range(m):
                if i != j and comp_matrix[current_masks[i]][current_masks[j]]:
                    comp_counts[i] += 1
        
        # Build a set for O(1) membership testing
        current_set = set(current_masks)
        
        while time.time() < deadline - 3.0:
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate candidate using a smarter strategy:
            # 40%: random from full space
            # 30%: from B_6 sublattice (bits 0-5)
            # 20%: from B_6 sublattice (bits 6-11)
            # 10%: sample-based best of 5 random candidates
            
            r = random.random()
            if r < 0.4:
                new_mask = random.randint(0, full_mask)
            elif r < 0.7:
                new_mask = random.randint(0, (1 << 6) - 1)
            elif r < 0.9:
                new_mask = random.randint(0, (1 << 6) - 1)
                new_mask <<= 6
            else:
                # Sample a few random masks and pick the best
                best_cand = -1
                best_cand_score = -1
                for _ in range(5):
                    cand = random.randint(0, full_mask)
                    if cand in current_set or cand == old_mask:
                        continue
                    sc = 0
                    for j in range(m):
                        if j != idx and comp_matrix[cand][current_masks[j]]:
                            sc += 1
                    if sc > best_cand_score:
                        best_cand_score = sc
                        best_cand = cand
                if best_cand == -1:
                    new_mask = random.randint(0, full_mask)
                else:
                    new_mask = best_cand
            
            if new_mask == old_mask or new_mask in current_set:
                continue

            # Calculate delta efficiently
            # Old mask's contribution: comp_counts[idx]
            # New mask's contribution: count of new_mask comparable with other masks
            old_comp = comp_counts[idx]
            
            new_comp = 0
            for j in range(m):
                if j != idx and comp_matrix[new_mask][current_masks[j]]:
                    new_comp += 1
            
            delta = new_comp - old_comp
            
            # Simulated annealing acceptance
            if delta >= 0:
                # Accept
                current_set.remove(old_mask)
                current_set.add(new_mask)
                current_masks[idx] = new_mask
                score += delta
                comp_counts[idx] = new_comp
                no_improve_count = 0
                
                if score > best_score:
                    best_score = score
                    best_solution = list(current_masks)
            else:
                # Accept with probability exp(delta / T)
                if random.random() < math.exp(delta / T):
                    current_set.remove(old_mask)
                    current_set.add(new_mask)
                    current_masks[idx] = new_mask
                    score += delta
                    comp_counts[idx] = new_comp
                no_improve_count += 1
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
            
            # If stuck, restart with higher temperature
            if no_improve_count > max_no_improve:
                T = 4.0
                no_improve_count = 0
        
        # Final polish: Exact local search with exhaustive candidate evaluation
        improved = True
        polish_iterations = 0
        max_polish_iterations = 15
        while improved and time.time() < deadline - 1.0 and polish_iterations < max_polish_iterations:
            improved = False
            polish_iterations += 1
            
            current_set = set(current_masks)
            
            for i in range(m):
                if time.time() > deadline - 1.0:
                    break
                
                old_mask = current_masks[i]
                old_comp = comp_counts[i]
                
                # Try ALL 4096 possible masks as replacements
                best_new_mask = None
                best_new_comp = old_comp
                
                for new_mask in range(1 << n):
                    if new_mask == old_mask or new_mask in current_set:
                        continue
                    
                    new_comp = 0
                    for j in range(m):
                        if j != i and comp_matrix[new_mask][current_masks[j]]:
                            new_comp += 1
                    
                    if new_comp > best_new_comp:
                        best_new_comp = new_comp
                        best_new_mask = new_mask
                
                if best_new_mask is not None:
                    current_set.remove(old_mask)
                    current_set.add(best_new_mask)
                    current_masks[i] = best_new_mask
                    score += (best_new_comp - old_comp)
                    comp_counts[i] = best_new_comp
                    improved = True
                    
                    if score > best_score:
                        best_score = score
                        best_solution = list(current_masks)

    # Convert to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
