import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Precompute LCM table for O(1) lookups
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            lcm_table[a][b] = a // gcd(a, b) * b
    
    # First, find all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == ab and lcm_table[b][c] == ab:
                    bad_triples.append((a, b, c))
                    
    # Build conflict structures
    # For each x, we want to know which pairs (u, v) in the rest of the set would block x.
    # conflict_pairs[x] = set of (u, v) with u < v, u != x, v != x, such that {u, v, x} is bad.
    conflict_pairs = [set() for _ in range(n + 1)]
    for a, b, c in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
        
    # Precompute conflict bitmasks for faster checking
    # For each x, a list of (u, v_mask) where v_mask is bitmask of elements that conflict with x when u is present
    conflict_list = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        temp_dict = {}
        for u, v in conflict_pairs[x]:
            if u not in temp_dict:
                temp_dict[u] = 0
            temp_dict[u] |= (1 << (v - 1))
            if v not in temp_dict:
                temp_dict[v] = 0
            temp_dict[v] |= (1 << (u - 1))
        conflict_list[x] = list(temp_dict.items())
    
    def can_add(x, mask):
        # Check if x can be added to the set represented by mask
        # x conflicts with pair (u,v) if both u and v are in mask
        for u, v_mask in conflict_list[x]:
            if mask & (1 << (u - 1)):
                if mask & v_mask:
                    return False
        return True

    def greedy(order, mask=0):
        selected = mask
        for x in order:
            if not (selected & (1 << (x - 1))):
                if can_add(x, selected):
                    selected |= (1 << (x - 1))
        return selected

    def popcount(x):
        return bin(x).count('1')
    
    def local_search(mask, max_iters=50):
        improved = True
        iters = 0
        s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
        
        while improved and time.time() < deadline - 2 and iters < max_iters:
            improved = False
            iters += 1
            
            # 1. Try to add any missing element
            for x in range(1, n + 1):
                if not (mask & (1 << (x - 1))):
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        s_list.append(x)
                        improved = True
                        break
            if improved:
                continue

            # 2. Try 1-for-2 swap: remove x, add y, z
            found_swap = False
            random.shuffle(s_list)
            for x in s_list:
                if time.time() > deadline - 2: break
                mask_no_x = mask & ~(1 << (x - 1))
                # Try to add two elements
                added_count = 0
                new_mask = mask_no_x
                for y in range(1, n + 1):
                    if y == x: continue
                    if not (new_mask & (1 << (y - 1))):
                        if can_add(y, new_mask):
                            new_mask |= (1 << (y - 1))
                            added_count += 1
                            if added_count == 2:
                                break
                if added_count >= 2:
                    if popcount(new_mask) > popcount(mask):
                        mask = new_mask
                        s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                        improved = True
                        found_swap = True
                        break
            if found_swap:
                continue

            # 3. Try 2-for-3 swap: remove x, y, add z, w, v
            # This is more expensive but can escape deeper local minima
            if time.time() < deadline - 3:
                random.shuffle(s_list)
                for i in range(min(len(s_list), 15)):
                    if time.time() > deadline - 3: break
                    x = s_list[i]
                    for j in range(i+1, min(len(s_list), i+8)):
                        y = s_list[j]
                        mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                        # Try to add three elements
                        added_count = 0
                        new_mask = mask_no_xy
                        for z in range(1, n + 1):
                            if z == x or z == y: continue
                            if not (new_mask & (1 << (z - 1))):
                                if can_add(z, new_mask):
                                    new_mask |= (1 << (z - 1))
                                    added_count += 1
                                    if added_count == 3:
                                        break
                        if added_count >= 3:
                            if popcount(new_mask) > popcount(mask):
                                mask = new_mask
                                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    continue

            # 4. Random perturbation if stuck
            if not improved and iters > 3:
                # Remove 2-3 random elements and try to rebuild
                if len(s_list) > 5:
                    remove_count = random.randint(2, 3)
                    remove_idx = random.sample(range(len(s_list)), min(remove_count, len(s_list)))
                    perturb_mask = mask
                    for idx in remove_idx:
                        perturb_mask &= ~(1 << (s_list[idx] - 1))
                    # Greedily add back
                    for x in range(1, n + 1):
                        if not (perturb_mask & (1 << (x - 1))):
                            if can_add(x, perturb_mask):
                                perturb_mask |= (1 << (x - 1))
                    if popcount(perturb_mask) > popcount(mask):
                        mask = perturb_mask
                        s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                        improved = True
                        continue

            if not improved:
                break
                
        return mask
    
    def simulated_annealing(mask, max_iters=200):
        """Simulated annealing to escape local minima"""
        current_mask = mask
        current_size = popcount(current_mask)
        best_mask = current_mask
        best_size = current_size
        
        # Temperature schedule
        T_start = 2.0
        T_end = 0.01
        T = T_start
        
        s_list = [i for i in range(1, n + 1) if (current_mask & (1 << (i - 1)))]
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Cool down
            T = T_start * (T_end / T_start) ** (it / max_iters)
            
            # Generate a neighbor: either add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add an element
                candidates = [x for x in range(1, n + 1) if not (current_mask & (1 << (x - 1)))]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if can_add(x, current_mask):
                    new_mask = current_mask | (1 << (x - 1))
                    new_size = current_size + 1
                    delta = new_size - current_size
                    # Always accept improvements
                    current_mask = new_mask
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                    s_list.append(x)
                else:
                    # Can't add directly, so skip
                    continue
            elif move_type < 0.6:
                # Try to remove an element
                if not s_list:
                    continue
                x = random.choice(s_list)
                new_mask = current_mask & ~(1 << (x - 1))
                new_size = current_size - 1
                delta = new_size - current_size
                # Accept with probability exp(-delta/T) if delta < 0
                if delta < 0 and random.random() < pow(2.718281828, -delta / max(T, 0.001)):
                    current_mask = new_mask
                    current_size = new_size
                    s_list.remove(x)
            else:
                # Try a swap: remove x, add y
                if not s_list:
                    continue
                x = random.choice(s_list)
                candidates = [y for y in range(1, n + 1) if y != x and not (current_mask & (1 << (y - 1)))]
                if not candidates:
                    continue
                y = random.choice(candidates)
                mask_no_x = current_mask & ~(1 << (x - 1))
                if can_add(y, mask_no_x):
                    new_mask = mask_no_x | (1 << (y - 1))
                    new_size = current_size  # Same size
                    delta = 0
                    # Accept with some probability for exploration
                    if random.random() < 0.5 or T > 0.5:
                        current_mask = new_mask
                        s_list.remove(x)
                        s_list.append(y)
                        # Try to add more elements after swap
                        for z in range(1, n + 1):
                            if not (current_mask & (1 << (z - 1))):
                                if can_add(z, current_mask):
                                    current_mask |= (1 << (z - 1))
                                    current_size += 1
                                    s_list.append(z)
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
        
        return best_mask

    best_mask = 0
    best_size = 0
    
    # Precompute degrees for sorting
    degrees = [len(conflict_pairs[k]) for k in range(n+1)]
    
    # Run multiple randomized greedy passes
    num_restarts = 500
    
    for i in range(num_restarts):
        if time.time() > deadline - 10: # Leave time for final polish and SA
            break
            
        # Different ordering strategies
        strategy = i % 5
        if strategy == 0:
            # Low degree first
            order = sorted(range(1, n + 1), key=lambda x: degrees[x] + random.random()*0.1)
        elif strategy == 1:
            # High degree first
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x] + random.random()*0.1)
        elif strategy == 2:
            # Random
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif strategy == 3:
            # Reverse order
            order = list(range(n, 0, -1))
            random.shuffle(order)
        else:
            # Degree-weighted random
            weights = [degrees[x] + 1 for x in range(1, n + 1)]
            order = list(range(1, n + 1))
            random.shuffle(order)
            # Sort by weighted random
            order = sorted(order, key=lambda x: random.random() / weights[x-1])
            
        mask = greedy(order)
        mask = local_search(mask, max_iters=30)
        
        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # Final polish with local search
    best_mask = local_search(best_mask, max_iters=50)
    best_size = popcount(best_mask)
    
    # Simulated annealing for final improvement
    if time.time() < deadline - 5:
        sa_mask = simulated_annealing(best_mask, max_iters=100)
        sa_size = popcount(sa_mask)
        if sa_size > best_size:
            best_size = sa_size
            best_mask = sa_mask
    
    # One more local search pass
    best_mask = local_search(best_mask, max_iters=30)
    
    result = [i for i in range(1, n + 1) if (best_mask & (1 << (i - 1)))]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result
