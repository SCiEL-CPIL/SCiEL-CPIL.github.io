import itertools
import time
import random

def construct(q, n, radius):
    # This solution is specialized for q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback to tetracode direct sum for other cases (though task is fixed)
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(itertools.product(range(3), repeat=n))[0]] # Dummy

    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # Precompute neighbors for all 729 words
    neighbors = []
    for c in range(729):
        digs = [(c // (3**i)) % 3 for i in range(6)]
        ns = [c]
        for i in range(6):
            for delta in [-1, 1]:
                new_dig = (digs[i] + delta) % 3
                weight = 3**i
                old_val = digs[i] * weight
                new_val = new_dig * weight
                ns.append(c - old_val + new_val)
        neighbors.append(ns)
    
    # Precompute neighbor bitmasks (using Python ints as arbitrary-precision bitsets)
    # bit w set in neighbor_mask[c] means word w is within distance 1 of codeword c
    neighbor_mask = [0] * 729
    for c in range(729):
        mask = 0
        for w in neighbors[c]:
            mask |= (1 << w)
        neighbor_mask[c] = mask
    
    # Initial set: Tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codes = []
    for h in tetracode:
        for t in itertools.product(range(3), repeat=2):
            code_digits = h + list(t)
            val = 0
            for i, d in enumerate(code_digits):
                val += d * (3**i)
            initial_codes.append(val)
    
    current_codes = list(set(initial_codes))
    code_set = set(current_codes)
    
    # Coverage count per word
    cov = [0] * 729
    # For each word, track which codewords cover it (as a set for intersection)
    coverers = [set() for _ in range(729)]
    
    for c in current_codes:
        for w in neighbors[c]:
            cov[w] += 1
            coverers[w].add(c)
            
    best_size = len(current_codes)
    best_codes = current_codes[:]
    
    def remove_code(c):
        for w in neighbors[c]:
            cov[w] -= 1
            coverers[w].remove(c)
        code_set.remove(c)
        current_codes.remove(c)
        
    def add_code(c):
        for w in neighbors[c]:
            cov[w] += 1
            coverers[w].add(c)
        code_set.add(c)
        current_codes.append(c)
        
    def is_removable(c):
        for w in neighbors[c]:
            if cov[w] == 1:
                return False
        return True
        
    def try_removals():
        removed = False
        while True:
            any_removed = False
            for c in current_codes[:]:
                if is_removable(c):
                    remove_code(c)
                    any_removed = True
                    removed = True
            if not any_removed:
                break
        return removed

    def try_swaps():
        """Try to find a beneficial swap: remove one codeword, add another that covers its exclusive words."""
        swapped = False
        # Find codewords with exclusive words (words only they cover)
        candidates_for_swap = []
        for c in current_codes:
            exclusive = []
            for w in neighbors[c]:
                if cov[w] == 1:
                    exclusive.append(w)
            if exclusive:
                candidates_for_swap.append((len(exclusive), c, exclusive))
                
        if not candidates_for_swap:
            return False
            
        # Sort by size of exclusive set (smallest first - easier to cover)
        candidates_for_swap.sort(key=lambda x: x[0])
        
        for size, c, exclusive in candidates_for_swap:
            # Find codewords that cover ALL exclusive words
            # Start with coverers of first exclusive word, intersect with rest
            common = set(coverers[exclusive[0]])
            for w in exclusive[1:]:
                common &= coverers[w]
                if not common:
                    break
            
            if not common:
                continue
            
            # Remove codewords already in our set
            common -= code_set
            if not common:
                continue
            
            # Try each candidate new codeword; pick one that maximizes additional coverage
            # (i.e., covers the most currently-uncovered or low-coverage words)
            best_new = None
            best_gain = -1
            for new_c in list(common)[:50]:  # Limit to 50 candidates for speed
                # Estimate gain: number of words where cov would increase
                gain = 0
                for w in neighbors[new_c]:
                    if cov[w] < 2:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_new = new_c
            
            if best_new is not None:
                remove_code(c)
                add_code(best_new)
                swapped = True
                break
        
        return swapped

    def try_multi_swaps():
        """Try removing up to 2 codewords and adding 1-2 new ones."""
        # For now, just try single swaps with multiple attempts
        # This is a placeholder for more complex moves
        return try_swaps()

    # Main Loop with simulated annealing-like perturbation
    no_improve_count = 0
    max_no_improve = 30
    temperature = 100.0
    cooling_rate = 0.95
    
    while time.time() < deadline:
        # 1. Try removing
        if try_removals():
            if len(current_codes) < best_size:
                best_size = len(current_codes)
                best_codes = current_codes[:]
                no_improve_count = 0
                temperature = 100.0  # Reset temperature on improvement
            else:
                no_improve_count += 1
            continue
            
        # 2. Try swapping
        if try_swaps():
            # After a swap, coverage changed; loop back to try removals
            no_improve_count = 0
            continue
            
        # 3. Stuck. Perturb based on temperature
        if no_improve_count > max_no_improve:
            # Revert to best and try a different random seed
            current_codes[:] = best_codes[:]
            code_set = set(current_codes)
            cov = [0] * 729
            coverers = [set() for _ in range(729)]
            for c in current_codes:
                for w in neighbors[c]:
                    cov[w] += 1
                    coverers[w].add(c)
            no_improve_count = 0
            
            # Add 3-5 random codewords for stronger perturbation
            num_perturb = random.randint(3, 5)
            for _ in range(num_perturb):
                for attempt in range(30):
                    r = random.randint(0, 728)
                    if r not in code_set:
                        add_code(r)
                        break
        
        else:
            # Temperature-based perturbation: higher temp = more aggressive
            num_perturb = max(1, int(temperature / 25))
            for _ in range(num_perturb):
                for attempt in range(30):
                    r = random.randint(0, 728)
                    if r not in code_set:
                        add_code(r)
                        break
        
        no_improve_count += 1
        temperature *= cooling_rate
        
        # After perturbation, run removals to get back down
        try_removals()
        
        if len(current_codes) < best_size:
            best_size = len(current_codes)
            best_codes = current_codes[:]
            no_improve_count = 0
            temperature = 100.0  # Reset temperature on improvement
        
    # Final polish: try more aggressive swaps and removals
    # Run multiple passes of removals and swaps
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        if try_removals():
            if len(current_codes) < best_size:
                best_size = len(current_codes)
                best_codes = current_codes[:]
        if try_swaps():
            if len(current_codes) < best_size:
                best_size = len(current_codes)
                best_codes = current_codes[:]
    
    # Convert best_codes back to list of lists
    result = []
    for c in best_codes:
        digs = [(c // (3**i)) % 3 for i in range(6)]
        result.append(digs)
        
    return result
