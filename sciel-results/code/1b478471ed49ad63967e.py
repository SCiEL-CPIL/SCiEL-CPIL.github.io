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
        
    # Greedy Set Cover Initialization
    # uncovered: bitmask of words not yet covered
    all_words_mask = (1 << 729) - 1
    uncovered = all_words_mask
    current_codes = []
    code_set = set()
    
    # Precompute for each word, which codewords cover it? 
    # For greedy, we need to pick c maximizing popcount(neighbor_mask[c] & uncovered).
    # This is O(729^2) per step, which is ~500k ops per step. 
    # 729 steps max -> 360M ops. Might be slow in Python.
    # Optimization: We can just run it. 729 is small enough.
    
    while uncovered:
        best_c = -1
        max_gain = 0
        # Iterate through all candidates
        # We can skip candidates already in code_set
        for c in range(729):
            if c in code_set:
                continue
            gain = bin(neighbor_mask[c] & uncovered).count('1')
            if gain > max_gain:
                max_gain = gain
                best_c = c
            # Early break if we cover all remaining?
            if max_gain == bin(uncovered).count('1'):
                break
        if best_c == -1:
            break # Should not happen
        current_codes.append(best_c)
        code_set.add(best_c)
        uncovered &= ~neighbor_mask[best_c]
        
    # Initialize coverage data structures
    cov = [0] * 729
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
        
    def try_2out_1in():
        """Try to remove 2 codewords and add 1."""
        # Identify codewords with exclusive coverage
        # U_c = {w in N[c] | cov[w] == 1}
        exclusives = {}
        for c in current_codes:
            ex = [w for w in neighbors[c] if cov[w] == 1]
            if ex:
                exclusives[c] = ex
        
        if len(exclusives) < 2:
            return False
            
        # Get list of keys
        keys = list(exclusives.keys())
        
        # To avoid O(N^2) full check, we can be smart.
        # But N is small (<=81). 81*80/2 = 3240 pairs.
        # For each pair, intersection of coverers.
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                c1 = keys[i]
                c2 = keys[j]
                U1 = exclusives[c1]
                U2 = exclusives[c2]
                
                # The union of exclusive words must be covered by the new codeword
                # Also, we must ensure that no OTHER word becomes uncovered.
                # Words in N[c1] \ U1 are covered by at least one other codeword in C \ {c1, c2}.
                # Words in N[c2] \ U2 are covered by at least one other codeword in C \ {c1, c2}.
                # So we only need to cover U1 U U2.
                
                # Find candidates for c3 that cover all words in U1 U U2
                # Intersection of coverers of all w in U1 U U2
                # Exclude c1 and c2 from candidates (they are being removed)
                # Exclude other existing codewords? No, c3 must be NEW (not in C).
                
                # Optimization: If U1 U U2 is large, intersection will be small.
                # Start with coverers of the word in U1 U U2 with smallest coverer set?
                # Or just iterate.
                
                all_exclusive = set(U1) | set(U2)
                if not all_exclusive:
                    continue
                
                # Pick one word to start intersection
                # Sort by number of coverers to minimize intersection size early
                sorted_words = sorted(all_exclusive, key=lambda w: len(coverers[w]))
                candidate_c3 = set(coverers[sorted_words[0]])
                for w in sorted_words[1:]:
                    candidate_c3 &= coverers[w]
                    if not candidate_c3:
                        break
                
                if not candidate_c3:
                    continue
                
                # c3 must not be in current_codes
                candidate_c3 -= code_set
                if not candidate_c3:
                    continue
                
                # We found a valid c3. 
                # Perform the move.
                # We can pick the first one, or the one with best "gain" (coverage of low-cov words).
                # For simplicity, pick the first.
                new_c = next(iter(candidate_c3))
                
                remove_code(c1)
                remove_code(c2)
                add_code(new_c)
                return True
                
        return False

    # Main Loop
    no_improve_count = 0
    max_no_improve = 50
    temperature = 100.0
    cooling_rate = 0.95
    
    while time.time() < deadline:
        # 1. Try removing
        if try_removals():
            if len(current_codes) < best_size:
                best_size = len(current_codes)
                best_codes = current_codes[:]
                no_improve_count = 0
                temperature = 100.0
            else:
                no_improve_count += 1
            continue
            
        # 2. Try swapping (1-out-1-in)
        if try_swaps():
            no_improve_count = 0
            continue
            
        # 3. Try 2-out-1-in
        if try_2out_1in():
            no_improve_count = 0
            # After 2-out-1-in, we might be able to remove more
            if try_removals():
                if len(current_codes) < best_size:
                    best_size = len(current_codes)
                    best_codes = current_codes[:]
                    no_improve_count = 0
                    temperature = 100.0
            continue
            
        # 4. Stuck. Perturb
        if no_improve_count > max_no_improve:
            # Revert to best
            current_codes[:] = best_codes[:]
            code_set = set(current_codes)
            cov = [0] * 729
            coverers = [set() for _ in range(729)]
            for c in current_codes:
                for w in neighbors[c]:
                    cov[w] += 1
                    coverers[w].add(c)
            no_improve_count = 0
            
            # Add random codewords
            num_perturb = random.randint(2, 5)
            for _ in range(num_perturb):
                for attempt in range(30):
                    r = random.randint(0, 728)
                    if r not in code_set:
                        add_code(r)
                        break
        
        else:
            # Mild perturbation
            num_perturb = max(1, int(temperature / 50))
            for _ in range(num_perturb):
                for attempt in range(30):
                    r = random.randint(0, 728)
                    if r not in code_set:
                        add_code(r)
                        break
        
        no_improve_count += 1
        temperature *= cooling_rate
        
        # Recover
        try_removals()
        if len(current_codes) < best_size:
            best_size = len(current_codes)
            best_codes = current_codes[:]
            no_improve_count = 0
            temperature = 100.0
        
    # Final polish
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        if try_removals() or try_swaps() or try_2out_1in():
            if len(current_codes) < best_size:
                best_size = len(current_codes)
                best_codes = current_codes[:]
    
    # Convert best_codes back to list of lists
    result = []
    for c in best_codes:
        digs = [(c // (3**i)) % 3 for i in range(6)]
        result.append(digs)
        
    return result
