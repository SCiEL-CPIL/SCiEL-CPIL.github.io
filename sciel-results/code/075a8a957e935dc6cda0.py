"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0
    
    N = 4096
    
    def check_add(x, selected, triple_sums):
        """Check if x can be added to selected without violating triple-sum uniqueness.
        Returns (ok, new_sums) where new_sums is the list of new triple sums if ok."""
        n = len(selected)
        seen_new = set()
        
        # Triples with one x: x + a_i + a_j for i <= j
        for i in range(n):
            for j in range(i, n):
                s = x + selected[i] + selected[j]
                if s in seen_new or s in triple_sums:
                    return False, None
                seen_new.add(s)
        
        # Triples with two x's: 2*x + a_i
        for i in range(n):
            s = 2 * x + selected[i]
            if s in seen_new or s in triple_sums:
                return False, None
            seen_new.add(s)
        
        # Triple with three x's
        s = 3 * x
        if s in seen_new or s in triple_sums:
            return False, None
        seen_new.add(s)
        
        return True, list(seen_new)
    
    def build_greedy(start_from=1, selected_init=None, triple_sums_init=None):
        """Greedy: add numbers in order, maintaining set of triple sums."""
        if selected_init is not None:
            selected = list(selected_init)
            triple_sums = set(triple_sums_init) if triple_sums_init else set()
        else:
            selected = []
            triple_sums = set()
        
        for x in range(start_from, N + 1):
            if time.time() > DEADLINE - 10:
                break
            if x in selected:
                continue
            ok, new_sums = check_add(x, selected, triple_sums)
            if ok:
                selected.append(x)
                triple_sums.update(new_sums)
        
        return selected, triple_sums
    
    # Phase 1: Standard greedy from 1
    result, triple_sums = build_greedy(1)
    current_size = len(result)
    
    # Phase 2: Try to fill remaining capacity by trying different starting points
    # and local swaps. The greedy from 1 may miss numbers that would fit if we
    # started differently or skipped some early numbers.
    
    # Phase 2a: Try greedy from various offsets to find alternative large sets
    if time.time() < DEADLINE - 20:
        for offset in [2, 3, 5, 8, 13, 21]:
            if time.time() > DEADLINE - 20:
                break
            alt, _ = build_greedy(offset)
            if len(alt) > current_size:
                result = alt
                current_size = len(result)
                # Rebuild triple_sums for the new result
                triple_sums = set()
                n = len(result)
                for i in range(n):
                    for j in range(i, n):
                        for k in range(j, n):
                            triple_sums.add(result[i] + result[j] + result[k])
    
    # Phase 2b: Local improvement - try swapping out one element for a larger one
    if time.time() < DEADLINE - 15:
        result_set = set(result)
        for _ in range(3):  # Limited improvement rounds
            if time.time() > DEADLINE - 15:
                break
            improved = False
            for i in range(len(result)):
                if time.time() > DEADLINE - 15:
                    break
                # Try removing result[i]
                temp = [x for x in result if x != result[i]]
                temp_set = set(temp)
                
                # Rebuild triple sums for temp
                triple_sums_temp = set()
                m = len(temp)
                for ii in range(m):
                    for jj in range(ii, m):
                        for kk in range(jj, m):
                            triple_sums_temp.add(temp[ii] + temp[jj] + temp[kk])
                
                # Try to add new numbers (scan from high to low to maximize spread)
                improved_list = list(temp)
                improved_set = set(temp)
                added = 0
                for x in range(N, 0, -1):
                    if time.time() > DEADLINE - 15:
                        break
                    if x in improved_set:
                        continue
                    ok, new_sums = check_add(x, improved_list, triple_sums_temp)
                    if ok:
                        improved_list.append(x)
                        improved_set.add(x)
                        triple_sums_temp.update(new_sums)
                        added += 1
                    if added >= 1:
                        break  # Only need to add one to maintain size
                
                if len(improved_list) > current_size:
                    result = improved_list
                    current_size = len(result)
                    improved = True
                    break  # Restart improvement loop with new result
            
            if not improved:
                break
    
    # Phase 2c: Random restarts with partial greedy
    if time.time() < DEADLINE - 10:
        rng = random.Random(42)
        for _ in range(20):
            if time.time() > DEADLINE - 10:
                break
            # Pick a random subset of current result as seed, then greedy-fill
            if len(result) > 10:
                seed_size = min(len(result), rng.randint(5, len(result) - 2))
                seed = rng.sample(result, seed_size)
                seed.sort()
                seed_set = set(seed)
                
                # Build triple sums for seed
                triple_sums_seed = set()
                m = len(seed)
                for ii in range(m):
                    for jj in range(ii, m):
                        for kk in range(jj, m):
                            triple_sums_seed.add(seed[ii] + seed[jj] + seed[kk])
                
                # Greedy fill from seed
                improved_list = list(seed)
                improved_set = set(seed)
                for x in range(1, N + 1):
                    if time.time() > DEADLINE - 10:
                        break
                    if x in improved_set:
                        continue
                    ok, new_sums = check_add(x, improved_list, triple_sums_seed)
                    if ok:
                        improved_list.append(x)
                        improved_set.add(x)
                        triple_sums_seed.update(new_sums)
                
                if len(improved_list) > current_size:
                    result = improved_list
                    current_size = len(result)
    
    return result
# EVOLVE-BLOCK-END
