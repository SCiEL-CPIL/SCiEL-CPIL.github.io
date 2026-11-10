import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    n = 12
    m = 40
    all_masks = list(range(1 << n))
    
    # Precompute popcount? Not strictly needed for comparability, but might help heuristics.
    
    def is_comparable(a, b):
        # Returns True if a is strict subset of b OR b is strict subset of a
        if a == b:
            return False
        return (a & b) == a or (a & b) == b

    def count_comparable(family):
        """Count unordered pairs in family where one is strict subset of other."""
        count = 0
        k = len(family)
        for i in range(k):
            mi = family[i]
            for j in range(i + 1, k):
                mj = family[j]
                if (mi & mj) == mi or (mi & mj) == mj:
                    count += 1
        return count

    def delta_score(family, idx_remove, add_mask):
        """
        Calculate change in comparable pair count when replacing family[idx_remove] with add_mask.
        """
        remove_mask = family[idx_remove]
        delta = 0
        for i in range(len(family)):
            if i == idx_remove:
                continue
            x = family[i]
            # Check comparability with add_mask
            if (x & add_mask) == x or (x & add_mask) == add_mask:
                delta += 1
            # Check comparability with remove_mask
            if (x & remove_mask) == x or (x & remove_mask) == remove_mask:
                delta -= 1
        return delta

    def greedy_fill(initial_family, target_size):
        """
        Greedily add sets to initial_family to reach target_size,
        maximizing the number of comparable pairs with existing sets.
        """
        family = initial_family[:]
        used = set(family)
        remaining = [m for m in all_masks if m not in used]
        
        while len(family) < target_size:
            best_candidate = None
            best_score = -1
            # To speed up, we can sample a subset of remaining if it's too slow, 
            # but 4096 items is small enough.
            # However, doing this for ~30 additions * 4096 candidates * 40 comparisons = ~5M ops.
            # This is fine.
            for cand in remaining:
                score = 0
                for fs in family:
                    if (cand & fs) == cand or (cand & fs) == fs:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            
            if best_candidate is None:
                break
            family.append(best_candidate)
            used.add(best_candidate)
            remaining.remove(best_candidate)
            
        return family

    def local_search(family, time_limit):
        """
        Run 1-swap hill climbing to improve the family.
        """
        family = family[:]
        current_score = count_comparable(family)
        best_family = family[:]
        best_score = current_score
        
        end_time = time.time() + time_limit
        improved = True
        passes = 0
        
        while improved and time.time() < end_time:
            improved = False
            passes += 1
            for i in range(len(family)):
                if time.time() > end_time:
                    break
                
                remove_mask = family[i]
                best_gain = 0
                best_cand = None
                
                # Try all candidates not in family
                # Optimization: We can precompute a list of unused masks, 
                # but maintaining it is overhead. 
                # Since m=40 and N=4096, iterating all 4096 and checking `in set` is fast enough.
                # Set lookup is O(1) average.
                used_set = set(family)
                
                for cand in all_masks:
                    if cand in used_set:
                        continue
                    
                    gain = delta_score(family, i, cand)
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = cand
                
                if best_cand is not None and best_gain > 0:
                    family[i] = best_cand
                    current_score += best_gain
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_family = family[:]
                    
        return best_family, best_score

    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    # --- Main Logic ---
    
    best_global_family = None
    best_global_score = -1
    
    # Strategy: Run multiple restarts with different initializations
    # Budget: ~80 seconds for search
    
    restarts = 10
    time_per_restart = (deadline - time.time() - 1.0) / restarts
    
    for r in range(restarts):
        if time.time() > deadline - 2.0:
            break
            
        # Generate Initial Family
        # Mix of strategies:
        if r == 0:
            # Pure Chain + Greedy
            chain = [0]
            mask = 0
            for k in range(n):
                mask |= (1 << k)
                chain.append(mask)
            init_family = greedy_fill(chain, m)
        elif r == 1:
            # Random Chain (shuffled vertices)
            perm = list(range(n))
            random.shuffle(perm)
            chain = [0]
            mask = 0
            for k in perm:
                mask |= (1 << k)
                chain.append(mask)
            init_family = greedy_fill(chain, m)
        elif r == 2:
            # Star + Greedy (Empty + all singletons)
            star = [0]
            for i in range(n):
                star.append(1 << i)
            init_family = greedy_fill(star, m)
        elif r == 3:
            # Random
            init_family = random.sample(all_masks, m)
        elif r == 4:
            # Greedy from scratch
            init_family = greedy_fill([0], m)
        elif r == 5:
            # Greedy from a random set of 5
            init_base = random.sample(all_masks, 5)
            init_family = greedy_fill(init_base, m)
        elif r == 6:
            # Greedy from a random chain of 5
            init_base = random.sample(all_masks, 5)
            # Sort by size to make a pseudo-chain
            init_base.sort(key=lambda x: bin(x).count('1'))
            init_family = greedy_fill(init_base, m)
        elif r == 7:
            # Random + Greedy polish
            init_base = random.sample(all_masks, 10)
            init_family = greedy_fill(init_base, m)
        elif r == 8:
            # Greedy from empty set only
            init_family = greedy_fill([0], m)
        else:
            # Random
            init_family = random.sample(all_masks, m)
        
        # Ensure valid init
        if len(init_family) != m:
            # Fallback
            init_family = random.sample(all_masks, m)
        
        # Run Local Search
        ls_time = min(time_per_restart, deadline - time.time() - 1.0)
        if ls_time < 0.1:
            continue
            
        ls_family, ls_score = local_search(init_family, ls_time)
        
        if ls_score > best_global_score:
            best_global_score = ls_score
            best_global_family = ls_family[:]
            
    # Final Local Search (1-swap hill climbing) on the best found family
    # We might have time left, so run a final polish on the best global family
    if best_global_family and time.time() < deadline - 1.0:
        family = best_global_family[:]
        current_score = best_global_score
        
        # Run local search with remaining time
        remaining_time = deadline - time.time() - 0.5
        if remaining_time > 0.1:
            final_family, final_score = local_search(family, remaining_time)
            if final_score > best_global_score:
                best_global_score = final_score
                best_global_family = final_family

    # Convert best family to list of lists
    result = []
    if best_global_family:
        for mask in best_global_family:
            result.append(mask_to_list(mask))
    else:
        # Emergency fallback
        result = [mask_to_list(i) for i in range(m)]
        
    # Validation
    if len(result) != m:
        # Emergency fallback
        result = [mask_to_list(i) for i in range(m)]
        
    # Ensure types are correct
    for i in range(len(result)):
        if not isinstance(result[i], list):
            result[i] = list(result[i])
        for j in range(len(result[i])):
            if not isinstance(result[i][j], int) or isinstance(result[i][j], bool):
                result[i][j] = int(result[i][j])

    return result
