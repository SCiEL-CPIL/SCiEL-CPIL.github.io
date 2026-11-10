import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = 12
    m = 40
    all_masks = list(range(1 << n))
    
    # Precompute popcount for potential sorting or heuristics, though not strictly needed for SA
    # popcount = [bin(x).count('1') for x in all_masks]

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

    def delta_score(family, remove_mask, add_mask):
        """
        Calculate change in comparable pair count when replacing remove_mask with add_mask.
        family contains remove_mask. add_mask is NOT in family.
        R = family - {remove_mask}
        Delta = sum_{x in R} (is_comparable(x, add_mask) - is_comparable(x, remove_mask))
        """
        delta = 0
        for x in family:
            if x == remove_mask:
                continue
            # Check comparability with add_mask and remove_mask
            comp_add = 1 if ((x & add_mask) == x or (x & add_mask) == add_mask) else 0
            comp_rem = 1 if ((x & remove_mask) == x or (x & remove_mask) == remove_mask) else 0
            delta += comp_add - comp_rem
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
            # but 4096 items is small enough for a few iterations.
            # However, doing this for 27 additions * 4096 candidates * 40 comparisons = ~4.5M ops.
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

    def simulated_annealing(initial_family, time_limit):
        """
        Run Simulated Annealing to improve the family.
        """
        family = initial_family[:]
        current_score = count_comparable(family)
        best_family = family[:]
        best_score = current_score
        
        # Temperature parameters
        T = 10.0
        cooling_rate = 0.995
        min_T = 0.01
        
        end_time = time.time() + time_limit
        no_improve_counter = 0
        max_no_improve = 2000 # Restart condition within SA
        
        while time.time() < end_time and T > min_T:
            # Pick a random index to replace
            i = random.randint(0, len(family) - 1)
            remove_mask = family[i]
            
            # Pick a random candidate not in family
            # Optimization: Pick from a random sample of all_masks to avoid O(N) search for unused
            # Or just pick random mask and check if in family (set lookup)
            cand = random.choice(all_masks)
            if cand in family:
                # Retry or pick another
                # Since m=40 and N=4096, collision is rare (~1%)
                continue
            
            add_mask = cand
            
            delta = delta_score(family, remove_mask, add_mask)
            
            # Acceptance criterion
            if delta > 0:
                accept = True
            else:
                # Accept with probability exp(delta / T)
                # delta is negative here
                if random.random() < math.exp(delta / T):
                    accept = True
                else:
                    accept = False
            
            if accept:
                family[i] = add_mask
                current_score += delta
                T *= cooling_rate
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
                    
                # Optional: If stuck for too long, restart SA with a perturbation
                # But we have outer restarts. Let's just let T cool.
            else:
                T *= cooling_rate
                no_improve_counter += 1
                
            # Check time frequently
            if time.time() > end_time:
                break

        return best_family, best_score

    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    # --- Main Logic ---
    
    best_global_family = None
    best_global_score = -1
    
    # Strategy: Run multiple restarts with different initializations
    # Budget: ~50 seconds for search
    
    restarts = 5
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
        else:
            # Greedy from scratch
            init_family = greedy_fill([0], m)
        
        # Ensure valid init
        if len(init_family) != m:
            # Fallback
            init_family = random.sample(all_masks, m)
        
        # Run SA
        sa_time = min(time_per_restart, deadline - time.time() - 1.0)
        if sa_time < 0.1:
            continue
            
        sa_family, sa_score = simulated_annealing(init_family, sa_time)
        
        if sa_score > best_global_score:
            best_global_score = sa_score
            best_global_family = sa_family[:]
            
    # Final Local Search (1-swap hill climbing) on the best found family
    if best_global_family and time.time() < deadline - 1.0:
        family = best_global_family[:]
        current_score = best_global_score
        used = set(family)
        
        # Quick Hill Climb
        improved = True
        end_time = deadline - 0.5
        while improved and time.time() < end_time:
            improved = False
            for i in range(len(family)):
                if time.time() > end_time:
                    break
                remove_mask = family[i]
                # Find best replacement
                best_gain = 0
                best_cand = None
                # Sample candidates to speed up? Or check all?
                # Checking all 4096 is 4096 * 40 = 160k ops.
                # For 40 items, that's 6.4M ops. Might be too slow if repeated many times.
                # Let's sample 100 random candidates per item.
                candidates = random.sample(all_masks, 100)
                for cand in candidates:
                    if cand in used:
                        continue
                    gain = delta_score(family, remove_mask, cand)
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = cand
                
                if best_cand is not None and best_gain > 0:
                    family[i] = best_cand
                    used.remove(remove_mask)
                    used.add(best_cand)
                    current_score += best_gain
                    improved = True
        
        if current_score > best_global_score:
            best_global_score = current_score
            best_global_family = family

    # Convert best family to list of lists
    result = []
    for mask in best_global_family:
        result.append(mask_to_list(mask))
        
    # Validation
    if len(result) != m:
        # Emergency fallback
        result = [mask_to_list(i) for i in range(m)]
        
    # Ensure types are correct
    for i in range(len(result)):
        if not isinstance(result[i], list):
            result[i] = list(result[i])
        for j in range(len(result[i])):
            if not isinstance(result[i][j], int):
                result[i][j] = int(result[i][j])
            if isinstance(result[i][j], bool):
                result[i][j] = int(result[i][j])

    return result
