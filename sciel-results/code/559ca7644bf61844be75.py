import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 12
    m = 40
    
    # All possible subsets as bitmasks
    # We don't need to iterate all 4096 in the main loop if we are smart, 
    # but for m=40 and n=12, 4096 is small enough for fast checks.
    
    # Helper to count comparable pairs
    def count_comparables(family):
        count = 0
        k = len(family)
        for i in range(k):
            fi = family[i]
            for j in range(i + 1, k):
                fj = family[j]
                # Strict subset: (fi & fj) == fi and fi != fj
                # Or (fi & fj) == fj and fi != fj
                # Since they are distinct, we just check if one is subset of other
                if (fi & fj) == fi or (fi & fj) == fj:
                    count += 1
        return count

    # 1. Generate a good initial solution using multiple disjoint chains
    # A chain of length L contributes L*(L-1)/2 pairs.
    # We want to partition the 40 slots into chains that are as long as possible
    # while remaining disjoint (no set appears in two chains).
    # Actually, sets CAN appear in multiple chains in the final family? No, family is a set of distinct subsets.
    # So we need 40 distinct subsets.
    # Strategy: Create chains that are "orthogonal" or use different dimensions.
    # Better: Just take the first 40 sets from a specific ordering that maximizes comparability.
    # Or: Start with a few long chains.
    # Max chain length is 13.
    # Let's try to construct 3 chains of length ~13.
    # Chain 1: 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095
    # Chain 2: Must not intersect Chain 1.
    # This is hard to do manually. 
    # Alternative Initial Heuristic: 
    # Sort all 4096 subsets by "potential" or just take a random set and refine?
    # Let's use a simple heuristic: 
    # Start with the empty set and full set.
    # Then greedily add the set that maximizes immediate gain.
    # This is what the seed does. 
    # To improve, we can use a "Multi-Start" SA or just a very strong SA.
    
    # Let's try a specific deterministic initialization:
    # Take all sets of size 0, 1, 2, 3, 4, 5, 6.
    # Sizes:
    # 0: 1
    # 1: 12
    # 2: 66
    # 3: 220
    # 4: 495
    # ...
    # Total 1 + 12 + 66 + 220 = 299 sets.
    # If we take all sets of size <= 2, we have 1 + 12 + 66 = 79 sets.
    # We only need 40.
    # If we take all sets of size 0 and 1 (13 sets) and 27 sets of size 2.
    # Pairs:
    # (0,1): 12 pairs
    # (1,1): 0
    # (0,2): 66 * 1? No, 0 is subset of all 2s. 1 * 66 = 66 pairs.
    # (1,2): Each 1-set {i} is subset of 2-sets containing i.
    # If we pick 27 2-sets, how many pairs with 1-sets?
    # Average 2-sets per 1-set?
    # This seems low. Chains are better.
    
    # Let's go with the Greedy Initialization from the seed, but run it multiple times with slight variations?
    # No, the seed's greedy is deterministic and found 2.54.
    # Let's use Simulated Annealing starting from a random valid family or the greedy result.
    
    # Get greedy result as starting point
    def greedy_init():
        family = []
        family_set = set()
        
        # Start with empty set
        family.append(0)
        family_set.add(0)
        
        while len(family) < m:
            best_candidate = -1
            best_gain = -1
            
            # Optimization: Only consider sets that are "close" to existing ones?
            # No, just scan all. 4096 * 40 * 40 is fast.
            for s in range(1 << n):
                if s in family_set:
                    continue
                
                gain = 0
                for f in family:
                    if (f & s) == f or (f & s) == s:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = s
            if best_candidate != -1:
                family.append(best_candidate)
                family_set.add(best_candidate)
        
        return family

    family = greedy_init()
    current_score = count_comparables(family)
    
    # Simulated Annealing
    # We need to be able to swap elements in the family.
    # State: list of 40 bitmasks.
    # Move: Pick a random index i, pick a random candidate s not in family, swap.
    # Delta: Calculate change in score.
    
    # To calculate delta efficiently:
    # Let A be the family without element at index i.
    # Old contribution of family[i] to score: C_old
    # New contribution of s to score: C_new
    # Delta = C_new - C_old
    
    # C_old = number of sets in A comparable to family[i]
    # C_new = number of sets in A comparable to s
    
    # Precompute comparability matrix? No, 40x40 is small.
    # But we do this millions of times.
    # Let's optimize the delta calculation.
    
    def get_comparable_count(family, s, exclude_index=-1):
        """Count how many sets in family are comparable to s, excluding index exclude_index."""
        count = 0
        for i, f in enumerate(family):
            if i == exclude_index:
                continue
            if (f & s) == f or (f & s) == s:
                count += 1
        return count

    # SA Parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.9998  # Slower cooling for more exploration
    max_iter_per_restart = 150000
    num_restarts = 5
    
    best_family = family[:]
    best_score = current_score
    
    # Precompute list of all masks for quick access
    all_masks = list(range(1 << n))
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Reset temperature for each restart
        T = 10.0
        current_score = count_comparables(family)
        
        for it in range(max_iter_per_restart):
            if time.time() > deadline:
                break
            
            # Pick random index to replace
            i = random.randint(0, m - 1)
            old_s = family[i]
            
            # Smart candidate generation: 70% biased toward comparable sets, 30% random
            if random.random() < 0.7:
                # Pick a random existing set and generate a candidate comparable to it
                ref_idx = random.randint(0, m - 1)
                if ref_idx == i:
                    ref_idx = (ref_idx + 1) % m
                ref_s = family[ref_idx]
                
                # Generate a set comparable to ref_s
                # Option 1: superset of ref_s (add random bits)
                # Option 2: subset of ref_s (remove random bits)
                if random.random() < 0.5:
                    # Superset: start with ref_s, add 1-3 random bits
                    new_s = ref_s
                    num_add = random.randint(1, 3)
                    for _ in range(num_add):
                        bit = random.randint(0, n - 1)
                        new_s |= (1 << bit)
                else:
                    # Subset: start with ref_s, remove 1-3 random bits
                    new_s = ref_s
                    num_remove = random.randint(1, 3)
                    for _ in range(num_remove):
                        bit = random.randint(0, n - 1)
                        new_s &= ~(1 << bit)
            else:
                # Pure random
                new_s = random.randint(0, (1 << n) - 1)
            
            if new_s == old_s:
                continue
            
            # Check if new_s is already in family (excluding i)
            is_dup = False
            for j, f in enumerate(family):
                if j != i and f == new_s:
                    is_dup = True
                    break
            if is_dup:
                continue
                
            # Calculate Delta
            count_new = 0
            count_old = 0
            
            for j, f in enumerate(family):
                if j == i:
                    continue
                if (f & new_s) == f or (f & new_s) == new_s:
                    count_new += 1
                if (f & old_s) == f or (f & old_s) == old_s:
                    count_old += 1
            
            delta = count_new - count_old
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                # Accept
                family[i] = new_s
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
            
            T *= alpha
            if T < T_min:
                T = T_min
        
        # After each restart, do a local search polish
        if time.time() < deadline - 5.0:
            # Local search: try to improve by single-element swaps
            improved = True
            while improved and time.time() < deadline - 3.0:
                improved = False
                for i in range(m):
                    old_s = family[i]
                    best_local_gain = 0
                    best_local_s = -1
                    
                    # Try a few candidates
                    for trial in range(20):
                        if random.random() < 0.6:
                            ref_idx = random.randint(0, m - 1)
                            if ref_idx == i:
                                ref_idx = (ref_idx + 1) % m
                            ref_s = family[ref_idx]
                            if random.random() < 0.5:
                                new_s = ref_s
                                for _ in range(random.randint(1, 3)):
                                    bit = random.randint(0, n - 1)
                                    new_s |= (1 << bit)
                            else:
                                new_s = ref_s
                                for _ in range(random.randint(1, 3)):
                                    bit = random.randint(0, n - 1)
                                    new_s &= ~(1 << bit)
                        else:
                            new_s = random.randint(0, (1 << n) - 1)
                        
                        if new_s == old_s:
                            continue
                        
                        is_dup = False
                        for j, f in enumerate(family):
                            if j != i and f == new_s:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                        
                        count_new = 0
                        for j, f in enumerate(family):
                            if j == i:
                                continue
                            if (f & new_s) == f or (f & new_s) == new_s:
                                count_new += 1
                        
                        count_old = 0
                        for j, f in enumerate(family):
                            if j == i:
                                continue
                            if (f & old_s) == f or (f & old_s) == old_s:
                                count_old += 1
                        
                        local_gain = count_new - count_old
                        if local_gain > best_local_gain:
                            best_local_gain = local_gain
                            best_local_s = new_s
                    
                    if best_local_s != -1 and best_local_gain > 0:
                        family[i] = best_local_s
                        current_score += best_local_gain
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_family = family[:]
    
    # Final verification and conversion
    # Ensure we return the best found
    family = best_family
    
    # Convert bitmasks to lists
    result = []
    for s in family:
        subset = []
        for i in range(n):
            if (s >> i) & 1:
                subset.append(i)
        result.append(subset)
        
    return result
