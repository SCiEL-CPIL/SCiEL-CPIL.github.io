import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin before 90s limit
    
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

    # Multi-chain initialization: build several disjoint chains then fill with greedy
    def multi_chain_init():
        family = []
        family_set = set()
        
        # Build chains by extending from small sets
        # Chain: start from empty, keep adding one element at a time
        # We want chains that are "disjoint" in terms of which sets they use
        
        # Strategy: create chains along different "directions"
        # Chain along bit 0: {}, {0}, {0,1}, {0,1,2}, ...
        # Chain along bit 1: {}, {1}, {1,2}, {1,2,3}, ...
        # But they share the empty set.
        
        # Better: use different starting points and extensions
        # Let's create chains that use different "spines"
        
        # Chain 1: {}, {0}, {0,1}, {0,1,2}, {0,1,2,3}, {0,1,2,3,4}, {0,1,2,3,4,5}, {0,1,2,3,4,5,6}, {0,1,2,3,4,5,6,7}, {0,1,2,3,4,5,6,7,8}, {0,1,2,3,4,5,6,7,8,9}, {0,1,2,3,4,5,6,7,8,9,10}, {0,1,2,3,4,5,6,7,8,9,10,11}
        # That's 13 sets.
        
        # For remaining 27 sets, use greedy from the existing family
        
        # Actually, let's just do greedy but with a better heuristic:
        # Prefer sets that are comparable to many existing sets AND have high "chain potential"
        
        family = greedy_init()
        return family

    family = greedy_init()
    current_score = count_comparables(family)
    
    # Try a few different greedy variants and keep the best
    def greedy_init_v2():
        """Greedy that prefers sets with higher comparability to existing family."""
        family = []
        family_set = set()
        
        # Start with a few anchor points
        anchors = [0, (1<<n)-1]
        for a in anchors:
            if a not in family_set:
                family.append(a)
                family_set.add(a)
        
        while len(family) < m:
            best_candidate = -1
            best_gain = -1
            best_secondary = -1
            
            for s in range(1 << n):
                if s in family_set:
                    continue
                
                gain = 0
                # Secondary: prefer sets that are "middle" of chains
                # i.e., have both subsets and supersets in the family
                has_lower = 0
                has_upper = 0
                for f in family:
                    if (f & s) == f:
                        if f != s:
                            has_lower += 1
                    elif (f & s) == s:
                        if f != s:
                            has_upper += 1
                    gain += 1
                
                # Prefer sets that extend existing chains
                secondary = min(has_lower, has_upper)
                
                if gain > best_gain or (gain == best_gain and secondary > best_secondary):
                    best_gain = gain
                    best_secondary = secondary
                    best_candidate = s
            
            if best_candidate != -1:
                family.append(best_candidate)
                family_set.add(best_candidate)
        
        return family

    # Try both and keep the best
    family_v2 = greedy_init_v2()
    score_v2 = count_comparables(family_v2)
    if score_v2 > current_score:
        family = family_v2
        current_score = score_v2
    
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
    alpha = 0.9995 # Cooling rate
    max_iter = 200000
    
    best_family = family[:]
    best_score = current_score
    
    # Precompute all subsets for faster random selection
    all_subsets = list(range(1 << n))
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick random index to replace
        i = random.randint(0, m - 1)
        old_s = family[i]
        
        # Smart candidate selection: bias toward sets comparable to existing family
        # with some probability, pick a random set that is comparable to at least one
        # existing set; otherwise pick fully random.
        if random.random() < 0.7:
            # Pick a random existing set (not at index i) and generate a comparable set
            ref_idx = random.randint(0, m - 1)
            while ref_idx == i:
                ref_idx = random.randint(0, m - 1)
            ref = family[ref_idx]
            
            # Generate a set comparable to ref:
            # Either a subset of ref or a superset of ref
            if random.random() < 0.5:
                # Subset of ref
                # Randomly clear some bits
                new_s = ref
                # Clear 1-3 random bits that are set
                set_bits = [b for b in range(n) if (ref >> b) & 1]
                if set_bits:
                    num_to_clear = min(len(set_bits), random.randint(1, 3))
                    to_clear = random.sample(set_bits, num_to_clear)
                    for b in to_clear:
                        new_s &= ~(1 << b)
            else:
                # Superset of ref
                new_s = ref
                # Randomly set some bits that are not set
                unset_bits = [b for b in range(n) if not (ref >> b) & 1]
                if unset_bits:
                    num_to_set = min(len(unset_bits), random.randint(1, 3))
                    to_set = random.sample(unset_bits, num_to_set)
                    for b in to_set:
                        new_s |= (1 << b)
        else:
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
        
    # Verify: exactly 40 distinct lists of distinct ints in 0..11
    assert len(result) == 40, f"Expected 40 sets, got {len(result)}"
    seen = set()
    for subset in result:
        assert isinstance(subset, list), f"Subset must be list, got {type(subset)}"
        assert len(subset) == len(set(subset)), f"Duplicate elements in {subset}"
        for x in subset:
            assert isinstance(x, int) and not isinstance(x, bool), f"Element {x} not a pure int"
            assert 0 <= x <= 11, f"Element {x} out of range"
        assert tuple(subset) not in seen, f"Duplicate set {subset}"
        seen.add(tuple(subset))
    
    return result
