import time
import random

def construct():
    n = 12
    m = 40
    total = 1 << n  # 4096
    deadline = time.time() + 82.0

    # Precompute comparability: comp[a][b] = True if a and b are comparable
    # Use a flat list for speed: comp[a * total + b]
    # Actually, let's use a list of sets or bitmasks for faster iteration
    # Better: precompute for each mask, the list of comparable masks
    # Even better: for delta calculation, we need count of comparables in family
    
    # Precompute: for each mask, which masks is it comparable to?
    # comp_set[mask] = set of masks comparable to mask
    # But that's 4096 sets, each potentially large. Let's use bitmasks.
    
    # Actually, the key insight: is_comparable(a,b) means (a&b)==a or (a&b)==b
    # i.e., a⊆b or b⊆a
    
    # For speed, precompute for each mask a bitmask of all comparable masks
    # comparable_mask[a] = bitmask where bit b is set if a and b are comparable
    comparable_mask = [0] * total
    for a in range(total):
        cm = 0
        for b in range(total):
            if (a & b) == a or (a & b) == b:
                cm |= (1 << b)
        comparable_mask[a] = cm
    
    # But 4096 bits doesn't fit in a Python int efficiently for counting.
    # Alternative: precompute count of comparables for any subset? No, too expensive.
    
    # Better approach: for a given family, compute comp_count[mask] = number of family members comparable to mask
    # Then delta for swapping a->b is: (comp_count[b] - [a,b comparable]) - comp_count[a]
    
    def calc_comp_counts(family_set):
        """For each mask, count how many members of family it's comparable to."""
        counts = [0] * total
        for a in family_set:
            # For each mask, increment if comparable to a
            # Use the precomputed comparable_mask
            cm = comparable_mask[a]
            # Count bits in cm that are in family? No, we want for each mask m, 
            # whether m is comparable to a.
            # Actually: counts[m] += 1 if m is comparable to a
            # So: for m in range(total): if (comparable_mask[a] >> m) & 1: counts[m] += 1
            # This is O(total) per family member, so O(m * total) = 40 * 4096 = 163840
            # That's fine.
            for m in range(total):
                if (cm >> m) & 1:
                    counts[m] += 1
        return counts
    
    def calc_score(family_list):
        score = 0
        L = len(family_list)
        for i in range(L):
            ai = family_list[i]
            for j in range(i + 1, L):
                bj = family_list[j]
                if (ai & bj) == ai or (ai & bj) == bj:
                    score += 1
        return score
    
    def greedy_construct():
        """Build family greedily: start with a chain, add sets maximizing comparable pairs."""
        family = []
        used = set()
        
        # Start with full chain: empty, {0}, {0,1}, ..., {0..11}
        cur = 0
        family.append(0)
        used.add(0)
        for i in range(n):
            cur |= (1 << i)
            family.append(cur)
            used.add(cur)
        
        all_masks = list(range(total))
        random.shuffle(all_masks)
        
        # Greedily add remaining sets
        while len(family) < m:
            best_gain = -1
            best_mask = None
            for s in all_masks:
                if s in used:
                    continue
                gain = 0
                for f in family:
                    if (s & f) == s or (s & f) == f:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = s
            if best_mask is None:
                break
            family.append(best_mask)
            used.add(best_mask)
        
        return family
    
    def hill_climb(start_family):
        """Exact local search: try all single swaps, keep best improvement."""
        family_set = set(start_family)
        comp_counts = calc_comp_counts(family_set)
        current_score = calc_score(list(family_set))
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            best_delta = 0
            best_a = None
            best_b = None
            
            for a in family_set:
                comp_a = comp_counts[a]
                for b in range(total):
                    if b in family_set:
                        continue
                    comp_b = comp_counts[b]
                    is_comp_ab = 1 if ((a & b) == a or (a & b) == b) else 0
                    delta = (comp_b - is_comp_ab) - comp_a
                    if delta > best_delta:
                        best_delta = delta
                        best_a = a
                        best_b = b
            
            if best_delta > 0:
                # Apply swap
                family_set.remove(best_a)
                family_set.add(best_b)
                current_score += best_delta
                
                # Update comp_counts
                # Decrement for all masks comparable with best_a
                cm_a = comparable_mask[best_a]
                for mask in range(total):
                    if (cm_a >> mask) & 1:
                        comp_counts[mask] -= 1
                # Increment for all masks comparable with best_b
                cm_b = comparable_mask[best_b]
                for mask in range(total):
                    if (cm_b >> mask) & 1:
                        comp_counts[mask] += 1
                
                improved = True
        
        return list(family_set), current_score
    
    def sa_polish(start_family):
        """SA-based polish to escape local optima."""
        family = list(start_family)
        current_score = calc_score(family)
        best_score = current_score
        best_family = family[:]
        
        T = 5.0
        min_T = 0.1
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 10000
        
        steps = 0
        max_steps = 50000
        
        while steps < max_steps and time.time() < deadline - 2:
            steps += 1
            no_improve += 1
            
            i = random.randint(0, m - 1)
            old_mask = family[i]
            
            # Mutation: flip 1-3 bits
            new_mask = old_mask
            num_flips = random.randint(1, 3)
            for _ in range(num_flips):
                bit = random.randint(0, n - 1)
                new_mask ^= (1 << bit)
            
            if new_mask == old_mask:
                continue
            
            # Check duplicate
            dup = False
            for k in range(m):
                if k != i and family[k] == new_mask:
                    dup = True
                    break
            if dup:
                continue
            
            # Calculate delta
            delta = 0
            for j in range(m):
                if j == i:
                    continue
                other = family[j]
                new_comp = 1 if ((new_mask & other) == new_mask or (new_mask & other) == other) else 0
                old_comp = 1 if ((old_mask & other) == old_mask or (old_mask & other) == other) else 0
                delta += new_comp - old_comp
            
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                family[i] = new_mask
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
                    no_improve = 0
            
            T *= cooling
            if no_improve > max_no_improve:
                break
        
        return best_family, best_score
    
    best_score = -1
    best_family = None
    
    # Generate diverse starting points
    starts = []
    
    # 1. Greedy chain
    starts.append(greedy_construct())
    
    # 2. Random starts
    for _ in range(3):
        starts.append(random.sample(range(total), m))
    
    # 3. Small universe starts (subsets of {0..5})
    small_candidates = [i for i in range(total) if (i >> 6) == 0]
    for _ in range(2):
        starts.append(random.sample(small_candidates, m))
    
    # 4. Another greedy variant: start with all singletons + empty + full
    family4 = [0]
    used4 = {0}
    for i in range(n):
        family4.append(1 << i)
        used4.add(1 << i)
    family4.append((1 << n) - 1)
    used4.add((1 << n) - 1)
    all_masks4 = list(range(total))
    random.shuffle(all_masks4)
    while len(family4) < m:
        best_gain = -1
        best_mask = None
        for s in all_masks4:
            if s in used4:
                continue
            gain = sum(1 for f in family4 if (s & f) == s or (s & f) == f)
            if gain > best_gain:
                best_gain = gain
                best_mask = s
        if best_mask is None:
            break
        family4.append(best_mask)
        used4.add(best_mask)
    starts.append(family4)
    
    for start in starts:
        if time.time() > deadline - 5:
            break
        
        # Phase 1: Exact hill climb
        hc_family, hc_score = hill_climb(start)
        if hc_score > best_score:
            best_score = hc_score
            best_family = hc_family[:]
        
        # Phase 2: SA polish from hill climb result
        if time.time() < deadline - 3:
            sa_family, sa_score = sa_polish(hc_family)
            if sa_score > best_score:
                best_score = sa_score
                best_family = sa_family[:]
        
        # Phase 3: Another hill climb from SA result
        if time.time() < deadline - 3:
            hc2_family, hc2_score = hill_climb(sa_family)
            if hc2_score > best_score:
                best_score = hc2_score
                best_family = hc2_family[:]
    
    if best_family is None:
        best_family = list(range(m))
    
    # Convert to lists of vertices
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
