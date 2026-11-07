import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    def time_left():
        return deadline - time.time()

    # Precompute comparability: comp[a][b] = 1 if a and b are comparable (a != b and one subset of other)
    # For speed, precompute for each mask, which masks it's comparable to
    # comp_table[a] = bitmask of all masks comparable to a
    comp_table = [0] * total_masks
    for a in range(total_masks):
        # a is comparable to b if (a & ~b) == 0 or (b & ~a) == 0, and a != b
        # (a & ~b) == 0 means a is subset of b
        # (b & ~a) == 0 means b is subset of a
        # Let's compute: subsets of a and supersets of a
        # Subsets of a: all masks s where s & ~a == 0, i.e., s is subset of a
        # Supersets of a: all masks s where a & ~s == 0, i.e., a is subset of s
        # Comparable to a = (subsets of a) union (supersets of a) minus {a}
        
        # Use bit manipulation: for each a, compute comparable masks
        comp = 0
        for b in range(total_masks):
            if a == b:
                continue
            if (a & ~b) == 0 or (b & ~a) == 0:
                comp |= (1 << b)
        comp_table[a] = comp

    def calc_score(family_set):
        """Calculate number of comparable pairs in family_set."""
        score = 0
        fam_list = list(family_set)
        L = len(fam_list)
        for i in range(L):
            for j in range(i + 1, L):
                if comp_table[fam_list[i]] & (1 << fam_list[j]):
                    score += 1
        return score

    def calc_comp_counts(family_set):
        """For each mask, count how many members of family it's comparable to."""
        counts = [0] * total_masks
        for a in family_set:
            ct = comp_table[a]
            for b in family_set:
                if ct & (1 << b):
                    counts[a] += 1
        return counts

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
        
        # Greedily add remaining sets
        while len(family) < m and time_left() > 5:
            best_gain = -1
            best_mask = None
            for s in range(total_masks):
                if s in used:
                    continue
                gain = 0
                ct = comp_table[s]
                for f in family:
                    if ct & (1 << f):
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
        """Local search: try all single swaps, keep best improvement."""
        family_set = set(start_family)
        fam_list = list(family_set)
        
        # Precompute comp counts for current family
        comp_counts = [0] * total_masks
        for a in family_set:
            ct = comp_table[a]
            for b in family_set:
                if ct & (1 << b):
                    comp_counts[a] += 1
        
        current_score = calc_score(family_set)
        
        improved = True
        while improved and time_left() > 3:
            improved = False
            
            best_delta = 0
            best_a = None
            best_b = None
            
            for a in fam_list:
                comp_a = comp_counts[a]
                ct_a = comp_table[a]
                for b in range(total_masks):
                    if b in family_set:
                        continue
                    comp_b = comp_counts[b]
                    is_comp_ab = 1 if (ct_a & (1 << b)) else 0
                    delta = (comp_b - is_comp_ab) - comp_a
                    if delta > best_delta:
                        best_delta = delta
                        best_a = a
                        best_b = b
            
            if best_delta > 0:
                # Remove best_a, add best_b
                family_set.remove(best_a)
                family_set.add(best_b)
                fam_list = list(family_set)
                current_score += best_delta
                
                # Update comp_counts
                ct_a = comp_table[best_a]
                ct_b = comp_table[best_b]
                for mask in fam_list:
                    if ct_a & (1 << mask):
                        comp_counts[mask] -= 1
                    if ct_b & (1 << mask):
                        comp_counts[mask] += 1
                
                improved = True
        
        return list(family_set), current_score

    # Generate diverse structured starts
    starts = []
    
    # Start 1: Pure greedy
    starts.append(greedy_construct())
    
    # Start 2: Multiple short chains
    family2 = []
    used2 = set()
    # Chain 1: 0, 1, 3, 7, 15, 31, 63 (7 sets)
    chain1 = [0, 1, 3, 7, 15, 31, 63]
    for c in chain1:
        family2.append(c)
        used2.add(c)
    # Chain 2: 2, 6, 14, 30, 62 (5 sets) - disjoint from chain1 where possible
    chain2 = [2, 6, 14, 30, 62]
    for c in chain2:
        if c not in used2:
            family2.append(c)
            used2.add(c)
    # Chain 3: 4, 12, 28, 60 (4 sets)
    chain3 = [4, 12, 28, 60]
    for c in chain3:
        if c not in used2:
            family2.append(c)
            used2.add(c)
    # Fill rest greedily
    while len(family2) < m and time_left() > 5:
        best_gain = -1
        best_mask = None
        for s in range(total_masks):
            if s in used2:
                continue
            gain = 0
            ct = comp_table[s]
            for f in family2:
                if ct & (1 << f):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = s
        if best_mask is None:
            break
        family2.append(best_mask)
        used2.add(best_mask)
    starts.append(family2)
    
    # Start 3: Random perturbation of greedy
    base = greedy_construct()
    random_start = base[:]
    for _ in range(3):
        i = random.randint(0, m - 1)
        candidates = [x for x in range(total_masks) if x not in set(random_start)]
        if candidates:
            random_start[i] = random.choice(candidates)
    starts.append(random_start)
    
    best_family = None
    best_score = -1

    for idx, start in enumerate(starts):
        if time_left() < 10:
            break
        fam, score = hill_climb(start)
        if score > best_score:
            best_score = score
            best_family = fam

    if best_family is None:
        best_family = greedy_construct()

    # Convert to lists of vertices
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
