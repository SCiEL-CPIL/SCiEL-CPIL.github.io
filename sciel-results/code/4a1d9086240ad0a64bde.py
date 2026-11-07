import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 80  # 10s safety margin

    def is_comparable(a, b):
        if a == b:
            return False
        return (a & ~b) == 0 or (b & ~a) == 0

    def calc_score(family_list):
        score = 0
        L = len(family_list)
        for i in range(L):
            ai = family_list[i]
            for j in range(i + 1, L):
                bj = family_list[j]
                if is_comparable(ai, bj):
                    score += 1
        return score

    def calc_comp_counts(family_set):
        """For each mask, count how many members of family it's comparable to."""
        counts = [0] * total_masks
        for a in family_set:
            for mask in range(total_masks):
                if is_comparable(a, mask):
                    counts[mask] += 1
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
        
        all_masks = list(range(total_masks))
        
        # Greedily add remaining sets
        while len(family) < m:
            best_gain = -1
            best_mask = None
            for s in all_masks:
                if s in used:
                    continue
                gain = 0
                for f in family:
                    if is_comparable(s, f):
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
        comp_counts = calc_comp_counts(family_set)
        current_score = calc_score(list(family_set))
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            best_delta = 0
            best_a = None
            best_b = None
            
            for a in family_set:
                comp_a = comp_counts[a]
                for b in range(total_masks):
                    if b in family_set:
                        continue
                    comp_b = comp_counts[b]
                    is_comp_ab = 1 if is_comparable(a, b) else 0
                    delta = (comp_b - is_comp_ab) - comp_a
                    if delta > best_delta:
                        best_delta = delta
                        best_a = a
                        best_b = b
            
            if best_delta > 0:
                family_set.remove(best_a)
                family_set.add(best_b)
                current_score += best_delta
                
                for mask in range(total_masks):
                    if is_comparable(best_a, mask):
                        comp_counts[mask] -= 1
                for mask in range(total_masks):
                    if is_comparable(best_b, mask):
                        comp_counts[mask] += 1
                
                improved = True
        
        return list(family_set), current_score

    # Generate diverse structured starts
    starts = []
    
    # Start 1: Pure greedy
    starts.append(greedy_construct())
    
    # Start 2: Greedy with different chain order
    family2 = []
    used2 = set()
    cur = 0
    family2.append(0)
    used2.add(0)
    # Reverse order chain
    for i in range(n - 1, -1, -1):
        cur |= (1 << i)
        family2.append(cur)
        used2.add(cur)
    all_masks = list(range(total_masks))
    while len(family2) < m:
        best_gain = -1
        best_mask = None
        for s in all_masks:
            if s in used2:
                continue
            gain = 0
            for f in family2:
                if is_comparable(s, f):
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
    # Randomly swap a few elements
    for _ in range(3):
        i = random.randint(0, m - 1)
        candidates = [x for x in range(total_masks) if x not in set(random_start)]
        if candidates:
            random_start[i] = random.choice(candidates)
    starts.append(random_start)
    
    # Start 4: Another random perturbation
    base = greedy_construct()
    random_start2 = base[:]
    for _ in range(5):
        i = random.randint(0, m - 1)
        candidates = [x for x in range(total_masks) if x not in set(random_start2)]
        if candidates:
            random_start2[i] = random.choice(candidates)
    starts.append(random_start2)
    
    best_family = None
    best_score = -1

    for idx, start in enumerate(starts):
        if time.time() >= deadline - 5:
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
