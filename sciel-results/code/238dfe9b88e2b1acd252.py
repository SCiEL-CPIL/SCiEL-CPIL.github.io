import time
import random
from itertools import combinations

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit

    def time_left():
        return deadline - time.time()

    # Precompute subset/superset relationships using bit tricks
    subsets_of = [[] for _ in range(total_masks)]
    supersets_of = [[] for _ in range(total_masks)]
    
    for a in range(total_masks):
        sub = a
        while True:
            subsets_of[a].append(sub)
            if sub == 0:
                break
            sub = (sub - 1) & a
    
    for a in range(total_masks):
        for s in range(total_masks):
            if (a & ~s) == 0:
                supersets_of[a].append(s)

    # Build comparability table
    comp_table = [set() for _ in range(total_masks)]
    for a in range(total_masks):
        for s in subsets_of[a]:
            if s != a:
                comp_table[a].add(s)
        for s in supersets_of[a]:
            if s != a:
                comp_table[a].add(s)

    def calc_score(family_list):
        score = 0
        L = len(family_list)
        for i in range(L):
            for j in range(i + 1, L):
                if family_list[j] in comp_table[family_list[i]]:
                    score += 1
        return score

    def calc_comp_counts(family_list):
        counts = {}
        for a in family_list:
            count = 0
            for b in family_list:
                if a != b and b in comp_table[a]:
                    count += 1
            counts[a] = count
        return counts

    def greedy_construct(start_chain=None):
        if start_chain is None:
            start_chain = []
            cur = 0
            start_chain.append(cur)
            for i in range(n):
                cur |= (1 << i)
                start_chain.append(cur)
        
        family = list(start_chain)
        used = set(family)
        
        while len(family) < m and time_left() > 3:
            best_gain = -1
            best_mask = None
            for s in range(total_masks):
                if s in used:
                    continue
                gain = 0
                for f in family:
                    if s in comp_table[f]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = s
            if best_mask is None:
                break
            family.append(best_mask)
            used.add(best_mask)
        
        return family

    def hill_climb(start_family, max_iterations=500):
        family_set = set(start_family)
        fam_list = list(family_set)
        current_score = calc_score(fam_list)
        
        improved = True
        iterations = 0
        while improved and time_left() > 2 and iterations < max_iterations:
            improved = False
            iterations += 1
            
            best_delta = 0
            best_a = None
            best_b = None
            
            comp_counts = calc_comp_counts(fam_list)
            
            for a in fam_list:
                comp_a = comp_counts[a]
                for b in range(total_masks):
                    if b in family_set:
                        continue
                    new_count = 0
                    for f in fam_list:
                        if f != a and b in comp_table[f]:
                            new_count += 1
                    delta = new_count - comp_a
                    if delta > best_delta:
                        best_delta = delta
                        best_a = a
                        best_b = b
            
            if best_delta > 0:
                family_set.remove(best_a)
                family_set.add(best_b)
                fam_list = list(family_set)
                current_score += best_delta
                improved = True
        
        return list(family_set), current_score

    def two_opt(family_list):
        family_set = set(family_list)
        fam_list = list(family_set)
        current_score = calc_score(fam_list)
        
        improved = True
        while improved and time_left() > 2:
            improved = False
            best_delta = 0
            best_a1 = None
            best_a2 = None
            best_b1 = None
            best_b2 = None
            
            for i in range(len(fam_list)):
                for j in range(i + 1, len(fam_list)):
                    a1 = fam_list[i]
                    a2 = fam_list[j]
                    for b1 in range(total_masks):
                        if b1 in family_set:
                            continue
                        for b2 in range(total_masks):
                            if b2 in family_set or b2 == b1:
                                continue
                            new_family = [x for x in fam_list if x != a1 and x != a2] + [b1, b2]
                            new_score = calc_score(new_family)
                            delta = new_score - current_score
                            if delta > best_delta:
                                best_delta = delta
                                best_a1 = a1
                                best_a2 = a2
                                best_b1 = b1
                                best_b2 = b2
            
            if best_delta > 0:
                family_set.remove(best_a1)
                family_set.remove(best_a2)
                family_set.add(best_b1)
                family_set.add(best_b2)
                fam_list = list(family_set)
                current_score += best_delta
                improved = True
        
        return list(family_set), current_score

    # Generate multiple diverse starts
    starts = []
    
    # Start 1: Full chain + greedy fill
    starts.append(greedy_construct())
    
    # Start 2: Multiple short chains
    family2 = []
    used2 = set()
    chain_a = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095]
    for c in chain_a:
        if c < total_masks and c not in used2:
            family2.append(c)
            used2.add(c)
    chain_b = [2, 6, 14, 30, 62, 126, 254, 510, 1022, 2046]
    for c in chain_b:
        if c < total_masks and c not in used2:
            family2.append(c)
            used2.add(c)
    chain_c = [4, 12, 28, 60, 124, 252, 508, 1020]
    for c in chain_c:
        if c < total_masks and c not in used2:
            family2.append(c)
            used2.add(c)
    while len(family2) < m and time_left() > 3:
        best_gain = -1
        best_mask = None
        for s in range(total_masks):
            if s in used2:
                continue
            gain = 0
            for f in family2:
                if s in comp_table[f]:
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
    random.seed(42)
    for _ in range(5):
        i = random.randint(0, m - 1)
        candidates = [x for x in range(total_masks) if x not in set(random_start)]
        if candidates:
            random_start[i] = random.choice(candidates)
    starts.append(random_start)

    # Start 4: All sets of size <= 3, then fill
    family4 = []
    used4 = set()
    for size in range(4):
        for combo in combinations(range(n), size):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if mask not in used4:
                family4.append(mask)
                used4.add(mask)
            if len(family4) >= m:
                break
        if len(family4) >= m:
            break
    while len(family4) < m and time_left() > 3:
        best_gain = -1
        best_mask = None
        for s in range(total_masks):
            if s in used4:
                continue
            gain = 0
            for f in family4:
                if s in comp_table[f]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = s
        if best_mask is None:
            break
        family4.append(best_mask)
        used4.add(best_mask)
    starts.append(family4)

    # Start 5: Dense cluster around a mid-sized set
    family5 = []
    used5 = set()
    # Pick a central mask with many comparables
    central = 0b010101010101  # alternating bits
    family5.append(central)
    used5.add(central)
    # Add all subsets and supersets greedily by gain
    candidates = [s for s in range(total_masks) if s not in used5]
    while len(family5) < m and time_left() > 3:
        best_gain = -1
        best_mask = None
        for s in candidates:
            if s in used5:
                continue
            gain = 0
            for f in family5:
                if s in comp_table[f]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = s
        if best_mask is None or best_gain <= 0:
            break
        family5.append(best_mask)
        used5.add(best_mask)
        candidates.remove(best_mask)
    # Fill remaining if needed
    while len(family5) < m and time_left() > 3:
        best_gain = -1
        best_mask = None
        for s in range(total_masks):
            if s in used5:
                continue
            gain = 0
            for f in family5:
                if s in comp_table[f]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = s
        if best_mask is None:
            break
        family5.append(best_mask)
        used5.add(best_mask)
    starts.append(family5)

    best_family = None
    best_score = -1

    for idx, start in enumerate(starts):
        if time_left() < 5:
            break
        fam, score = hill_climb(start)
        if score > best_score:
            best_score = score
            best_family = fam

    if best_family is None:
        best_family = greedy_construct()

    # Final polish: try 2-opt on best result if time permits
    if time_left() > 10:
        polished, polished_score = two_opt(best_family)
        if polished_score > best_score:
            best_family = polished
            best_score = polished_score

    # Convert to lists of vertices
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
