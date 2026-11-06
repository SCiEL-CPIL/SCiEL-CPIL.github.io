import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    candidates = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        candidates.append(mask)
    
    N = len(candidates)  # 1820
    
    # Precompute pairwise intersections for O(1) sunflower check
    pair_int = [[0] * N for _ in range(N)]
    for i in range(N):
        ci = candidates[i]
        for j in range(i + 1, N):
            v = ci & candidates[j]
            pair_int[i][j] = v
            pair_int[j][i] = v
    
    def is_sunflower(i, j, k):
        return pair_int[i][j] == pair_int[i][k] == pair_int[j][k]
    
    def can_add(fam, idx):
        n = len(fam)
        for a_pos in range(n):
            a = fam[a_pos]
            for b_pos in range(a_pos + 1, n):
                b = fam[b_pos]
                if is_sunflower(a, b, idx):
                    return False
        return True
    
    def find_blocker(fam, idx):
        n = len(fam)
        for a_pos in range(n):
            a = fam[a_pos]
            for b_pos in range(a_pos + 1, n):
                b = fam[b_pos]
                if is_sunflower(a, b, idx):
                    return (a, b)
        return None
    
    best_family = []
    best_size = 0
    
    def greedy(seed_offset):
        random.seed(42 + seed_offset)
        order = list(range(N))
        random.shuffle(order)
        fam = []
        fam_set = set()
        for idx in order:
            if can_add(fam, idx):
                fam.append(idx)
                fam_set.add(idx)
        return fam, fam_set
    
    for attempt in range(25):
        if time.time() > deadline - 8:
            break
        
        fam, fam_set = greedy(attempt)
        sz = len(fam)
        
        if sz > best_size:
            best_size = sz
            best_family = fam.copy()
        
        # Phase 1: direct additions
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for idx in range(N):
                if idx in fam_set:
                    continue
                if can_add(fam, idx):
                    fam.append(idx)
                    fam_set.add(idx)
                    sz += 1
                    changed = True
                    if sz > best_size:
                        best_size = sz
                        best_family = fam.copy()
        
        # Phase 2: swaps
        for swap_round in range(200):
            if time.time() > deadline - 5:
                break
            random.seed(50000 + attempt * 1000 + swap_round)
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                add_idx = random.randint(0, N - 1)
                if add_idx in fam_set:
                    continue
                if can_add(fam, add_idx):
                    fam.append(add_idx)
                    fam_set.add(add_idx)
                    sz += 1
                    if sz > best_size:
                        best_size = sz
                        best_family = fam.copy()
                else:
                    blocker = find_blocker(fam, add_idx)
                    if blocker:
                        a, b = blocker
                        for victim in (a, b):
                            fam.remove(victim)
                            fam_set.discard(victim)
                            if can_add(fam, add_idx):
                                fam.append(add_idx)
                                fam_set.add(add_idx)
                            else:
                                fam.append(victim)
                                fam_set.add(victim)
        
        # Final greedy pass
        random.seed(60000 + attempt)
        order = list(range(N))
        random.shuffle(order)
        for idx in order:
            if idx not in fam_set and can_add(fam, idx):
                fam.append(idx)
                fam_set.add(idx)
        
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = fam.copy()
    
    # Final greedy pass on best
    random.seed(70000)
    order = list(range(N))
    random.shuffle(order)
    best_set = set(best_family)
    for idx in order:
        if idx not in best_set and can_add(best_family, idx):
            best_family.append(idx)
            best_set.add(idx)
    
    result = []
    for m in best_family:
        s = [i for i in range(1, 17) if (m >> i) & 1]
        result.append(sorted(s))
    
    return result
