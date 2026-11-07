import time
import random
import itertools

def _count_comparable(family_masks):
    # family_masks: list of ints (bitmasks)
    n = len(family_masks)
    cnt = 0
    for i in range(n):
        a = family_masks[i]
        for j in range(i+1, n):
            b = family_masks[j]
            if (a & b) == a and a != b:
                cnt += 1
            elif (a & b) == b and a != b:
                cnt += 1
    return cnt

def construct():
    deadline = time.time() + 80.0  # safety margin under 90s
    n = 12
    m = 40
    all_masks = list(range(1 << n))  # 4096
    
    best = None
    best_score = -1
    
    def score_of(fam):
        return _count_comparable(fam)
    
    # Precompute subset relation bitsets for speed in local search
    # For each mask, precompute list of masks it's a strict subset of, and strict superset of
    # 4096 x 4096 bitsets = 4096 * 4096 bits = 2M bits = 256KB, fine
    # Use Python ints as bitsets
    subset_of = [0] * (1 << n)  # subset_of[a] has bit b set if a is strict subset of b
    superset_of = [0] * (1 << n)
    for a in range(1 << n):
        for b in range(1 << n):
            if a != b and (a & b) == a:
                subset_of[a] |= (1 << b)
                superset_of[b] |= (1 << a)
    
    def greedy_init(rng):
        # Start with empty and full, then greedily add
        fam = [0, (1 << n) - 1]
        fam_set = set(fam)
        # Candidate ordering: prefer middle sizes
        candidates = [mm for mm in all_masks if mm not in fam_set]
        # Sort by popcount near n/2 first for diversity, but greedy by gain
        while len(fam) < m:
            best_gain = -1
            best_cand = None
            for c in candidates:
                gain = 0
                for f in fam:
                    if (c & f) == c and c != f:
                        gain += 1
                    elif (c & f) == f and c != f:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cand = c
            fam.append(best_cand)
            fam_set.add(best_cand)
            candidates.remove(best_cand)
        return fam
    
    def local_search(fam, rng):
        fam = fam[:]
        fam_set = set(fam)
        cur = score_of(fam)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swap: remove one, add another
            for i in range(m):
                if time.time() > deadline:
                    break
                a = fam[i]
                # Try adding a new element not in fam_set
                # Sample a few candidates
                for _ in range(30):
                    c = rng.choice(all_masks)
                    if c in fam_set:
                        continue
                    new_fam = fam[:]
                    new_fam[i] = c
                    ns = score_of(new_fam)
                    if ns > cur:
                        fam = new_fam
                        fam_set.discard(a)
                        fam_set.add(c)
                        cur = ns
                        improved = True
                        break
        return fam, cur
    
    rng = random.Random(12345)
    
    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        fam = greedy_init(rng)
        fam, sc = local_search(fam, rng)
        if sc > best_score:
            best_score = sc
            best = fam[:]
    
    # Final: convert masks to lists of ints
    if best is None:
        best = list(range(m))
    result = []
    for mask in best:
        result.append([b for b in range(n) if (mask >> b) & 1])
    return result
