"""Rewritten program: find 12-set in [0,128] minimizing largest Sidon subset size."""
import time
import random
from itertools import combinations

def _is_sidon_fast(subset):
    """Check if sorted subset is Sidon: all pair sums (i<=j) distinct."""
    n = len(subset)
    seen = set()
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon(ambient):
    """Exact largest Sidon subset size."""
    n = len(ambient)
    for k in range(n, 4, -1):
        for sub in combinations(ambient, k):
            if _is_sidon_fast(sub):
                return k
    for sub in combinations(ambient, 4):
        if _is_sidon_fast(sub):
            return 4
    return 3

def _count_sidon_pairs(ambient):
    """Count number of 4-tuples (a,b,c,d) with a<=b, c<=d, a+b=c+d, not all equal.
    More collisions means harder to extract large Sidon subsets."""
    n = len(ambient)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = ambient[i] + ambient[j]
            sums[s] = sums.get(s, 0) + 1
    # Count number of sum values that have multiplicity > 1
    collisions = 0
    for s, cnt in sums.items():
        if cnt > 1:
            collisions += cnt * (cnt - 1) // 2
    return collisions

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s kill
    
    best_set = None
    best_score = 12
    
    random.seed(42)
    
    # Phase 1: Generate diverse candidate sets
    candidates = []
    
    # Strategy A: Arithmetic progressions (many sum collisions)
    for a in range(0, 120, 2):
        for d in [1, 2, 3, 4, 5]:
            s = [a + i * d for i in range(12)]
            if max(s) <= 128 and len(set(s)) == 12:
                candidates.append(sorted(s))
    
    # Strategy B: Two tight clusters
    for c1 in range(5, 60, 5):
        for c2 in range(70, 125, 5):
            s1 = [c1 + i for i in range(6)]
            s2 = [c2 + i for i in range(6)]
            s = s1 + s2
            if len(set(s)) == 12 and max(s) <= 128:
                candidates.append(sorted(s))
    
    # Strategy C: Three clusters
    for c1 in range(5, 30, 5):
        for c2 in range(40, 70, 5):
            for c3 in range(90, 120, 5):
                s = [c1 + i for i in range(4)] + [c2 + i for i in range(4)] + [c3 + i for i in range(4)]
                if len(set(s)) == 12 and max(s) <= 128:
                    candidates.append(sorted(s))
    
    # Strategy D: Random dense clusters
    for _ in range(1000):
        n_clusters = random.choice([2, 3, 4])
        centers = sorted(random.sample(range(10, 120), n_clusters))
        s = []
        for idx, c in enumerate(centers):
            if idx < n_clusters - 1:
                cluster_size = 12 // n_clusters
            else:
                cluster_size = 12 - len(s)
            for i in range(cluster_size):
                val = c + random.randint(-4, 4)
                if 0 <= val <= 128 and val not in s:
                    s.append(val)
        while len(s) < 12:
            val = random.randint(0, 128)
            if val not in s:
                s.append(val)
        s = sorted(s[:12])
        if len(set(s)) == 12:
            candidates.append(s)
    
    # Strategy E: Random
    for _ in range(500):
        s = sorted(random.sample(range(129), 12))
        candidates.append(s)
    
    # Deduplicate
    seen_sets = set()
    unique_candidates = []
    for s in candidates:
        key = tuple(s)
        if key not in seen_sets:
            seen_sets.add(key)
            unique_candidates.append(s)
    
    # Phase 2: Quick filter using collision count, then exact evaluation
    # Sort by collision count (more collisions = potentially better)
    scored_candidates = []
    for s in unique_candidates:
        coll = _count_sidon_pairs(s)
        scored_candidates.append((coll, s))
    
    scored_candidates.sort(key=lambda x: -x[0])  # Most collisions first
    
    # Evaluate top candidates exactly
    for coll, s in scored_candidates[:2000]:
        if time.time() > deadline - 15:
            break
        sc = _max_sidon(s)
        if sc < best_score:
            best_score = sc
            best_set = s[:]
        if best_score <= 4:
            break
    
    # If we haven't found good sets, evaluate more
    if best_score > 5:
        for coll, s in scored_candidates[2000:5000]:
            if time.time() > deadline - 15:
                break
            sc = _max_sidon(s)
            if sc < best_score:
                best_score = sc
                best_set = s[:]
            if best_score <= 4:
                break
    
    # Phase 3: Local search / hill climbing
    if best_set is not None and best_score > 4:
        cur = best_set[:]
        cur_sc = best_score
        stagnant = 0
        max_stagnant = 500
        
        while time.time() < deadline - 3:
            if stagnant > max_stagnant:
                break
            found_improve = False
            for i in range(12):
                if time.time() > deadline - 3:
                    break
                old = cur[i]
                # Try nearby values with larger range
                offsets = list(range(-20, 21))
                random.shuffle(offsets)
                for off in offsets:
                    new_val = old + off
                    if new_val < 0 or new_val > 128:
                        continue
                    if new_val == old:
                        continue
                    if new_val in cur:
                        continue
                    trial = cur[:]
                    trial[i] = new_val
                    trial.sort()
                    sc = _max_sidon(trial)
                    if sc < cur_sc:
                        cur = trial
                        cur_sc = sc
                        best_set = cur[:]
                        best_score = cur_sc
                        stagnant = 0
                        found_improve = True
                        if cur_sc <= 4:
                            break
                    else:
                        stagnant += 1
                if cur_sc <= 4:
                    break
            if cur_sc <= 4:
                break
            
            # Occasionally try a larger jump
            if stagnant > 100 and time.time() < deadline - 5:
                i = random.randint(0, 11)
                old = cur[i]
                new_val = random.randint(0, 128)
                if new_val != old and new_val not in cur:
                    trial = cur[:]
                    trial[i] = new_val
                    trial.sort()
                    sc = _max_sidon(trial)
                    if sc < cur_sc:
                        cur = trial
                        cur_sc = sc
                        best_set = cur[:]
                        best_score = cur_sc
                        stagnant = 0
                        if cur_sc <= 4:
                            break
    
    # Phase 4: Try specific structured sets known to have many sum collisions
    if best_score > 4 and time.time() < deadline - 3:
        # Sets with many additive relations
        structured = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
            [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12],
            [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
            [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
            [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
            [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13],
            [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
            [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 24],
        ]
        
        for s in structured:
            if time.time() > deadline - 3:
                break
            if len(set(s)) != 12 or max(s) > 128:
                continue
            sc = _max_sidon(s)
            if sc < best_score:
                best_score = sc
                best_set = s[:]
            if best_score <= 4:
                break
    
    if best_set is None:
        best_set = list(range(12))
    
    return best_set
