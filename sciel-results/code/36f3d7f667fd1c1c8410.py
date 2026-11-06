"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
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
    # Check k=4
    for sub in combinations(ambient, 4):
        if _is_sidon_fast(sub):
            return 4
    return 3

def construct():
    start = time.time()
    deadline = start + 85
    
    best_set = None
    best_score = 12
    
    # Precompute: try to find sets with max Sidon size = 5
    # Strategy: use clustered values to force sum collisions
    
    random.seed(12345)
    
    # Generate candidate sets
    candidates = []
    
    # Dense clusters
    for _ in range(500):
        # Create 2-3 clusters
        n_clusters = random.choice([2, 3])
        centers = sorted(random.sample(range(10, 120), n_clusters))
        s = []
        for c in centers:
            cluster_size = 12 // n_clusters
            if centers.index(c) == len(centers) - 1:
                cluster_size = 12 - len(s)
            for i in range(cluster_size):
                val = c + random.randint(-3, 3)
                if 0 <= val <= 128 and val not in s:
                    s.append(val)
        while len(s) < 12:
            val = random.randint(0, 128)
            if val not in s:
                s.append(val)
        s = sorted(s[:12])
        if len(set(s)) == 12:
            candidates.append(s)
    
    # Random
    for _ in range(300):
        s = sorted(random.sample(range(129), 12))
        candidates.append(s)
    
    # Evaluate
    for s in candidates:
        if time.time() > deadline - 10:
            break
        sc = _max_sidon(s)
        if sc < best_score:
            best_score = sc
            best_set = s[:]
        if best_score <= 4:
            break
    
    # Local improvement
    if best_set is not None and best_score > 4:
        cur = best_set[:]
        cur_sc = best_score
        stagnant = 0
        while time.time() < deadline - 3:
            if stagnant > 200:
                break
            found_improve = False
            for i in range(12):
                if time.time() > deadline - 3:
                    break
                old = cur[i]
                # Try nearby values first
                offsets = list(range(-10, 11))
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
    
    if best_set is None:
        best_set = list(range(12))
    
    return best_set
# EVOLVE-BLOCK-END
