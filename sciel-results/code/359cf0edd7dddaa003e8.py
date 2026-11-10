"""Model-and-optimize: parameterize the 12-value ambient set, fit by exhaustive
Sidon-subset evaluation, minimize the largest Sidon (incl. diagonals) subset size."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _largest_sidon(s):
    """Exact largest subset of s with all unordered pair sums (incl. 2x) unique.
    Enumerates subsets by size descending; returns the largest valid size."""
    n = len(s)
    if n == 0:
        return 0
    from itertools import combinations
    for size in range(n, 0, -1):
        for combo in combinations(range(n), size):
            vals = [s[i] for i in combo]
            k = len(vals)
            seen_sums = set()
            is_sidon = True
            for a in range(k):
                for b in range(a, k):
                    sm = vals[a] + vals[b]
                    if sm in seen_sums:
                        is_sidon = False
                        break
                    seen_sums.add(sm)
                if not is_sidon:
                    break
            if is_sidon:
                return size
    return 0

def construct():
    deadline = time.time() + 80.0   # safety margin under 90s kill
    rng = random.Random(12345)
    best_set = None
    best_val = 10**9

    def try_set(vals):
        nonlocal best_set, best_val
        vals = sorted(set(vals))
        if len(vals) != 12:
            return
        v = _largest_sidon(vals)
        if v < best_val:
            best_val = v
            best_set = list(vals)
            if best_val <= 4:
                return  # can't do much better; minimum plausible is small

    # Key insight: to minimize the largest Sidon subset, we want the ambient set
    # to have many additive collisions. Clusters of closely-spaced values create
    # many duplicate pair sums, making it hard to find large Sidon subsets.
    
    # Seed with structured candidates
    structured = [
        list(range(0, 12)),
        [0,1,2,3,4,5, 64,65,66,67,68,69],
        [0,1,2,3,4,5, 120,121,122,123,124,125],
        [10,11,12,13,14,15, 70,71,72,73,74,75],
        [0,1,2, 40,41,42, 80,81,82, 120,121,122][:12],
        [0,1,2, 30,31,32, 60,61,62, 90,91,92][:12],
        [i*2 for i in range(12)],
        [i*3 for i in range(12)],
        [0,1,2,3,4,5,6, 32,33,34,35,36],
        [0,1,2,3,4,5,6,7, 64,65,66,67],
    ]
    for sc in structured:
        if len(set(sc)) == 12 and all(0 <= x <= 128 for x in sc):
            try_set(sc)

    # Random search with cluster bias
    def random_cluster_candidate():
        """Generate a random candidate with clustered structure."""
        method = rng.random()
        if method < 0.3:
            # Two clusters of 6
            c1_start = rng.randint(0, 60)
            c2_start = rng.randint(64, 123)
            vals = set()
            for i in range(6):
                vals.add(c1_start + i)
            for i in range(6):
                vals.add(c2_start + i)
            if len(vals) != 12:
                while len(vals) < 12:
                    vals.add(rng.randint(0, 128))
            return sorted(list(vals)[:12])
        elif method < 0.5:
            # Three clusters of 4
            c1 = rng.randint(0, 30)
            c2 = rng.randint(40, 80)
            c3 = rng.randint(90, 124)
            vals = set()
            for i in range(4):
                vals.add(c1 + i)
            for i in range(4):
                vals.add(c2 + i)
            for i in range(4):
                vals.add(c3 + i)
            if len(vals) != 12:
                while len(vals) < 12:
                    vals.add(rng.randint(0, 128))
            return sorted(list(vals)[:12])
        elif method < 0.7:
            # One big cluster of 8 + 4 scattered
            c1 = rng.randint(0, 120)
            vals = set()
            for i in range(8):
                vals.add(c1 + i)
            while len(vals) < 12:
                v = rng.randint(0, 128)
                if v not in vals:
                    vals.add(v)
            return sorted(list(vals)[:12])
        else:
            # Fully random
            return sorted(rng.sample(range(0, 129), 12))

    def neighbor(vals):
        vals = list(vals)
        i = rng.randrange(12)
        delta = rng.choice([-1, 1])
        newv = vals[i] + delta * rng.randint(1, 4)
        if 0 <= newv <= 128 and newv not in vals:
            vals[i] = newv
            vals.sort()
        return vals

    iters = 0
    while time.time() < deadline:
        cand = random_cluster_candidate()
        try_set(cand)
        if best_set is not None:
            for _ in range(5):
                nb = neighbor(best_set)
                try_set(nb)
        iters += 1
        if iters % 5000 == 0 and best_val <= 5:
            break

    if best_set is None:
        return list(range(12))
    return best_set
# EVOLVE-BLOCK-END
