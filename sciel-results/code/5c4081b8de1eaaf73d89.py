"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _count_sum_collisions(s):
    """Count number of repeated pair sums (including diagonals) in set s."""
    sums = []
    for i in range(len(s)):
        for j in range(i, len(s)):
            sums.append(s[i] + s[j])
    # Count collisions: number of sums that appear more than once
    from collections import Counter
    c = Counter(sums)
    collisions = 0
    for v in c.values():
        if v > 1:
            collisions += v - 1
    return collisions

def _largest_sidon_subset_size(s):
    """Find the largest subset of s with all pair sums (incl. diagonals) unique."""
    n = len(s)
    best = 0
    # Try all subsets from largest to smallest
    for size in range(n, 0, -1):
        for subset in combinations(s, size):
            # Check if subset is Sidon
            sums = set()
            ok = True
            for i in range(size):
                for j in range(i, size):
                    sm = subset[i] + subset[j]
                    if sm in sums:
                        ok = False
                        break
                    sums.add(sm)
                if not ok:
                    break
            if ok:
                return size
    return 1

def _evaluate_set(s):
    """Return the largest Sidon subset size for set s (lower is better)."""
    return _largest_sidon_subset_size(s)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline to be safe
    
    # Structural approach: start with sets that have high additive structure
    # Use arithmetic progressions and perturbations
    
    best_set = list(range(12))
    best_score = _evaluate_set(best_set)
    
    # Generate candidate sets with high collision potential
    random.seed(42)
    
    # Strategy: use clusters of close numbers to create many sum collisions
    candidates = []
    
    # Pattern 1: Two tight clusters
    for base1 in range(0, 60, 5):
        for base2 in range(base1 + 15, 129, 5):
            cluster1 = [base1 + i for i in range(6)]
            cluster2 = [base2 + i for i in range(6)]
            if max(cluster2) <= 128:
                candidates.append(cluster1 + cluster2)
    
    # Pattern 2: Three clusters
    for b1 in range(0, 40, 5):
        for b2 in range(b1 + 10, 70, 5):
            for b3 in range(b2 + 10, 129, 5):
                c1 = [b1 + i for i in range(4)]
                c2 = [b2 + i for i in range(4)]
                c3 = [b3 + i for i in range(4)]
                if max(c3) <= 128:
                    candidates.append(c1 + c2 + c3)
    
    # Pattern 3: Single tight cluster with some spread
    for base in range(0, 100, 3):
        s = [base + i for i in range(12)]
        if max(s) <= 128:
            candidates.append(s)
    
    # Pattern 4: Mixed - some tight, some spread
    for base in range(0, 80, 7):
        tight = [base + i for i in range(8)]
        spread = [base + 10 + i*3 for i in range(4)]
        s = tight + spread
        if max(s) <= 128 and len(set(s)) == 12:
            candidates.append(s)
    
    # Evaluate candidates
    for s in candidates:
        if time.time() > deadline:
            break
        if len(set(s)) != 12:
            continue
        if any(x < 0 or x > 128 for x in s):
            continue
        score = _evaluate_set(s)
        if score < best_score:
            best_score = score
            best_set = list(s)
            if best_score <= 4:  # Very good, stop early
                break
    
    # Local search / perturbation around best
    if time.time() < deadline - 5:
        current = list(best_set)
        current_score = best_score
        for iteration in range(500):
            if time.time() > deadline - 2:
                break
            # Random perturbation
            new_set = list(current)
            idx = random.randint(0, 11)
            new_val = random.randint(0, 128)
            while new_val in new_set:
                new_val = random.randint(0, 128)
            new_set[idx] = new_val
            
            if len(set(new_set)) != 12:
                continue
            if any(x < 0 or x > 128 for x in new_set):
                continue
            
            score = _evaluate_set(new_set)
            if score < current_score:
                current = new_set
                current_score = score
                if current_score < best_score:
                    best_score = current_score
                    best_set = list(current)
                    if best_score <= 4:
                        break
            elif random.random() < 0.1:  # Simulated annealing acceptance
                current = new_set
    
    return best_set
# EVOLVE-BLOCK-END
