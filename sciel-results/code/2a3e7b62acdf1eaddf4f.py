"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_sidon_subset(subset):
    """Check if a subset has all unordered pair sums (including diagonals) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _largest_sidon_subset_size(ambient):
    """Find the size of the largest Sidon subset of ambient."""
    n = len(ambient)
    if n <= 1:
        return n
    if n <= 2:
        return 2 if _is_sidon_subset(ambient) else 1
    # Try subsets from largest to smallest
    for size in range(n, 0, -1):
        for comb in combinations(ambient, size):
            if _is_sidon_subset(list(comb)):
                return size
    return 1

def _count_sum_collisions(ambient):
    """Count the number of non-trivial additive quadruples a+b=c+d."""
    # For each pair of pairs, check if they have the same sum
    pairs = list(combinations(range(len(ambient)), 2))
    sum_map = {}
    for i, j in pairs:
        s = ambient[i] + ambient[j]
        if s not in sum_map:
            sum_map[s] = []
        sum_map[s].append((i, j))
    collisions = 0
    for s, pairs_list in sum_map.items():
        if len(pairs_list) > 1:
            collisions += len(pairs_list) - 1
    return collisions

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best_set = list(range(12))
    best_sidon_size = _largest_sidon_subset_size(best_set)
    
    random.seed(42)
    
    # Try structured sets with high collision density
    candidate_sets = []
    
    # Arithmetic progressions have many collisions
    for start in range(0, 100):
        for step in range(1, 12):
            s = [start + i * step for i in range(12)]
            if all(0 <= x <= 128 for x in s):
                candidate_sets.append(s)
    
    # Sets with many repeated differences
    for base in range(0, 50):
        for d1 in range(1, 20):
            for d2 in range(1, 20):
                if d1 == d2:
                    continue
                s = [base + i * d1 for i in range(6)] + [base + 5 * d1 + i * d2 for i in range(6)]
                if all(0 <= x <= 128 for x in s) and len(set(s)) == 12:
                    candidate_sets.append(s)
    
    # Random sets
    for _ in range(2000):
        s = random.sample(range(0, 129), 12)
        candidate_sets.append(s)
    
    # Evaluate candidate sets
    for s in candidate_sets:
        if time.time() > deadline:
            break
        sz = _largest_sidon_subset_size(s)
        if sz < best_sidon_size:
            best_sidon_size = sz
            best_set = s[:]
            if best_sidon_size <= 4:
                break
    
    # Local search: try swapping elements to improve
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                for j in range(12):
                    if i == j:
                        continue
                    for new_val in range(0, 129):
                        if new_val in best_set:
                            continue
                        old_val = best_set[j]
                        best_set[j] = new_val
                        sz = _largest_sidon_subset_size(best_set)
                        if sz < best_sidon_size:
                            best_sidon_size = sz
                            improved = True
                            if best_sidon_size <= 4:
                                break
                        else:
                            best_set[j] = old_val
                    if best_sidon_size <= 4:
                        break
                if best_sidon_size <= 4:
                    break
    
    return best_set
# EVOLVE-BLOCK-END
