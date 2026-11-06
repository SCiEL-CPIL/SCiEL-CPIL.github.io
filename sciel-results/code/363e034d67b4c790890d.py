# EVOLVE-BLOCK-START
import random
import time
import itertools
import numpy as np

def _is_sidon_fast(subset):
    """Fast Sidon check with early termination using a set of sums."""
    n = len(subset)
    if n <= 1:
        return True
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _count_additive_relations(ambient):
    """Count the number of (not necessarily distinct) 4-tuples (a,b,c,d) in ambient
    with a+b=c+d where {a,b} != {c,d}. More relations = harder to find large Sidon subsets."""
    n = len(ambient)
    relations = 0
    sum_count = {}
    for i in range(n):
        for j in range(i, n):
            s = ambient[i] + ambient[j]
            if s in sum_count:
                relations += sum_count[s]
            sum_count[s] = sum_count.get(s, 0) + 1
    return relations

def _max_sidon_size_exact(ambient, time_limit=None, start_time=None):
    """Find largest Sidon subset exactly by trying all combinations from large to small.
    Returns the maximum k such that some k-subset is Sidon."""
    if start_time is None:
        start_time = time.time()
    n = len(ambient)
    ambient = list(ambient)
    
    # Check if full set is Sidon
    if _is_sidon_fast(ambient):
        return n
    
    # Try decreasing k
    for k in range(n - 1, 1, -1):
        if time_limit and (time.time() - start_time) > time_limit:
            break
        found = False
        for combo in itertools.combinations(ambient, k):
            if _is_sidon_fast(list(combo)):
                return k
            if time_limit and (time.time() - start_time) > time_limit:
                break
        if time_limit and (time.time() - start_time) > time_limit:
            break
    return 2

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    random.seed(42)
    
    best_set = None
    best_sidon_size = 999
    
    evaluated_sets = set()
    
    def evaluate(candidate):
        nonlocal best_set, best_sidon_size
        if len(candidate) != 12 or len(set(candidate)) != 12:
            return
        if any(x < 0 or x > 128 for x in candidate):
            return
        key = tuple(sorted(candidate))
        if key in evaluated_sets:
            return
        evaluated_sets.add(key)
        
        sidon_size = _max_sidon_size_exact(candidate, time_limit=2.0, start_time=start_time)
        if sidon_size < best_sidon_size:
            best_sidon_size = sidon_size
            best_set = list(candidate)
    
    # Strategy 1: Dense consecutive sets (most collisions)
    # Consecutive integers have maximum pair-sum collisions
    for start in range(0, 120):
        if time.time() > deadline:
            break
        candidate = list(range(start, start + 12))
        if candidate[-1] <= 128:
            evaluate(candidate)
    
    # Strategy 2: Arithmetic progressions with small step
    for step in range(1, 6):
        for start in range(0, 129 - 11 * step + 1):
            if time.time() > deadline:
                break
            candidate = [start + i * step for i in range(12)]
            evaluate(candidate)
    
    # Strategy 3: Dense clusters - groups of consecutive numbers
    # e.g., [0,1,2, 5,6,7, 10,11,12, 15,16,17]
    for gap in range(1, 15):
        for cluster_size in range(1, 7):
            num_clusters = 12 // cluster_size
            remainder = 12 % cluster_size
            if time.time() > deadline:
                break
            for base in range(0, 50):
                candidate = []
                for c in range(num_clusters + (1 if remainder else 0)):
                    cs = cluster_size if c < num_clusters else remainder
                    for i in range(cs):
                        val = base + c * gap + i
                        if val <= 128:
                            candidate.append(val)
                if len(candidate) == 12:
                    evaluate(candidate)
    
    # Strategy 4: Random dense sets (small range)
    for _ in range(500):
        if time.time() > deadline:
            break
        # Pick 12 random numbers from a small range
        range_size = random.randint(12, 40)
        start = random.randint(0, 128 - range_size + 1)
        candidate = random.sample(range(start, start + range_size), 12)
        evaluate(candidate)
    
    # Strategy 5: Local search from best found
    if best_set is not None and best_sidon_size > 2:
        current = list(best_set)
        current_sidon = best_sidon_size
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, 11)
            old_val = current[idx]
            # Try values near current position for denser sets
            if random.random() < 0.5:
                new_val = random.randint(max(0, old_val - 10), min(128, old_val + 10))
            else:
                new_val = random.randint(0, 128)
            
            if new_val == old_val or new_val in current:
                continue
            
            current[idx] = new_val
            sidon_size = _max_sidon_size_exact(current, time_limit=1.5, start_time=start_time)
            
            if sidon_size < best_sidon_size:
                best_sidon_size = sidon_size
                best_set = list(current)
                current_sidon = sidon_size
            else:
                current[idx] = old_val
    
    # Strategy 6: Try to force no-5-Sidon by maximizing sum multiplicities
    # Use a set where many pairs share the same sum
    # e.g., pairs that sum to the same value
    for _ in range(200):
        if time.time() > deadline:
            break
        # Create pairs that sum to specific targets
        candidate = []
        used = set()
        targets = random.sample(range(2, 250), 6)  # 6 targets, 2 elements each
        for t in targets:
            # Find two distinct numbers summing to t
            found = False
            for a in range(0, 129):
                b = t - a
                if a != b and a not in used and b not in used and 0 <= a <= 128 and 0 <= b <= 128:
                    candidate.append(a)
                    candidate.append(b)
                    used.add(a)
                    used.add(b)
                    found = True
                    break
            if not found:
                candidate = None
                break
        if candidate and len(candidate) == 12:
            evaluate(candidate)
    
    # Final fallback
    if best_set is None or len(set(best_set)) != 12 or any(x < 0 or x > 128 for x in best_set):
        return list(range(12))
    
    return best_set
# EVOLVE-BLOCK-END
