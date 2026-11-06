import time
import numpy as np
import random

BASE = 51975

# Precompute all divisors of 51975 greater than 1
def _get_divisors(n):
    divs = set()
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)

DIVISORS = _get_divisors(BASE)

# Precompute index arrays for each (m, r) pair
# For each divisor m and residue r, store the array of indices r, r+m, r+2m, ... < BASE
# This avoids repeated np.arange calls and slicing overhead in tight loops
_INDEX_CACHE = {}
for m in DIVISORS:
    for r in range(m):
        _INDEX_CACHE[(m, r)] = np.arange(r, BASE, m)

def _greedy_build(order, base=BASE):
    """
    Greedy initialization: add classes in given order, picking best residue for each.
    Returns (classes, counts, uncovered_count)
    """
    counts = np.zeros(base, dtype=np.int8)
    uncovered = base
    classes = []
    
    for m in order:
        best_r = 0
        best_gain = -1
        
        # Find the residue r for this modulus m that covers the most currently uncovered points
        for r in range(m):
            idx = _INDEX_CACHE[(m, r)]
            # Gain is the number of points in this residue class that are currently uncovered
            gain = np.sum(counts[idx] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        if best_gain <= 0:
            # If no gain, we might still want to add it? 
            # The problem asks for a set of classes. Adding a class with 0 gain doesn't hurt score 
            # but doesn't help. However, distinct moduli are required if we want to use all? 
            # No, "distinct moduli m > 1 dividing...". It doesn't say we MUST use all.
            # "With every divisor... available once". This implies we can choose a subset.
            # If gain is 0, adding it is useless for minimizing uncovered count. 
            # We can skip it to keep the list small? Or include it?
            # Including it doesn't change the score. Excluding it doesn't change the score.
            # Let's skip if gain is 0 to avoid clutter, unless we are forced to use all?
            # The prompt says "construct() -> classes [[m, r], ...] with distinct moduli...".
            # It doesn't explicitly say "use all divisors".
            # "With every divisor of 51975 available once... minimize".
            # This phrasing suggests we are selecting from the available pool.
            # So skipping 0-gain classes is valid and keeps the output cleaner.
            # However, sometimes a 0-gain class in greedy might become useful if we 
            # rearrange others? No, greedy is forward-only.
            # If we skip, we save time in local search (fewer classes to swap).
            continue

        classes.append([m, best_r])
        idx = _INDEX_CACHE[(m, best_r)]
        counts[idx] += 1
        uncovered -= best_gain
        
    return classes, counts, uncovered

def _local_search(classes, counts, uncovered, base=BASE, time_limit=10.0):
    """
    Local search: try changing residues for each class.
    Uses cached indices for speed.
    """
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Randomize order to break symmetry and avoid local minima cycles
        random.shuffle(classes)
        
        for i in range(len(classes)):
            if (time.time() - start_time) > time_limit:
                break
                
            m, r = classes[i]
            old_idx = _INDEX_CACHE[(m, r)]
            
            # Try all other residues for this modulus
            # Optimization: We can iterate r_new in a random order or just sequential.
            # Sequential is fine, but random might escape plateaus faster.
            # Let's try sequential first for determinism in early steps, 
            # but since we shuffle classes, the order of moduli changes.
            
            for r_new in range(m):
                if r_new == r:
                    continue
                
                new_idx = _INDEX_CACHE[(m, r_new)]
                
                # Calculate delta in uncovered count
                # If we move class from r to r_new:
                # 1. Points in old_idx lose coverage from this class.
                #    If count was 1, they become uncovered (+1 to uncovered count).
                #    If count was >1, they remain covered (0 change).
                # 2. Points in new_idx gain coverage from this class.
                #    If count was 0, they become covered (-1 to uncovered count).
                #    If count was >0, they remain covered (0 change).
                
                # Note: old_idx and new_idx are disjoint because r != r_new and m is the modulus.
                
                old_counts = counts[old_idx]
                new_counts = counts[new_idx]
                
                # Number of points that become uncovered
                lost_coverage = np.sum(old_counts == 1)
                # Number of points that become covered
                gained_coverage = np.sum(new_counts == 0)
                
                delta = lost_coverage - gained_coverage
                
                if delta < 0:
                    # Improvement
                    counts[old_idx] -= 1
                    counts[new_idx] += 1
                    classes[i][1] = r_new
                    uncovered += delta
                    improved = True
                    # Restart sweep on improvement to ensure we don't miss cascading improvements
                    break
    
    return classes, counts, uncovered

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We should aim to finish by 140s.
    deadline = 140.0
    
    best_classes = []
    best_counts = None
    best_uncovered = BASE
    
    # Phase 1: Greedy Initialization with multiple strategies
    strategies = [
        sorted(DIVISORS, reverse=True), # Descending
        sorted(DIVISORS),               # Ascending
    ]
    
    # Add a few random shuffles
    for _ in range(3):
        shuffled = DIVISORS.copy()
        random.shuffle(shuffled)
        strategies.append(shuffled)
    
    for order in strategies:
        if time.time() > deadline - 20:
            break
            
        classes, counts, uncovered = _greedy_build(order, BASE)
        
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_classes = [list(c) for c in classes]
            best_counts = counts.copy()
            
    if not best_classes:
        # Fallback if something went wrong (should not happen)
        best_classes = [[DIVISORS[0], 0]]
        best_counts = np.zeros(BASE, dtype=np.int8)
        best_counts[_INDEX_CACHE[(best_classes[0][0], 0)]] = 1
        best_uncovered = BASE - 1

    # Phase 2: Local Search on the best greedy result
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Use most of the remaining time for local search
        # Leave a small buffer
        search_time = max(1.0, remaining_time - 2.0)
        best_classes, best_counts, best_uncovered = _local_search(
            best_classes, best_counts, best_uncovered, BASE, time_limit=search_time
        )
        
    # Phase 3: If we have significant time left, try another random restart + local search
    if time.time() < deadline - 15:
        # Try one more random greedy + short local search
        shuffled = DIVISORS.copy()
        random.shuffle(shuffled)
        classes_r, counts_r, uncovered_r = _greedy_build(shuffled, BASE)
        
        if uncovered_r < best_uncovered:
            best_uncovered = uncovered_r
            best_classes = [list(c) for c in classes_r]
            best_counts = counts_r.copy()
            
        # If this new start is competitive, give it some local search time
        if time.time() < deadline - 5:
            time_left = deadline - time.time() - 2
            if time_left > 2:
                classes_r, counts_r, uncovered_r = _local_search(
                    classes_r, counts_r, uncovered_r, BASE, time_limit=time_left
                )
                if uncovered_r < best_uncovered:
                    best_uncovered = uncovered_r
                    best_classes = [list(c) for c in classes_r]
                    best_counts = counts_r.copy()

    return best_classes

# Quick self-test if run directly (not required for submission, but good for debug)
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    # Calculate score for verification
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in res:
        covered[r::m] = 1
    score = BASE - np.sum(covered)
    print(f"Time: {t1-t0:.2f}s, Score: {score}, Classes: {len(res)}")
