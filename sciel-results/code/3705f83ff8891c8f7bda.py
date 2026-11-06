"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _count_sums(chosen):
    """Count all pairwise sums a+b where a<=b from chosen."""
    counts = {}
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            s = chosen[i] + chosen[j]
            counts[s] = counts.get(s, 0) + 1
    return counts

def _can_add(chosen, counts, x):
    """Check if x can be added to chosen without violating B2[2]."""
    n = len(chosen)
    # Check sums with existing elements
    for i in range(n):
        s = x + chosen[i]
        if counts.get(s, 0) >= 2:
            return False
    # Check 2*x
    if counts.get(2 * x, 0) >= 2:
        return False
    return True

def _greedy_construct(order):
    """Greedy construction following a specific order of candidates."""
    chosen = []
    counts = {}
    for x in order:
        if _can_add(chosen, counts, x):
            chosen.append(x)
            # Update counts
            for y in chosen[:-1]:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            s2 = 2 * x
            counts[s2] = counts.get(s2, 0) + 1
    return chosen

def _local_improve(chosen, max_iters=500):
    """Try to improve the set by swapping elements."""
    chosen = sorted(chosen)
    counts = _count_sums(chosen)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try removing each element and adding a different one
        for i in range(len(chosen)):
            # Try removing chosen[i]
            temp_chosen = chosen[:i] + chosen[i+1:]
            temp_counts = _count_sums(temp_chosen)
            # Try adding each candidate not in temp_chosen
            for x in range(1, 1001):
                if x in temp_chosen:
                    continue
                if _can_add(temp_chosen, temp_counts, x):
                    new_chosen = temp_chosen + [x]
                    new_counts = _count_sums(new_chosen)
                    # Check if this is better (same size but maybe better structure)
                    # For now, just accept any valid swap
                    chosen = sorted(new_chosen)
                    counts = new_counts
                    improved = True
                    break
            if improved:
                break
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Try multiple different orderings
    best_set = []
    best_size = 0
    
    # Ordering 1: Natural ascending
    order1 = list(range(1, 1001))
    set1 = _greedy_construct(order1)
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1
    
    if time.time() > deadline:
        return best_set
    
    # Ordering 2: Descending
    order2 = list(range(1000, 0, -1))
    set2 = _greedy_construct(order2)
    if len(set2) > best_size:
        best_size = len(set2)
        best_set = set2
    
    if time.time() > deadline:
        return best_set
    
    # Ordering 3: Random shuffle with multiple seeds
    random.seed(42)
    for trial in range(10):
        order = list(range(1, 1001))
        random.shuffle(order)
        s = _greedy_construct(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
        if time.time() > deadline - 30:
            break
    
    if time.time() > deadline - 20:
        return best_set
    
    # Local improvement on the best set found
    improved = _local_improve(best_set, max_iters=100)
    if len(improved) > len(best_set):
        best_set = improved
    
    return best_set
# EVOLVE-BLOCK-END
