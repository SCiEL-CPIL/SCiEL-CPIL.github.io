"""Find a large subset of [1,256] where no distinct pair a,b has (a+b) | (a*b)."""
import time
import random
from itertools import combinations

def _conflict(a, b):
    """Return True if (a+b) divides (a*b) for distinct a,b."""
    s = a + b
    if s == 0:
        return False
    return (a * b) % s == 0

def _build_conflict_set(n=256):
    """Precompute which pairs conflict."""
    conflicts = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if _conflict(a, b):
                conflicts.add((a, b))
    return conflicts

def _greedy_add(set_list, conflicts, candidates):
    """Greedily add elements from candidates to set_list, maintaining no conflicts."""
    current = set(set_list)
    added = []
    for x in candidates:
        ok = True
        for y in current:
            if (x, y) in conflicts or (y, x) in conflicts:
                ok = False
                break
        if ok:
            current.add(x)
            added.append(x)
    return added

def _local_search(initial_set, conflicts, time_limit=1.0):
    """Try to improve the set by swapping out elements for better ones."""
    current = set(initial_set)
    best = set(initial_set)
    best_size = len(initial_set)
    
    start = time.time()
    deadline = start + time_limit
    
    # Try single swaps: remove one, try adding another
    all_elements = list(range(1, 257))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element and adding a better one
        to_try = list(current)
        random.shuffle(to_try)
        for rem in to_try:
            if time.time() >= deadline:
                break
            trial = current - {rem}
            # Find best candidate to add
            best_add = None
            best_score = -1
            for x in all_elements:
                if x in trial:
                    continue
                # Check if x can be added
                ok = True
                for y in trial:
                    if (x, y) in conflicts or (y, x) in conflicts:
                        ok = False
                        break
                if ok:
                    # Score: prefer smaller numbers (they conflict less?)
                    # Actually just check it's valid
                    best_add = x
                    break  # Just take the first valid one for speed
            if best_add is not None and best_add not in current:
                new_set = trial | {best_add}
                if len(new_set) > len(current):
                    current = new_set
                    improved = True
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
    
    return best

def _multi_start_greedy(conflicts, num_starts=20, time_limit=5.0):
    """Run multiple greedy passes with different orderings."""
    best_set = set()
    best_size = 0
    
    start = time.time()
    deadline = start + time_limit
    
    all_elements = list(range(1, 257))
    
    for i in range(num_starts):
        if time.time() >= deadline:
            break
        
        # Different ordering strategies
        order = list(all_elements)
        if i % 4 == 0:
            pass  # natural order
        elif i % 4 == 1:
            order.sort(reverse=True)
        elif i % 4 == 2:
            random.shuffle(order)
        else:
            # Sort by number of conflicts (fewest first)
            def num_conflicts(x):
                return sum(1 for y in all_elements if y != x and ((x, y) in conflicts or (y, x) in conflicts))
            order.sort(key=num_conflicts)
        
        current = set()
        for x in order:
            ok = True
            for y in current:
                if (x, y) in conflicts or (y, x) in conflicts:
                    ok = False
                    break
            if ok:
                current.add(x)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    return best_set

def construct():
    start = time.time()
    deadline = start + 80  # 80s safety margin
    
    # Precompute conflicts
    conflicts = _build_conflict_set(256)
    
    # Initial: all odd numbers (known valid set)
    initial = set(range(1, 257, 2))
    best = set(initial)
    best_size = len(initial)
    
    # Multi-start greedy
    result = _multi_start_greedy(conflicts, num_starts=30, time_limit=20.0)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Local search to improve
    if time.time() < deadline - 10:
        improved = _local_search(best, conflicts, time_limit=15.0)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    # Try to add more elements via greedy on remaining
    if time.time() < deadline - 5:
        remaining = [x for x in range(1, 257) if x not in best]
        added = _greedy_add(best, conflicts, remaining)
        if added:
            best = best | set(added)
            best_size = len(best)
    
    # Final validation
    valid = True
    for a, b in combinations(sorted(best), 2):
        if _conflict(a, b):
            valid = False
            break
    
    if not valid or len(best) == 0:
        return list(range(1, 257, 2))
    
    return sorted(best)

# EVOLVE-BLOCK-END
