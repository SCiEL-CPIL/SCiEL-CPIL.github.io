import time
from math import lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute incompatibility: incompatible[a][b] = True if lcm(a,b) <= 256
    # Use a set for each number of incompatible partners
    incompatible = {x: set() for x in candidates}
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            if lcm(a, b) <= N:
                incompatible[a].add(b)
                incompatible[b].add(a)
    
    def is_compatible(x, selected_set):
        """Check if x is compatible with all in selected_set."""
        for y in selected_set:
            if y in incompatible[x]:
                return False
        return True
    
    def reciprocal_sum(selected):
        return sum(1.0 / x for x in selected)
    
    # Greedy construction: pick highest reciprocal (smallest number) that's compatible
    # But we need to be smart about ordering
    # Let's try multiple greedy orders
    
    best_set = set()
    best_score = 0.0
    
    # Strategy 1: Greedy by smallest number first
    selected = set()
    for x in candidates:  # ascending order = largest reciprocal first
        if is_compatible(x, selected):
            selected.add(x)
    score = reciprocal_sum(selected)
    if score > best_score:
        best_score = score
        best_set = set(selected)
    
    # Strategy 2: Greedy with lookahead - at each step, pick the number that gives best marginal gain
    # considering future compatibility
    selected2 = set()
    remaining = list(candidates)
    while remaining:
        best_x = None
        best_future = -1
        for x in remaining:
            if is_compatible(x, selected2):
                # Count how many remaining would still be compatible after adding x
                future_count = 0
                for y in remaining:
                    if y != x and is_compatible(y, selected2 | {x}):
                        future_count += 1
                # Score: reciprocal + small bonus for future flexibility
                val = 1.0 / x + 0.001 * future_count
                if val > best_future:
                    best_future = val
                    best_x = x
        if best_x is None:
            break
        selected2.add(best_x)
        remaining.remove(best_x)
    
    score2 = reciprocal_sum(selected2)
    if score2 > best_score:
        best_score = score2
        best_set = set(selected2)
    
    # Local search: try 1-for-1 swaps
    def local_search(current_set, iterations=1000):
        current_score = reciprocal_sum(current_set)
        improved = True
        iters = 0
        while improved and iters < iterations:
            improved = False
            iters += 1
            if time.time() - start_time > timeout - 5:
                break
            # Try adding a single element
            for x in candidates:
                if x not in current_set and is_compatible(x, current_set):
                    new_set = current_set | {x}
                    new_score = reciprocal_sum(new_set)
                    if new_score > current_score + 1e-12:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        break
            # Try 1-for-1 swaps
            for x in list(current_set):
                for y in candidates:
                    if y not in current_set:
                        # Check if y is compatible with current_set - {x}
                        test_set = current_set - {x}
                        if is_compatible(y, test_set):
                            new_set = test_set | {y}
                            new_score = reciprocal_sum(new_set)
                            if new_score > current_score + 1e-12:
                                current_set = new_set
                                current_score = new_score
                                improved = True
                                break
                if improved:
                    break
        return current_set, current_score
    
    best_set, best_score = local_search(best_set)
    
    # Try local search from both starting points
    if time.time() - start_time < timeout - 10:
        set2, score2 = local_search(selected2)
        if score2 > best_score:
            best_score = score2
            best_set = set2
    
    return list(best_set)
