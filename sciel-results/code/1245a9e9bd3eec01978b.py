import random
import time

def construct():
    n = 512
    
    # Precompute conflicts
    # conflicts[x] is a list of frozensets of 3 integers
    conflicts = [[] for _ in range(n + 1)]
    
    for x in range(1, n + 1):
        # x is the 4th term
        for d in range(1, (x - 1) // 3 + 1):
            a = x - 3 * d
            conflicts[x].append(frozenset((a, a + d, a + 2 * d)))
            
        # x is the 2nd term
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            a = x - d
            conflicts[x].append(frozenset((a, x + d, x + 2 * d)))
            
        # x is the 3rd term
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            a = x - 2 * d
            conflicts[x].append(frozenset((a, a + d, x + d)))
            
        # x is the 1st term
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            conflicts[x].append(frozenset((x + d, x + 2 * d, x + 3 * d)))
            
    # Calculate conflict counts for heuristic ordering
    conflict_counts = [0] * (n + 1)
    for x in range(1, n + 1):
        conflict_counts[x] = len(conflicts[x])

    def is_valid_addition(x, current_set):
        for tri in conflicts[x]:
            if tri.issubset(current_set):
                return False
        return True

    def greedy(order):
        selected = set()
        for x in order:
            if is_valid_addition(x, selected):
                selected.add(x)
        return selected

    def local_search(initial_set):
        s = set(initial_set)
        improved = True
        while improved:
            improved = False
            # Try to add
            for x in range(1, n + 1):
                if x not in s and is_valid_addition(x, s):
                    s.add(x)
                    improved = True
            # Try to swap
            if not improved:
                # To speed up, we can limit swaps or use a time check
                # But let's do a full pass if possible
                for u in list(s):
                    for v in range(1, n + 1):
                        if v not in s:
                            new_s = (s - {u}) | {v}
                            if is_valid_addition(v, new_s):
                                s = new_s
                                improved = True
                                break
                    if improved:
                        break
        return s

    # Strategy 1: Random Greedy
    best_set = set()
    best_size = 0
    
    start_time = time.time()
    time_limit = 50.0 # Leave 10s for safety
    
    # Run random greedy
    for _ in range(1000):
        if time.time() - start_time > time_limit:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Strategy 2: Sorted by Conflict Count (Descending)
    order_desc = sorted(range(1, n + 1), key=lambda x: conflict_counts[x], reverse=True)
    s = greedy(order_desc)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        
    # Strategy 3: Sorted by Conflict Count (Ascending)
    order_asc = sorted(range(1, n + 1), key=lambda x: conflict_counts[x])
    s = greedy(order_asc)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        
    # Apply Local Search to the best set
    if time.time() - start_time < time_limit - 5.0:
        improved_set = local_search(best_set)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
            
    return sorted(best_set)
