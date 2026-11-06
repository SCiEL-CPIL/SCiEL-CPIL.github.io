"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    def try_greedy(order):
        """Try greedy inclusion in given order, return the set."""
        reachable = 1  # bit 0 set, sum 0 is reachable
        selected = []
        for x in order:
            # Check if adding x creates sum 200
            # new sums = old sums shifted by x
            new_reachable = reachable | (reachable << x)
            if (new_reachable >> 200) & 1:
                continue
            selected.append(x)
            reachable = new_reachable
        return selected
    
    # Try multiple orderings and pick the best
    best_set = []
    
    # Strategy 1: Descending (original seed approach)
    s1 = try_greedy(range(100, 0, -1))
    if len(s1) > len(best_set):
        best_set = s1
    
    if time.time() > deadline:
        return best_set if best_set else [1]
    
    # Strategy 2: Ascending
    s2 = try_greedy(range(1, 101))
    if len(s2) > len(best_set):
        best_set = s2
    
    if time.time() > deadline:
        return best_set if best_set else [1]
    
    # Strategy 3: Even numbers first, then odd
    evens = list(range(2, 101, 2))
    odds = list(range(1, 101, 2))
    s3 = try_greedy(evens + odds)
    if len(s3) > len(best_set):
        best_set = s3
    
    if time.time() > deadline:
        return best_set if best_set else [1]
    
    # Strategy 4: Large evens, small odds, etc. - try a few more
    # Strategy: sorted by value mod 200, preferring values that are far from complementing existing sums
    # Simpler: try random-ish permutations with fixed seeds for determinism
    import random
    
    best_len = len(best_set)
    
    # Try a few deterministic "smart" orderings
    # Order by: prefer elements > 100 (none), then elements where 200-x is large (harder to form)
    # Actually, prefer elements x where 200-x is NOT in the set we're building
    # This is complex; let's try a few more simple orderings
    
    # Strategy 5: Multiples of 2 first (evens), descending
    s5 = try_greedy(sorted(evens, reverse=True))
    if len(s5) > best_len:
        best_set = s5
        best_len = len(s5)
    
    if time.time() > deadline:
        return best_set if best_set else [1]
    
    # Strategy 6: Multiples of 3 first
    mult3 = list(range(3, 101, 3))
    others = [x for x in range(1, 101) if x % 3 != 0]
    s6 = try_greedy(sorted(mult3, reverse=True) + sorted(others, reverse=True))
    if len(s6) > best_len:
        best_set = s6
        best_len = len(s6)
    
    if time.time() > deadline:
        return best_set if best_set else [1]
    
    # Strategy 7: Try local search - start with best set, try to add excluded elements
    # First, get the set of excluded elements from best_set
    best_set_set = set(best_set)
    excluded = [x for x in range(1, 101) if x not in best_set_set]
    
    # Try to add excluded elements one by one (in various orders)
    for order in [excluded, sorted(excluded, reverse=True), sorted(excluded)]:
        current = list(best_set)
        current_set = set(current)
        # Recompute reachable for current
        reachable = 1
        for x in current:
            reachable |= (reachable << x)
        
        improved = False
        for x in order:
            if x in current_set:
                continue
            new_reachable = reachable | (reachable << x)
            if not ((new_reachable >> 200) & 1):
                current.append(x)
                current_set.add(x)
                reachable = new_reachable
                improved = True
        
        if improved and len(current) > best_len:
            best_set = current
            best_len = len(current)
    
    if time.time() > deadline:
        return best_set if best_set else [1]
    
    # Strategy 8: Try removing some elements and re-adding others
    # Start from best, try swapping: remove one, add two
    # This is more complex; skip for now if time is tight
    
    # Final: ensure nonempty
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
