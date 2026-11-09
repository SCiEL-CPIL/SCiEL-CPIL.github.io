import time
import random
import numpy as np

def _subset_sum_bitset(S, target=200):
    """
    Fast subset sum using Python integer bitset.
    Returns True if any subset of S sums to target.
    """
    bits = 1  # bit 0 set
    mask = (1 << (target + 1)) - 1
    for x in S:
        if x > target:
            continue
        bits |= (bits << x) & mask
    return (bits >> target) & 1 == 1

def _subset_sum_reachable(S, target=200):
    """
    Return the set of all achievable sums up to target.
    """
    reachable = {0}
    for x in S:
        if x > target:
            continue
        new = {s + x for s in reachable if s + x <= target}
        reachable |= new
    return reachable

def _can_add(S, x, target=200):
    """
    Check if adding x to S keeps no subset summing to target.
    Uses the existing reachable sums.
    """
    if x > target:
        return True
    # If target - x is in reachable sums, adding x creates a conflict
    # But we need to check if target is already reachable, or if target-x is reachable
    # Actually: after adding x, new reachable = old | {s+x for s in old if s+x <= target}
    # We need target not in new reachable
    # target in new reachable iff target in old OR (target - x in old and target - x + x == target)
    # Wait, that's not quite right. Let me think again.
    # New sums = old_sums ∪ {s + x : s in old_sums, s + x <= target}
    # target in new_sums iff target in old_sums OR (target - x in old_sums)
    # Because if target - x in old_sums, then (target - x) + x = target is in new_sums.
    # But we also need to make sure we're not double-counting.
    # Actually the condition is: target not in old_sums AND target - x not in old_sums
    # Wait, that's sufficient but is it necessary? 
    # If target is in new_sums, then either:
    # 1. target was in old_sums (but we assume S is valid, so no)
    # 2. target = s + x for some s in old_sums, i.e., target - x in old_sums
    # So yes: can add x iff target - x not in old_sums (given S is already valid)
    
    # But we need the reachable set. Let's compute it efficiently.
    # For performance, we'll pass the reachable set as a parameter in the main loop.
    pass

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    def _is_valid(S, target=200):
        """Check if S has no subset summing to target."""
        bits = 1
        mask = (1 << (target + 1)) - 1
        for x in S:
            if x > target:
                continue
            bits |= (bits << x) & mask
        return (bits >> target) & 1 == 0
    
    def _greedy_with_reachable(order, target=200):
        """
        Greedy: add elements in given order, tracking reachable sums.
        """
        selected = []
        reachable = {0}
        for x in order:
            if x > target:
                selected.append(x)
                continue
            # Can we add x? target must not become reachable.
            # target becomes reachable iff (target - x) in reachable
            if (target - x) in reachable:
                continue
            selected.append(x)
            # Update reachable
            new_sums = {s + x for s in reachable if s + x <= target}
            reachable |= new_sums
        
        return selected
    
    def _local_search_improve(S, target=200, max_iters=500):
        """
        Local search: try to add elements not in S.
        """
        current = list(S)
        if not _is_valid(current, target):
            return current
        
        best = current[:]
        best_size = len(current)
        
        # Compute reachable for current set
        reachable = _subset_sum_reachable(current, target)
        
        improved = True
        while improved and (time.time() - start_time) < deadline - 2:
            improved = False
            # Try to add each element not in current
            for x in range(1, 101):
                if x in current:
                    continue
                if x > target:
                    # Always safe to add
                    current.append(x)
                    reachable = _subset_sum_reachable(current, target)
                    if len(current) > best_size:
                        best_size = len(current)
                        best = current[:]
                    improved = True
                    break
                
                # Can we add x?
                if (target - x) not in reachable:
                    current.append(x)
                    new_sums = {s + x for s in reachable if s + x <= target}
                    reachable |= new_sums
                    if len(current) > best_size:
                        best_size = len(current)
                        best = current[:]
                    improved = True
                    break
        
        return best
    
    # Strategy 1: Multiple greedy orderings
    best_set = []
    best_size = 0
    
    orderings = [
        list(range(100, 0, -1)),      # descending
        list(range(1, 101)),          # ascending
        sorted(range(1, 101), key=lambda x: abs(x - 100)),  # by distance from 100
        sorted(range(1, 101), key=lambda x: x % 13),        # modular
        sorted(range(1, 101), key=lambda x: -x % 7),        # reverse modular
        list(range(100, 0, -2)) + list(range(99, 0, -2)),   # evens desc then odds desc
        list(range(101, 0, -2)) + list(range(100, 0, -2)),  # odds desc then evens desc
    ]
    
    for order in orderings:
        result = _greedy_with_reachable(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Strategy 2: Local search improvement from best
    if best_set and (time.time() - start_time) < deadline - 10:
        improved = _local_search_improve(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    # Strategy 3: Randomized restarts
    while (time.time() - start_time) < deadline - 5:
        # Random ordering
        order = list(range(1, 101))
        random.shuffle(order)
        result = _greedy_with_reachable(order)
        
        # Improve with local search
        if (time.time() - start_time) < deadline - 3:
            result = _local_search_improve(result, max_iters=200)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
        
        # Early termination if we've found a very good set
        if best_size >= 95:
            break
    
    # Final verification
    if best_set and not _is_valid(best_set):
        # Remove elements to fix
        for x in sorted(best_set, reverse=True):
            if _is_valid(best_set):
                break
            best_set.remove(x)
    
    if not best_set:
        return [100]
    
    return best_set
