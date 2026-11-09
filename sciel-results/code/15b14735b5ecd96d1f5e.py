import time
import random

def _can_sum_to_200(S):
    """Check if any subset of S sums to exactly 200 using bitset DP."""
    if not S:
        return False
    reachable = 1  # bit i set means sum i is achievable
    for x in S:
        if x > 200:
            continue
        reachable |= (reachable << x)
        reachable &= (1 << 201) - 1
    return bool(reachable & (1 << 200))

def _find_conflict_bitset(S):
    """Find a subset of S that sums to 200. Returns list or None."""
    if not S:
        return None
    if not _can_sum_to_200(S):
        return None
    
    S_list = list(S)
    reach = [False] * 201
    last_elem = [-1] * 201
    prev_sum = [-1] * 201
    reach[0] = True
    
    for i, x in enumerate(S_list):
        if x > 200:
            continue
        for s in range(200 - x, -1, -1):
            if reach[s] and not reach[s + x]:
                reach[s + x] = True
                last_elem[s + x] = i
                prev_sum[s + x] = s
    
    if not reach[200]:
        return None
    
    subset = []
    s = 200
    while s > 0:
        i = last_elem[s]
        if i == -1:
            break
        subset.append(S_list[i])
        s = prev_sum[s]
    
    return subset if subset else None

def _find_multiple_conflicts(S, num_conflicts=5):
    """Find multiple distinct conflicts in S."""
    conflicts = []
    S_list = list(S)
    if not _can_sum_to_200(S_list):
        return conflicts
    
    # Find first conflict
    c1 = _find_conflict_bitset(S_list)
    if c1:
        conflicts.append(c1)
    
    # Try to find more by removing elements from first conflict and re-searching
    if c1:
        for elem in c1:
            reduced = [x for x in S_list if x != elem]
            c2 = _find_conflict_bitset(reduced)
            if c2 and c2 not in conflicts:
                conflicts.append(c2)
                if len(conflicts) >= num_conflicts:
                    break
    
    return conflicts

def _deletion_strategy(S):
    """Start with full set, delete elements to eliminate conflicts."""
    S = list(S)
    while _can_sum_to_200(S):
        # Find a conflict
        conflict = _find_conflict_bitset(S)
        if not conflict:
            break
        
        # Find which element to remove: try each element in the conflict
        # and pick the one whose removal allows keeping the most elements
        best_removal = None
        best_size_after = -1
        
        for elem in conflict:
            candidate = [x for x in S if x != elem]
            # Check if removing this element resolves all conflicts
            if not _can_sum_to_200(candidate):
                if len(candidate) > best_size_after:
                    best_size_after = len(candidate)
                    best_removal = elem
        
        if best_removal is None:
            # All single removals still leave conflicts; remove the largest in conflict
            best_removal = max(conflict)
        
        S.remove(best_removal)
    
    return S

def _greedy_addition(order):
    """Greedy addition in given order."""
    selected = []
    for x in order:
        if not _can_sum_to_200(selected + [x]):
            selected.append(x)
    return selected

def _local_search_add(current_set, excluded, deadline):
    """Try to add excluded elements one by one."""
    current_set = set(current_set)
    excluded = list(excluded)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in excluded[:]:
            if x in current_set:
                continue
            if not _can_sum_to_200(current_set | {x}):
                current_set.add(x)
                excluded.remove(x)
                improved = True
    return list(current_set)

def _swap_search(current_set, excluded, deadline):
    """Try to swap: add an excluded element, remove an included element."""
    current_set = set(current_set)
    excluded = list(excluded)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in excluded[:]:
            if x in current_set:
                continue
            test_set = current_set | {x}
            if _can_sum_to_200(test_set):
                conflict = _find_conflict_bitset(list(test_set))
                if conflict:
                    # Try removing each element in the conflict (except x)
                    for elem in conflict:
                        if elem == x:
                            continue
                        new_set = test_set - {elem}
                        if not _can_sum_to_200(new_set):
                            # Check if this swap is beneficial (net gain or same size but different composition)
                            current_set = new_set
                            excluded = [x for x in excluded if x not in current_set]
                            improved = True
                            break
                    if improved:
                        break
    return list(current_set)

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    best_set = None
    best_size = 0
    
    def update_best(s):
        nonlocal best_set, best_size
        if s and len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
    
    # Strategy 1: Deterministic greedy additions
    for order in [range(100, 0, -1), range(1, 101), 
                  sorted(range(1, 101), key=lambda x: x % 7),
                  sorted(range(1, 101), key=lambda x: -x % 13)]:
        if time.time() > deadline - 5:
            break
        selected = _greedy_addition(order)
        update_best(selected)
    
    # Strategy 2: Deletion strategy (start with all, remove conflicts)
    if time.time() < deadline - 20:
        result = _deletion_strategy(list(range(1, 101)))
        update_best(result)
    
    # Strategy 3: Randomized greedy additions
    while time.time() < deadline - 15:
        S = list(range(1, 101))
        random.shuffle(S)
        selected = _greedy_addition(S)
        update_best(selected)
    
    # Strategy 4: Local search - add excluded elements
    if best_set is not None and time.time() < deadline - 10:
        current_set = set(best_set)
        excluded = [x for x in range(1, 101) if x not in current_set]
        result = _local_search_add(current_set, excluded, deadline - 5)
        update_best(result)
    
    # Strategy 5: Swap search
    if best_set is not None and time.time() < deadline - 3:
        current_set = set(best_set)
        excluded = [x for x in range(1, 101) if x not in current_set]
        result = _swap_search(current_set, excluded, deadline - 1)
        update_best(result)
    
    # Final: verify best_set is valid
    if best_set is not None:
        # Ensure it's valid
        if _can_sum_to_200(best_set):
            # This shouldn't happen, but as a safety check
            # Try to fix by removing one element
            conflict = _find_conflict_bitset(best_set)
            if conflict:
                for elem in conflict:
                    candidate = [x for x in best_set if x != elem]
                    if not _can_sum_to_200(candidate):
                        best_set = candidate
                        break
    
    return best_set
