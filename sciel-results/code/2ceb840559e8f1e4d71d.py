import time
import random
import sys

N = 256

# EVOLVE-BLOCK-START

def _is_valid_solution(arr):
    """
    Verifies:
    1. All elements are distinct integers in [1, N].
    2. It is a Sidon set (all unordered pair sums a+b with a<=b are distinct).
    3. It is maximal (adding any missing element from [1, N] destroys the Sidon property).
    """
    if not arr:
        return False
    
    arr_set = set(arr)
    if len(arr_set) != len(arr):
        return False
    
    for x in arr:
        if x < 1 or x > N:
            return False
            
    # Compute all sums
    sums = set()
    for i in range(len(arr)):
        a = arr[i]
        for j in range(i, len(arr)):
            b = arr[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
            
    # Check maximality
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        
        # Try to add x
        conflict = False
        for a in arr:
            if (x + a) in sums:
                conflict = True
                break
        if not conflict:
            if (2 * x) in sums:
                conflict = True
        
        if not conflict:
            # x can be added, so the set is not maximal.
            return False
        
    return True

def _can_add(x, current_set, sums):
    """Check if x can be added to current_set while maintaining Sidon property."""
    if (2 * x) in sums:
        return False
    for a in current_set:
        if (x + a) in sums:
            return False
    return True

def _is_maximal_partial(current_set, sums, remaining_slots, max_element):
    """
    Pruning heuristic: Check if it's possible to reach a maximal set of the required size.
    This is a weak check for performance, primarily ensuring we don't get stuck 
    with a set that *cannot* be made maximal with the remaining elements.
    
    Actually, a stronger pruning: 
    If the current set is NOT maximal with respect to elements > max_element, 
    and we have no more elements to add (remaining_slots == 0), it's a dead end.
    
    For IDDFS, we are building a set of size K. 
    If we have selected K elements, we MUST verify maximality.
    If we haven't selected K elements yet, we just need to ensure we *can* add more.
    
    A useful pruning: If there are no candidates left that can be added, 
    and we haven't reached size K, fail.
    """
    if remaining_slots == 0:
        # We must check if the current set is maximal.
        # But we are in the middle of backtracking, so we only know about elements <= max_element?
        # No, the set is fixed. We check maximality against ALL [1, N].
        return _is_valid_solution(current_set)
    
    # Check if there is at least one candidate in (max_element, N] that can be added.
    # If not, we cannot reach size K.
    for x in range(max_element + 1, N + 1):
        if _can_add(x, current_set, sums):
            return True
    return False

def _backtrack_search(start_elem, target_size, deadline):
    """
    Attempts to find a maximal Sidon set of exactly 'target_size' 
    using elements from [start_elem, N].
    Returns the set if found, None otherwise.
    """
    # We maintain state globally for speed in this specific constrained environment, 
    # but passing state is safer. Let's use a recursive helper with local state.
    
    # Optimization: We want SMALL sets. 
    # If we find a set of size K, we can stop.
    
    # State:
    # current_set: list of selected elements
    # sums: set of all pair sums
    # last_elem: last element added (to ensure order)
    
    def dfs(current_set, sums, last_elem):
        if time.time() > deadline:
            return None
        
        if len(current_set) == target_size:
            # Check maximality
            if _is_valid_solution(current_set):
                return current_set[:]
            return None
        
        # Pruning: If we can't fill the remaining slots, return None
        # Remaining slots = target_size - len(current_set)
        # Available elements = N - last_elem
        if N - last_elem < target_size - len(current_set):
            return None
        
        # Try adding next elements
        # Heuristic: Try larger elements first? Or smaller?
        # Larger elements create larger sums, potentially leaving more room for smaller ones?
        # Actually, for Sidon sets, larger gaps are often better.
        # Let's try from N down to last_elem + 1.
        
        for x in range(N, last_elem, -1):
            if _can_add(x, current_set, sums):
                # Add x
                new_sums = sums.copy()
                for a in current_set:
                    new_sums.add(x + a)
                new_sums.add(2 * x)
                
                new_set = current_set + [x]
                
                res = dfs(new_set, new_sums, x)
                if res is not None:
                    return res
        return None

    return dfs([], set(), 0)

def construct():
    start_time = time.time()
    # Safety margin: 55s
    deadline = start_time + 55.0
    
    # We know from literature/experience that minimal maximal Sidon sets for N=256
    # are likely around 12-16. 
    # The current best is ~14 (score 1.23 implies size 14 if seed was 17? 
    # Wait, score = seed/value. If seed is 17 and score 1.23, value = 17/1.23 = 13.8. 
    # So current best is likely 14.
    # Let's try to find 13, 12, 11...
    
    # Start from a lower bound. 
    # A Sidon set of size k requires sums up to ~k^2. 
    # Maximal sets are denser.
    # Let's try sizes 10 to 16.
    
    best_set = None
    best_size = float('inf')
    
    # Try small sizes first
    for target_size in range(10, 17):
        if time.time() > deadline:
            break
            
        # Randomize the start? No, the search space is ordered.
        # To improve diversity, we can randomize the candidate order slightly, 
        # but strict backtracking on N=256 might be too slow if the tree is large.
        
        # Optimization: Use a randomized restart approach within the target size.
        # If deterministic fails, try randomizing the "start" or candidate order.
        
        # Actually, for small target sizes (10-13), the search tree might be manageable 
        # if we prune well. But N=256 is large.
        
        # Alternative: Hybrid. 
        # 1. Try deterministic IDDFS for small k.
        # 2. If that fails, fall back to greedy + local search for larger k.
        
        res = _backtrack_search(1, target_size, deadline)
        if res is not None:
            best_set = res
            best_size = target_size
            break # Found the smallest!
            
    # If IDDFS didn't find anything small (or timed out), use the robust greedy method
    # to ensure we return *something* valid, hopefully as small as possible.
    if best_set is None:
        # Fallback to the original strategy's strength: Greedy with random restarts
        # to find a good maximal set.
        
        rng = random.Random(42)
        
        # Try a few specific orders
        orders = [
            list(range(N, 0, -1)),
            list(range(1, N + 1)),
        ]
        
        for order in orders:
            if time.time() > deadline:
                break
            # Greedy construct
            selected = []
            sums = set()
            for x in order:
                if _can_add(x, selected, sums):
                    selected.append(x)
                    for a in selected:
                        sums.add(x + a)
                    sums.add(2 * x)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = selected[:]
                
        # Random restarts for greedy
        while time.time() < deadline - 5.0:
            order = list(range(1, N + 1))
            rng.shuffle(order)
            
            selected = []
            sums = set()
            for x in order:
                if _can_add(x, selected, sums):
                    selected.append(x)
                    for a in selected:
                        sums.add(x + a)
                    sums.add(2 * x)
                    
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = selected[:]
                if best_size <= 13:
                    break

    # Final verification
    if best_set is None or not _is_valid_solution(best_set):
        # Absolute fallback: A known valid set (if we have one) or generate one
        # Greedy ascending usually works
        selected = []
        sums = set()
        for x in range(1, N + 1):
            if _can_add(x, selected, sums):
                selected.append(x)
                for a in selected:
                    sums.add(x + a)
                sums.add(2 * x)
        best_set = selected

    return best_set

# EVOLVE-BLOCK-END
