import time
import random
import numpy as np

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
        # New sums are {x+a for a in arr} U {2x}
        # If any of these are in `sums`, or if there's a conflict among them, x cannot be added.
        # If x CAN be added (no conflicts), the set is NOT maximal.
        
        # Check conflicts with existing sums
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

def _greedy_construct(order, rng=None):
    """
    Greedily constructs a Sidon set by iterating through `order`.
    """
    if order is None:
        order = list(range(1, N + 1))
        if rng:
            rng.shuffle(order)
        else:
            random.shuffle(order)
            
    selected = []
    sums = set()
    
    for x in order:
        conflict = False
        for a in selected:
            if (x + a) in sums:
                conflict = True
                break
        if conflict:
            continue
            
        if (2 * x) in sums:
            continue
            
        selected.append(x)
        for a in selected:
            sums.add(x + a)
        sums.add(2 * x)
        
    return selected

def _local_search_reduce(current_set, rng, deadline):
    """
    Attempts to reduce the size of a maximal Sidon set by:
    1. Trying to remove an element and repair maximality by adding smaller elements.
    2. Swapping elements to allow removal of another element.
    
    Returns the smallest set found.
    """
    current = sorted(current_set)
    best = current[:]
    best_size = len(current)
    
    # Precompute sums for current set
    def get_sums(arr):
        sums = set()
        for i in range(len(arr)):
            a = arr[i]
            for j in range(i, len(arr)):
                b = arr[j]
                s = a + b
                if s in sums:
                    return None  # Not Sidon
                sums.add(s)
        return sums
    
    def is_maximal(arr, sums):
        arr_set = set(arr)
        for x in range(1, N + 1):
            if x in arr_set:
                continue
            # Check if x can be added
            conflict = False
            for a in arr:
                if (x + a) in sums:
                    conflict = True
                    break
            if not conflict and (2 * x) in sums:
                conflict = True
            if not conflict:
                return False
        return True
    
    def try_add_smallest(arr, sums, max_size):
        """
        Given a Sidon set `arr` (not necessarily maximal), add the smallest possible elements
        from [1, N] to make it maximal, without increasing size beyond max_size.
        Returns the new set and sums, or None if impossible within max_size.
        """
        arr = sorted(arr)
        sums = get_sums(arr)
        if sums is None:
            return None
        arr_set = set(arr)
        
        # Try to add elements in increasing order to block gaps
        # We want to keep size small, so only add if necessary for maximality
        # Actually, we can just greedily add the smallest elements that don't break Sidon
        # until it's maximal.
        
        # But we want minimal size. Let's try to add the smallest available element
        # that doesn't conflict, repeatedly, until maximal.
        
        # More sophisticated: only add elements that are "forced" or that help block many gaps.
        # For now, simple greedy: add smallest element that can be added.
        
        new_arr = arr[:]
        new_sums = sums
        new_arr_set = arr_set
        
        while not is_maximal(new_arr, new_sums):
            if len(new_arr) >= max_size:
                return None
            
            added = False
            for x in range(1, N + 1):
                if x in new_arr_set:
                    continue
                # Check if x can be added
                conflict = False
                for a in new_arr:
                    if (x + a) in new_sums:
                        conflict = True
                        break
                if conflict:
                    continue
                if (2 * x) in new_sums:
                    continue
                
                # Add x
                new_arr.append(x)
                new_arr.append(2 * x)
                for a in new_arr:
                    new_sums.add(x + a)
                new_sums.add(2 * x)
                new_arr_set.add(x)
                added = True
                break
            
            if not added:
                return None
        
        return sorted(new_arr), new_sums
    
    # Main loop: try to remove elements
    max_attempts = 1000
    attempts = 0
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        arr = sorted(best)
        sums = get_sums(arr)
        if sums is None:
            continue
        arr_set = set(arr)
        
        if not is_maximal(arr, sums):
            # Try to repair
            res = try_add_smallest(arr, sums, len(arr))
            if res:
                new_arr, new_sums = res
                if len(new_arr) < best_size:
                    best = new_arr
                    best_size = len(new_arr)
            continue
        
        # Try removing a random element
        if len(arr) > 5:
            idx = rng.randrange(len(arr))
            removed = arr[idx]
            new_arr = [x for x in arr if x != removed]
            
            # Check if new_arr is still Sidon
            new_sums = get_sums(new_arr)
            if new_sums is None:
                continue
            
            # Try to make it maximal again
            res = try_add_smallest(new_arr, new_sums, len(best))
            if res:
                final_arr, final_sums = res
                if len(final_arr) < best_size:
                    best = final_arr
                    best_size = len(final_arr)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_set = []
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Initial deterministic passes
    orders = [
        list(range(N, 0, -1)),
        list(range(1, N + 1)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        s = _greedy_construct(order)
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            
    # Randomized restarts
    while time.time() < deadline:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        s = _greedy_construct(order, rng)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            
            if best_size <= 13:
                break
    
    # Local search to reduce size
    if time.time() < deadline and best_set:
        improved = _local_search_reduce(best_set, rng, deadline)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    if not _is_valid_solution(best_set):
        # Fallback to a known valid set
        # Greedy from descending order is usually valid
        s = _greedy_construct(list(range(N, 0, -1)))
        if _is_valid_solution(s):
            return s
        # Last resort
        s = _greedy_construct(list(range(1, N + 1)))
        return s
        
    return best_set

# EVOLVE-BLOCK-END
