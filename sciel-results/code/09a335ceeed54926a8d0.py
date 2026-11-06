import time
import sys

# Increase recursion limit just in case, though depth is max 20
sys.setrecursionlimit(1000)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for 120s limit
    
    # Initial set
    initial = [1, 2, 5, 11]
    
    # Pre-compute differences for initial set
    def get_diffs(A):
        diffs = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diffs.add(A[i] - A[j] if A[i] > A[j] else A[j] - A[i])
        return diffs

    initial_diffs = get_diffs(initial)
    
    # We want to find a set of size K containing initial
    # K can be at most 20 (theoretical max for length 283/284)
    # We try from 20 down to len(initial)
    
    best_set = initial[:]
    
    def search(A, diffs, target_size):
        nonlocal best_set, deadline
        
        if time.time() > deadline:
            return None
        
        if len(A) == target_size:
            return A[:]
        
        # Try to add a new element x
        # To optimize, we can try x in a specific order.
        # Since the initial elements are small, the ruler likely extends to the right.
        # We iterate x from 1 to 284.
        # We can skip x if it's already in A.
        
        # Heuristic: Try x that are "far" from existing ones first? 
        # Or just iterate.
        # Let's iterate x from 1 to 284.
        
        for x in range(1, 285):
            if x in A:
                continue
            
            # Check if x can be added
            # Calculate differences with all elements in A
            new_diffs = set()
            valid = True
            for y in A:
                d = abs(x - y)
                if d in diffs:
                    valid = False
                    break
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Recurse
            # Sort A to maintain canonical form? 
            # A is not necessarily sorted in my loop, but I can sort it.
            # Actually, keeping A sorted helps with "x in A" check if we use binary search, 
            # but set is O(1).
            # Let's keep A as a list, but for "x in A" we can use a set for membership.
            # I'll pass A as a tuple or list. For efficiency, let's use a set for membership 
            # and a list for iteration?
            # The current `A` is a list. `x in A` is O(N). N is small.
            
            # Update state
            A_new = A + [x]
            diffs_new = diffs.union(new_diffs)
            
            res = search(A_new, diffs_new, target_size)
            if res:
                return res
        
        return None

    # We need to be careful with "x in A" if A is a list.
    # Let's use a set for A as well in the search to speed up membership.
    # But we need to iterate over A to compute diffs.
    
    def search_opt(A_list, A_set, diffs, target_size):
        nonlocal best_set, deadline
        
        if time.time() > deadline:
            return None
        
        if len(A_list) == target_size:
            return A_list[:]
        
        # Try adding x
        # Optimization: We can restrict x range?
        # The max element in the final set is <= 284.
        # The min element is >= 1.
        
        for x in range(1, 285):
            if x in A_set:
                continue
            
            new_diffs = set()
            valid = True
            for y in A_list:
                d = abs(x - y)
                if d in diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Recurse
            A_list_new = A_list + [x]
            A_set_new = A_set.union([x])
            diffs_new = diffs.union(new_diffs)
            
            res = search_opt(A_list_new, A_set_new, diffs_new, target_size)
            if res:
                return res
        
        return None

    # Try sizes from 20 down to 5
    # The initial set has size 4.
    # We need at least 4.
    for k in range(20, 3, -1):
        if time.time() > deadline:
            break
        res = search_opt(initial, set(initial), initial_diffs, k)
        if res:
            return res
        # If we fail for k, we might have found a smaller one? 
        # No, the search is exhaustive for that k.
        # We don't track partial bests easily here unless we modify search to update best_set.
        # But if we find a size 19, we should return it if 20 fails.
        # The loop handles this: if 20 fails, it tries 19.
        
    # If no larger set found, return initial
    return initial
