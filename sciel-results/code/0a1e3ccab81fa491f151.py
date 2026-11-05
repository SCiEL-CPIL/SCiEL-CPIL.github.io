import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    N = 24
    # Precompute reciprocals
    recips = {i: Fraction(1, i) for i in range(1, N + 1)}
    
    # Helper: Check if target is a subset sum of reciprocals of elements in 'elements'
    # Uses meet-in-the-middle for efficiency
    def is_subset_sum(target, elements):
        if not elements:
            return target == Fraction(0, 1)
        
        n = len(elements)
        if n <= 10:
            # Brute force
            for mask in range(1, 1 << n):
                s = Fraction(0, 1)
                for i in range(n):
                    if mask & (1 << i):
                        s += recips[elements[i]]
                        if s > target:
                            break
                if s == target:
                    return True
            return False
        
        # Meet in the middle
        mid = n // 2
        left = elements[:mid]
        right = elements[mid:]
        
        left_sums = set()
        for mask in range(1 << len(left)):
            s = Fraction(0, 1)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += recips[left[i]]
            if s <= target:
                left_sums.add(s)
        
        right_sums = set()
        for mask in range(1 << len(right)):
            s = Fraction(0, 1)
            for i in range(len(right)):
                if mask & (1 << i):
                    s += recips[right[i]]
            if s <= target:
                right_sums.add(s)
        
        for ls in left_sums:
            needed = target - ls
            if needed in right_sums:
                # Ensure non-empty subset if target > 0
                # If target > 0, and ls=0, needed=target. 
                # If needed is in right_sums, it could be 0 (empty right) if target=0? 
                # But target is 1/x or 1/y - 1/x, which is > 0.
                # So at least one part must be non-zero.
                # If ls > 0, left is non-empty.
                # If ls == 0, then needed = target > 0, so right sum is target > 0, so right is non-empty.
                return True
        return False

    def can_add(S, x):
        """
        Check if x can be added to S while maintaining reciprocal-subsum-free property.
        S is a set of ints.
        """
        # 1. Check if 1/x is a subset sum of S
        if is_subset_sum(recips[x], list(S)):
            return False
        
        # 2. Check if adding x creates a violation for any y in S
        for y in S:
            if y < x:
                # 1/y = 1/x + sum(1/z for z in T)
                # sum(1/z for z in T) = 1/y - 1/x
                target = recips[y] - recips[x]
                if target < 0:
                    continue
                # T must be a subset of S \ {y}
                others = [z for z in S if z != y]
                if is_subset_sum(target, others):
                    return False
        return True

    # Strategy: Greedy with Local Search
    best_set = set()
    
    # Initial greedy construction
    current = set()
    for x in range(N, 0, -1):
        if can_add(current, x):
            current.add(x)
            
    if len(current) > len(best_set):
        best_set = set(current)
        
    # Local Search: Try to improve
    # We try to add elements, or swap elements to allow adding more.
    # Since we want to maximize size, we primarily look for additions.
    # If no addition is possible, we try swaps to see if they lead to a state where an addition is possible.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = set(best_set)
        
        # 1. Try to add any element
        added = False
        for x in range(N, 0, -1):
            if x not in current:
                if can_add(current, x):
                    current.add(x)
                    improved = True
                    added = True
                    break
        
        if added:
            if len(current) > len(best_set):
                best_set = set(current)
            continue
        
        # 2. If no addition, try swaps
        # Remove one, add one. This maintains size.
        # We hope this unblocks an addition in the next iteration.
        # To avoid infinite loops, we can use a temperature or just random swaps.
        # Let's try all single swaps and see if the resulting set allows an immediate addition.
        swap_found = False
        for i in sorted(current, reverse=True):
            for j in range(N, 0, -1):
                if j in current:
                    continue
                new_set = set(current)
                new_set.remove(i)
                # Check if we can add j to new_set
                if can_add(new_set, j):
                    new_set.add(j)
                    # Now check if we can add anything else to new_set
                    # If yes, we definitely want to take this swap
                    for k in range(N, 0, -1):
                        if k not in new_set:
                            if can_add(new_set, k):
                                # This swap leads to a larger set
                                new_set.add(k)
                                if len(new_set) > len(best_set):
                                    best_set = set(new_set)
                                    improved = True
                                    swap_found = True
                                break
                    if swap_found:
                        break
            if swap_found:
                break
        
        # If no improvement found via swap+add, we might be stuck at a local optimum.
        # We can try a random restart or a more aggressive search.
        # For now, if no improvement, we break to save time.
        # But we can try random swaps just to shuffle?
        # Given the time limit, and that greedy usually finds a good solution, 
        # and swap+add covers many cases, this should be sufficient.
        
    return sorted(best_set)
