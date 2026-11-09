"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    def check_pair(a, b):
        """Return True if (a*b) % (a+b) != 0 (i.e., compatible)."""
        return (a * b) % (a + b) != 0
    
    def is_valid_set(s):
        """Check if all distinct pairs in set s are compatible."""
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if not check_pair(lst[i], lst[j]):
                    return False
        return True
    
    def can_add(num, s):
        """Check if num can be added to set s."""
        for x in s:
            if x == num:
                continue
            if not check_pair(x, num):
                return False
        return True
    
    # Start with all odd numbers (guaranteed valid since sum of two odds is even,
    # product is odd, so even number can't divide odd number)
    odds = set(x for x in range(1, 257) if x % 2 == 1)
    best_set = set(odds)
    best_size = len(best_set)
    
    evens = [x for x in range(2, 257) if x % 2 == 0]
    
    # Multiple greedy attempts with different random orderings
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        current = set(odds)
        random.shuffle(evens)
        
        for e in evens:
            if time.time() > deadline:
                break
            if can_add(e, current):
                current.add(e)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    # Local search: try to add more elements by removing conflicting ones
    # and then re-adding
    for ls_iter in range(50):
        if time.time() > deadline:
            break
        
        current = set(best_set)
        remaining = [x for x in range(1, 257) if x not in current]
        random.shuffle(remaining)
        
        improved = False
        for r in remaining:
            if time.time() > deadline:
                break
            # Try adding r; if it conflicts, see if we can remove one conflicting element
            conflicts = [x for x in current if not check_pair(x, r)]
            if not conflicts:
                current.add(r)
                improved = True
            elif len(conflicts) == 1:
                # Try removing the single conflict and adding r
                c = conflicts[0]
                # Check if removing c allows us to add r and maybe more
                new_set = current - {c}
                new_set.add(r)
                if len(new_set) > len(current):
                    current = new_set
                    improved = True
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
            improved = True
        
        if not improved:
            break
    
    return list(best_set)
# EVOLVE-BLOCK-END
