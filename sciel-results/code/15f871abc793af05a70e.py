import time
import random

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Data structures
    S = []  # List of chosen integers
    in_S = [False] * (max_val + 1)  # in_S[i] is True if i is in S
    # counts[s] = number of pairs (a,b) in S with a<=b and a+b=s
    # Max sum is 2000
    counts = [0] * (2 * max_val + 1)
    
    # Helper to add a number to the set and update counts
    # Returns True if successful, False if it would violate B2[2]
    def try_add(x):
        # Check 2*x
        if counts[2 * x] >= 2:
            return False
        # Check x + y for y in S
        for y in S:
            if y == x:
                continue
            s = x + y
            if counts[s] >= 2:
                return False
        # If we get here, it's valid.
        # Perform addition
        S.append(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
        return True

    # Helper to remove a number
    def try_remove(x):
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1

    # Helper to check if a swap (remove a, add c) is valid
    # Does not modify state, just checks
    def is_valid_swap(a, c):
        # Sums affected:
        # Removed: 2*a, a+y (y in S, y!=a)
        # Added: 2*c, c+y (y in S\{a})
        
        # We need to check if any resulting count > 2
        
        # Let's use a temporary dictionary for deltas.
        deltas = {}
        
        # Remove a
        deltas[2*a] = deltas.get(2*a, 0) - 1
        for y in S:
            if y != a:
                s = a + y
                deltas[s] = deltas.get(s, 0) - 1
        
        # Add c
        deltas[2*c] = deltas.get(2*c, 0) + 1
        for y in S:
            if y != a:
                s = c + y
                deltas[s] = deltas.get(s, 0) + 1
        
        # Check all affected sums
        for s, d in deltas.items():
            new_count = counts[s] + d
            if new_count > 2:
                return False
            if new_count < 0:
                return False # Should not happen
        
        return True

    # Helper to find blocking elements for candidate c
    # Returns a list of elements in S that prevent c from being added
    def find_blockers(c):
        blockers = set()
        # Check 2*c
        if counts[2 * c] >= 2:
            # Find all pairs (a,b) in S with a+b = 2*c
            for a in S:
                b = 2 * c - a
                if b in in_S and b >= a and a != b:
                    blockers.add(a)
                    blockers.add(b)
                elif a == b and in_S[a]:
                    blockers.add(a)
        # Check c + y for y in S
        for y in S:
            s = c + y
            if counts[s] >= 2:
                # Find all pairs (a,b) in S with a+b = s
                for a in S:
                    b = s - a
                    if b in in_S and b >= a and a != b:
                        blockers.add(a)
                        blockers.add(b)
                    elif a == b and in_S[a]:
                        blockers.add(a)
        return list(blockers)

    # 1. Greedy Initialization
    for i in range(1, max_val + 1):
        if try_add(i):
            pass
        # Greedy just skips if invalid. It doesn't backtrack.

    # 2. Local Search
    # We will try to add random unused numbers.
    # If it fails, we try to swap a blocking element with the candidate.
    
    # To speed up random selection of unused numbers, we can keep a list of unused numbers.
    # But maintaining it is overhead. 
    # Random picking from 1..1000 and checking in_S is O(1).
    
    while time.time() < deadline:
        c = random.randint(1, max_val)
        if in_S[c]:
            continue
            
        # Try to add c
        if try_add(c):
            continue
            
        # If add failed, try to find a swap
        # We need to identify why it failed.
        # The failure is due to some sum K having count >= 2 in S, and c adding to it.
        # Or 2*c having count >= 2.
        
        # Let's try targeted swaps with blocking elements first
        if not S:
            continue
            
        # Find blockers and try to swap with them
        blockers = find_blockers(c)
        
        # Try up to 10 targeted swaps with blockers first
        swapped = False
        for a in blockers:
            if is_valid_swap(a, c):
                # Perform swap
                try_remove(a)
                try_add(c)
                swapped = True
                break
        
        # If targeted swap failed, try a few random swaps
        if not swapped:
            for _ in range(5):
                a = random.choice(S)
                if is_valid_swap(a, c):
                    # Perform swap
                    try_remove(a)
                    try_add(c)
                    swapped = True
                    break
        
        # If we didn't swap, we just discard c for now.
        # The loop continues.

    return S
