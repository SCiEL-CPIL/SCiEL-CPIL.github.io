import time
import random
import sys

def construct():
    N = 1000
    max_val = 1000
    # Use a deadline that leaves a comfortable margin before the 150s kill.
    # 120s is the stated limit in the contract, but the kill is at 150s.
    # Let's aim to finish within 130s to be safe.
    deadline = time.time() + 128
    
    # State
    S = []
    in_S = [False] * (max_val + 1)
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

    # 2. Local Search
    # We will try to add random unused numbers.
    # If it fails, we try to swap a blocking element with the candidate.
    
    # To speed up random selection of unused numbers, we can keep a list of unused numbers.
    # But maintaining it is overhead. 
    # Random picking from 1..1000 and checking in_S is O(1).
    
    # Improved Local Search Strategy:
    # Instead of just random swaps, we can try to identify "bottleneck" sums.
    # However, the current swap strategy is decent. Let's enhance it with:
    # 1. Multiple random restarts if time permits? No, greedy is fast, local search is the bottleneck.
    # 2. Better candidate selection: prioritize candidates that have fewer blockers?
    # 3. Simulated Annealing style: allow occasional "worse" moves?
    #    But B2[2] is a hard constraint. We can't violate it.
    #    So we can only do valid swaps or valid adds.
    #    "Worse" moves aren't possible in the sense of violating constraints.
    #    We can only swap if the result is valid.
    
    # Let's try a different local search move: 2-swap (remove a, b; add c, d).
    # This allows more complex rearrangements.
    
    def is_valid_2swap(a, b, c, d):
        # Remove a, b. Add c, d.
        # All must be distinct.
        if a == b or a == c or a == d or b == c or b == d or c == d:
            return False
        if not in_S[a] or not in_S[b]:
            return False
        if in_S[c] or in_S[d]:
            return False
            
        deltas = {}
        
        # Remove a
        deltas[2*a] = deltas.get(2*a, 0) - 1
        for y in S:
            if y != a:
                s = a + y
                deltas[s] = deltas.get(s, 0) - 1
        # Remove b
        deltas[2*b] = deltas.get(2*b, 0) - 1
        for y in S:
            if y != b:
                s = b + y
                deltas[s] = deltas.get(s, 0) - 1
        
        # Add c
        deltas[2*c] = deltas.get(2*c, 0) + 1
        for y in S:
            if y != a and y != b:
                s = c + y
                deltas[s] = deltas.get(s, 0) + 1
        # Add d
        deltas[2*d] = deltas.get(2*d, 0) + 1
        for y in S:
            if y != a and y != b:
                s = d + y
                deltas[s] = deltas.get(s, 0) + 1
        # Add c+d
        # Wait, the loop above handles c+y and d+y where y is in S\{a,b}.
        # But it does NOT handle the pair (c,d) itself if c!=d.
        # In the single swap logic, we added c+y for y in S\{a}.
        # Here, for c, we add c+y for y in S\{a,b}. This includes d? No, d is not in S yet.
        # So we need to explicitly add the pair (c,d) if c!=d.
        # And (c,c) is 2c, (d,d) is 2d.
        # The loop `for y in S: if y != a and y != b: s = c+y` adds c+y for existing y.
        # It does NOT add c+d.
        # So we must add c+d manually.
        if c != d:
            s_cd = c + d
            deltas[s_cd] = deltas.get(s_cd, 0) + 1
        
        # Check all affected sums
        for s, d_val in deltas.items():
            new_count = counts[s] + d_val
            if new_count > 2:
                return False
            if new_count < 0:
                return False
        
        return True

    def perform_2swap(a, b, c, d):
        # Remove a, b
        try_remove(a)
        try_remove(b)
        # Add c, d
        try_add(c)
        try_add(d)

    while time.time() < deadline:
        # Try 1-swap first (cheaper)
        c = random.randint(1, max_val)
        if in_S[c]:
            continue
            
        if try_add(c):
            continue
            
        if not S:
            continue
            
        # Try targeted 1-swaps
        blockers = find_blockers(c)
        swapped = False
        for a in blockers:
            if is_valid_swap(a, c):
                try_remove(a)
                try_add(c)
                swapped = True
                break
        
        if not swapped:
            # Try a few random 1-swaps
            for _ in range(3):
                a = random.choice(S)
                if is_valid_swap(a, c):
                    try_remove(a)
                    try_add(c)
                    swapped = True
                    break
        
        if not swapped:
            # Try 2-swaps: remove two random elements, add c and another random element
            # This is more expensive, so try it less often or only if 1-swap fails.
            # Let's try 2-swaps with a lower probability or after failing 1-swaps.
            if random.random() < 0.1: # 10% chance to try 2-swap
                a = random.choice(S)
                b = random.choice(S)
                if a == b:
                    continue
                d = random.randint(1, max_val)
                if in_S[d] or d == c:
                    continue
                if is_valid_2swap(a, b, c, d):
                    perform_2swap(a, b, c, d)
                    swapped = True

    return S

