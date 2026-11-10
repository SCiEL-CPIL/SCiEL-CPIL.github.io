import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    
    # Precompute conflicts
    # conflicts[i] is a set of j such that i and j conflict
    conflicts = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            x = i // g
            y = j // g
            if g % (x + y) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Initial set: all odds
    S = set(range(1, N + 1, 2))
    
    # List of evens not in S
    evens = [i for i in range(2, N + 1, 2) if i not in S]
    
    # Greedy addition of non-conflicting evens
    # We repeatedly scan and add any even that has no conflict with current S
    # To speed up, we can just iterate until no change
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate over a copy since we modify evens
        for e in list(evens):
            if time.time() > deadline:
                break
            # Check if e conflicts with any element in S
            # conflicts[e] contains all numbers that conflict with e
            # We need conflicts[e] to be disjoint from S
            if not conflicts[e].intersection(S):
                S.add(e)
                evens.remove(e)
                changed = True
                # Optional: break and restart loop to handle new conflicts?
                # No, adding e might create conflicts for other evens, but we check against S dynamically.
                # So we can continue scanning.
    
    # Local Search / Swapping
    # Try to improve by swapping elements
    # If we remove an odd o, we might be able to add multiple evens.
    # Let's try a simple hill climbing:
    # Try to remove one element from S and add a set of evens that are now free.
    # This is complex. Let's stick to the greedy result if it's good enough.
    # The knowledge says greedy gets 1.445.
    
    # To be safe, let's add a simple local search for swaps of 1 odd for 1 even if it allows adding more?
    # Actually, just returning the greedy result is likely sufficient based on the prompt's hint.
    
    return list(S)
# EVOLVE-BLOCK-END
