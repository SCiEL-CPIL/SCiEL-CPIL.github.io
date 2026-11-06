import time
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 50
    
    # Initial set: 1 to 22
    S = list(range(1, 23))
    
    # Precompute lcm compatibility? 
    # Or just check on the fly.
    
    def is_compatible(x, S):
        for y in S:
            if x * y // gcd(x, y) > limit:
                return False
        return True
        
    # Greedy add
    for x in range(23, limit + 1):
        if time.time() > deadline - 10:
            break
        if is_compatible(x, S):
            S.append(x)
            
    # Local search: try to swap elements
    # For each element in S, try removing it and adding a better one
    # Or just try to add more if possible (already done)
    # Try to replace elements to allow more additions?
    
    # A simple improvement: 
    # Iterate through candidates not in S.
    # If a candidate x is not compatible with S, find an element y in S 
    # such that removing y allows x to be added, and the new set is larger or same size but "better" (e.g. allows more future additions).
    # Given time constraints, a simple swap loop:
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S_copy = S[:]
        # Try adding each x not in S
        for x in range(1, limit + 1):
            if x in S_copy:
                continue
            if not is_compatible(x, S_copy):
                # Find a y in S_copy to remove
                # Heuristic: remove y that is "most conflicting" with x
                # Or just try all y? 512 * 25 is small.
                for y in S_copy:
                    S_temp = S_copy[:]
                    S_temp.remove(y)
                    if is_compatible(x, S_temp):
                        # Check if this swap is beneficial
                        # Ideally, we want to increase size or maintain size and improve "potential"
                        # For now, accept if it allows adding x and we haven't tried this x before?
                        # To avoid cycles, only accept if x > y or something?
                        # Or just accept and see if we can add more later.
                        # Let's just accept if it leads to a valid set.
                        # To ensure progress, we can require that the new set allows adding at least one more element 
                        # or simply use a random restart / hill climbing.
                        
                        # Simple heuristic: accept swap if x is larger (expands range) 
                        # or if we can add more elements afterwards.
                        # Let's just accept and mark improved.
                        S_copy = S_temp
                        S_copy.append(x)
                        improved = True
                        break
                if improved:
                    S = S_copy
                    break # Restart loop with new S
            else:
                # If compatible, just add it (should have been added in greedy phase, but maybe S changed)
                S_copy.append(x)
                S = S_copy
                improved = True
                break
                
    return S
