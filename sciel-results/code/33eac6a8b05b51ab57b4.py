import time
import math

def construct():
    start_time = time.time()
    limit = 55.0 # Safety margin
    
    N = 256
    odds = [i for i in range(1, N + 1) if i % 2 != 0]
    evens = [i for i in range(1, N + 1) if i % 2 == 0]
    
    # Initialize with all odds
    S = set(odds)
    
    # Precompute conflicts for evens against odds
    # conflict[e] = list of odds that conflict with e
    conflicts = {e: [] for e in evens}
    
    def is_conflict(a, b):
        # a, b distinct
        g = math.gcd(a, b)
        x = a // g
        y = b // g
        # Condition: (x+y) divides g
        if g % (x + y) == 0:
            return True
        return False

    for e in evens:
        for o in odds:
            if is_conflict(e, o):
                conflicts[e].append(o)
                
    # Greedy pass 1: Add evens with 0 conflicts
    # Greedy pass 2: Add evens with 1 conflict (swap)
    # We should process evens in some order.
    # Maybe process those with fewer conflicts first?
    # Sort evens by number of conflicts with the CURRENT set S.
    # But S changes.
    
    # Simple approach: Iterate through evens, try to add.
    # To maximize, we should add those with 0 conflicts first.
    # Then those with 1 conflict.
    
    # Pass 1: 0 conflicts
    for e in evens:
        if time.time() - start_time > limit:
            break
        # Check conflicts with current S
        c = [o for o in S if is_conflict(e, o)]
        if len(c) == 0:
            S.add(e)
            
    # Pass 2: 1 conflict
    for e in evens:
        if time.time() - start_time > limit:
            break
        if e in S:
            continue
        c = [o for o in S if is_conflict(e, o)]
        if len(c) == 1:
            # Swap
            S.remove(c[0])
            S.add(e)
            
    # Pass 3: Try to add remaining evens if we can remove >1 conflicts?
    # This reduces size, so generally bad.
    # However, if the removed odds were "expensive" (conflicting with many other potential evens),
    # it might be good.
    # Given time, let's stick to non-decreasing size moves.
    
    # Local Search: Try to swap an even in S with an even not in S?
    # Or swap an odd in S with an even not in S?
    # We already did swaps for 1 conflict.
    
    # Let's do a few iterations of local search to see if we can improve.
    # Try to add an even e not in S.
    # If it conflicts with k odds in S.
    # If k == 0, add.
    # If k == 1, swap.
    # We did this.
    
    # What if we remove an odd o from S, and then add multiple evens that conflicted with o?
    # If o conflicted with e1, e2, e3.
    # Remove o. Add e1, e2, e3.
    # Net change: -1 + 3 = +2.
    # This is a promising move!
    
    # Let's implement a search for such moves.
    # For each odd o in S:
    #   Count how many evens e (not in S) conflict ONLY with o (among odds in S)?
    #   No, they might conflict with other odds too.
    #   If we remove o, the conflicts with o are gone.
    #   Let E_o be the set of evens not in S that conflict with o.
    #   If we remove o, we can potentially add evens in E_o that have no OTHER conflicts in S\{o}.
    #   Let's check for each o in S (odds only):
    #       potential_gains = []
    #       for e in evens not in S:
    #           if is_conflict(o, e):
    #               # Check if e has any other conflict in S - {o}
    #               other_conflicts = [x for x in S if x != o and is_conflict(e, x)]
    #               if len(other_conflicts) == 0:
    #                   potential_gains.append(e)
    #       if len(potential_gains) > 1:
    #           # Remove o, add all potential_gains
    #           S.remove(o)
    #           for e in potential_gains:
    #               S.add(e)
    #           # This increases size by len(potential_gains) - 1
    #           # If len > 1, size increases.
            
    # This is O(N^3) roughly?
    # N=256. Odds ~128. Evens ~128.
    # For each odd (128), for each even (128), check conflicts with S (256).
    # 128 * 128 * 256 ~ 4M operations. Fast enough.
    
    improved = True
    while improved and (time.time() - start_time < limit):
        improved = False
        # Make a copy to iterate over
        current_odds_in_S = [x for x in S if x % 2 != 0]
        for o in current_odds_in_S:
            if time.time() - start_time > limit:
                break
            potential_gains = []
            for e in evens:
                if e in S:
                    continue
                if is_conflict(o, e):
                    # Check other conflicts in S
                    other_conflicts = []
                    for x in S:
                        if x == o:
                            continue
                        if is_conflict(e, x):
                            other_conflicts.append(x)
                            break
                    if len(other_conflicts) == 0:
                        potential_gains.append(e)
            
            if len(potential_gains) > 1:
                S.remove(o)
                for e in potential_gains:
                    S.add(e)
                improved = True
                # Restart loop as S changed significantly
                break
                
    return list(S)
