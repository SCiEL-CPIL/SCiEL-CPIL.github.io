import time
import random
from fractions import Fraction

def construct():
    N = 256
    # Precompute 3-APs
    # aps_for[y] will be a list of tuples (u, v) such that y, u, v form a 3-AP
    aps_for = [[] for _ in range(N + 1)]
    
    for b in range(2, N):
        min_a = max(1, 2*b - N)
        for a in range(min_a, b):
            c = 2*b - a
            # AP is (a, b, c)
            # For a: others are b, c
            aps_for[a].append((b, c))
            # For b: others are a, c
            aps_for[b].append((a, c))
            # For c: others are a, b
            aps_for[c].append((a, b))

    S = set()
    in_set = [False] * (N + 1)
    
    def can_add(y):
        for u, v in aps_for[y]:
            if in_set[u] and in_set[v]:
                return False
        return True

    # Greedy Initialization
    for x in range(1, N + 1):
        if can_add(x):
            S.add(x)
            in_set[x] = True
            
    best_S = set(S)
    best_score = sum(Fraction(1, x) for x in S)
    
    start_time = time.time()
    timeout = 85.0
    
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        
        # Phase 1: Add
        # Try to add smallest y first
        for y in range(1, N + 1):
            if not in_set[y]:
                if can_add(y):
                    S.add(y)
                    in_set[y] = True
                    improved = True
                    break # Restart loop to re-evaluate other additions
                
        if improved:
            # Update best if score improved (it always does on add)
            current_score = sum(Fraction(1, x) for x in S)
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
            continue # Restart loop

        # Phase 2: Swap
        # Try to swap y (not in S) with x (in S) such that y < x
        # We iterate y from 1 upwards to prioritize high gain
        swapped = False
        for y in range(1, N + 1):
            if in_set[y]:
                continue
            
            # Find conflicts
            # A conflict is a pair (u, v) from aps_for[y] such that in_set[u] and in_set[v]
            # We need to find an x that is in ALL conflicting pairs.
            # If there are no conflicts, can_add(y) would have been True.
            
            # Optimization: If can_add(y) is true, we missed it in Phase 1? 
            # No, Phase 1 checks all y. If we are here, can_add(y) is False.
            
            # Let's find the intersection of elements in conflicting APs.
            # Initialize with the set of elements from the first conflict.
            # If no conflicts found, something is wrong, but can_add is false, so there must be.
            
            candidate_xs = None
            has_conflict = False
            
            for u, v in aps_for[y]:
                if in_set[u] and in_set[v]:
                    if not has_conflict:
                        candidate_xs = {u, v}
                        has_conflict = True
                    else:
                        candidate_xs = candidate_xs & {u, v}
                        if not candidate_xs:
                            break # No single element can resolve all conflicts
                
            if not has_conflict:
                # Should not happen if can_add is false
                continue
                
            if not candidate_xs:
                continue
                
            # Try to remove an x from candidate_xs
            # We want x > y to increase score.
            # We want the largest x to minimize loss? Or just any valid x?
            # Any x > y increases score.
            # Let's pick the largest x in candidate_xs that is in S and > y.
            # Actually, removing a larger x is "cheaper".
            # So sort candidate_xs descending.
            
            best_x_to_remove = -1
            for x in candidate_xs:
                if in_set[x] and x > y:
                    if best_x_to_remove == -1 or x > best_x_to_remove:
                        best_x_to_remove = x
            
            if best_x_to_remove != -1:
                # Perform swap
                S.remove(best_x_to_remove)
                in_set[best_x_to_remove] = False
                S.add(y)
                in_set[y] = True
                improved = True
                swapped = True
                break # Restart loop after a swap

        if improved:
            current_score = sum(Fraction(1, x) for x in S)
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
                
    if not best_S:
        return [1]
    return list(best_S)
