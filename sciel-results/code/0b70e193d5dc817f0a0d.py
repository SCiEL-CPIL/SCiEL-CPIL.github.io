import math
import random

def construct():
    N = 180
    
    # 1. Generate all forbidden triples
    triples = set()
    for k in range(1, N + 1):
        # a = kxy, b = kx(x+y), c = ky(x+y)
        # max(a,b,c) <= N
        # c is usually the largest if x,y >= 1.
        # k * y * (x+y) <= N
        for y in range(1, N + 1):
            if k * y * (y + 1) > N: # min x is 1, so x+y >= y+1
                break
            for x in range(1, N + 1):
                if k * x * (x + y) > N: # b <= N
                    break
                if math.gcd(x, y) != 1:
                    continue
                a = k * x * y
                b = k * x * (x + y)
                c = k * y * (x + y)
                if a <= N and b <= N and c <= N:
                    if len({a, b, c}) == 3:
                        triples.add(frozenset((a, b, c)))
                        
    # Map number to list of triples containing it
    num_to_triples = {i: [] for i in range(1, N + 1)}
    for t in triples:
        for v in t:
            num_to_triples[v].append(t)
            
    # 2. Greedy Initialization
    S = set(range(1, N + 1))
    
    # Precompute conflict counts for speed?
    # We'll iterate.
    changed = True
    while changed:
        changed = False
        # Find any conflicting triple
        conflict = None
        for t in triples:
            if t.issubset(S):
                conflict = t
                break
        
        if conflict:
            # Choose element to remove
            # Heuristic: Remove the element with the highest number of conflicts in S
            # Count conflicts for each v in conflict
            counts = {v: 0 for v in conflict}
            for v in conflict:
                for t2 in num_to_triples[v]:
                    if t2.issubset(S):
                        counts[v] += 1
            v_to_remove = max(counts, key=counts.get)
            S.remove(v_to_remove)
            changed = True
            
    # 3. Local Search
    # Try to add back removed elements
    # And perform swaps to escape local optima
    
    # We'll run a loop for a fixed number of iterations or until no improvement
    # Since N is small, we can be thorough.
    
    # Sort removed elements to add back in a consistent order?
    # Or just iterate.
    
    # Let's define a helper to check if v can be added to S
    def can_add(v, current_S):
        for t in num_to_triples[v]:
            others = t - {v}
            if others.issubset(current_S):
                return False
        return True

    # Phase 1: Add back if possible
    added = True
    while added:
        added = False
        for v in range(1, N + 1):
            if v not in S and can_add(v, S):
                S.add(v)
                added = True
                
    # Phase 2: Swaps to improve structure
    # We try to swap u in S with v not in S if it leads to a valid set.
    # This doesn't increase size immediately, but might allow future additions.
    # We limit the number of swaps to avoid infinite loops.
    
    max_swaps = 500
    swaps_done = 0
    while swaps_done < max_swaps:
        made_swap = False
        # Shuffle to avoid bias
        candidates = [v for v in range(1, N + 1) if v not in S]
        random.shuffle(candidates)
        
        for v in candidates:
            if can_add(v, S):
                # Should have been added in Phase 1, but maybe S changed?
                # If Phase 1 ran to completion, this shouldn't happen unless we didn't run it to completion.
                # We did run it to completion. So this is redundant.
                continue
                
            # Find conflicts for v
            conflicts = []
            for t in num_to_triples[v]:
                others = t - {v}
                if others.issubset(S):
                    conflicts.append(others)
                    
            if not conflicts:
                continue
                
            # Try to find a u in S that is in all conflicts?
            # If we remove u, do all conflicts disappear?
            # A conflict is a pair {x, y}. If u is in {x, y}, removing u breaks that triple.
            # We need to remove a set of elements from S such that every conflict triple is broken.
            # Simplest: Remove one element from each conflict.
            # If we remove a single u that appears in all conflict pairs, we can swap u for v.
            
            # Find intersection of all conflict pairs
            # If intersection is not empty, pick one u from intersection.
            # If we remove u, all conflicts involving v are broken.
            # Then we add v. Size stays same.
            
            # Let's try this heuristic.
            # Get all elements in S that are part of conflicts with v.
            potential_removes = set()
            for pair in conflicts:
                potential_removes.update(pair)
                
            # For each u in potential_removes, check if removing u breaks ALL conflicts.
            # i.e., is u in every pair in `conflicts`?
            for u in potential_removes:
                all_broken = True
                for pair in conflicts:
                    if u not in pair:
                        all_broken = False
                        break
                if all_broken:
                    # Check if S - {u} + {v} is valid.
                    # We know conflicts with v are broken by removing u.
                    # But removing u might create new conflicts? No, removing elements never creates new conflicts.
                    # Adding v might create conflicts with other triples not involving the broken ones?
                    # We checked `can_add` initially and it failed because of `conflicts`.
                    # If we remove u, those specific conflicts are gone.
                    # Are there other conflicts for v?
                    # `conflicts` list contains ALL triples from `num_to_triples[v]` that are subsets of S.
                    # So if we break all of them, v can be added.
                    # So yes, S' = S - {u} + {v} is valid.
                    
                    # Perform swap
                    S.remove(u)
                    S.add(v)
                    made_swap = True
                    swaps_done += 1
                    break # Restart candidate loop? Or just break inner?
                    # Breaking inner means we try next v.
                    # But S changed, so previous checks are invalid.
                    # It's safer to break out and restart the whole swap phase iteration.
                    break
            
            if made_swap:
                break
                
        if not made_swap:
            break
            
    return sorted(list(S))
