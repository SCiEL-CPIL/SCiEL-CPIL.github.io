import random
from math import gcd
from itertools import combinations

def construct():
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute all forbidden triples
    # A triple (a, b, c) is forbidden if gcd(a,b) == gcd(b,c) == gcd(c,a)
    forbidden_triples = []
    triple_indices = {} # Map index in forbidden_triples to (a,b,c)
    
    # To speed up, we can just generate them on the fly or precompute.
    # N=128, C(128,3) ~ 340k. Precomputation is fast.
    for i in range(N):
        a = numbers[i]
        for j in range(i+1, N):
            b = numbers[j]
            g_ab = gcd(a, b)
            for k in range(j+1, N):
                c = numbers[k]
                g_bc = gcd(b, c)
                if g_ab == g_bc:
                    g_ca = gcd(c, a)
                    if g_ab == g_ca:
                        forbidden_triples.append((i, j, k))
                        
    # We will work with indices 0..127
    active = [True] * N
    current_set = set(range(N))
    
    # Precompute for each index, the list of triple IDs it belongs to
    # This allows us to quickly update counts when we remove an element.
    element_to_triples = {i: [] for i in range(N)}
    for t_idx, (a, b, c) in enumerate(forbidden_triples):
        element_to_triples[a].append(t_idx)
        element_to_triples[b].append(t_idx)
        element_to_triples[c].append(t_idx)
        
    # active_triples: set of indices of forbidden_triples that are still "active"
    # A triple is active if all 3 of its elements are in current_set.
    # Initially all are active.
    active_triples = set(range(len(forbidden_triples)))
    
    # We need to efficiently find a forbidden triple to break.
    # And we need to choose which element to remove.
    # Heuristic: Remove the element in the chosen triple that has the smallest "degree"
    # (number of active triples it is part of).
    
    # To make this faster, we can maintain a count of active triples for each element.
    active_count = [0] * N
    for t_idx in active_triples:
        a, b, c = forbidden_triples[t_idx]
        active_count[a] += 1
        active_count[b] += 1
        active_count[c] += 1
        
    while active_triples:
        # Pick an arbitrary active triple (or the one with highest "weight"?)
        # Let's pick the first one found for simplicity, or randomize to avoid bias.
        # Randomizing the choice of triple to break can help escape local minima.
        # But let's try deterministic first: pick the triple with the smallest sum of degrees?
        # Or just pick any.
        
        # To ensure we make progress, we just pick one active triple.
        # Using a set makes iteration slow if we need to pick "best".
        # Let's just pick the first element from the set.
        # For better performance, we might want to pick the triple that allows the most "savings".
        
        t_idx = next(iter(active_triples))
        a, b, c = forbidden_triples[t_idx]
        
        # We must remove one of a, b, c.
        # Choose the one with the minimum active_count.
        # This minimizes the number of other triples we "deactivate" (which is good? 
        # Actually, deactivating triples is GOOD because it reduces constraints.
        # Wait. If we remove x, we deactivate all triples containing x.
        # We want to deactivate as MANY bad triples as possible?
        # No, we want to remove the element that "costs" us the least in terms of future flexibility.
        # Usually, removing the element involved in the FEWEST other constraints is best,
        # because it breaks the current constraint while leaving other options open.
        # So, min active_count is the correct heuristic.
        
        candidates_to_remove = [a, b, c]
        # Tie-breaker: remove the largest number? Or smallest?
        # Let's use active_count, then index.
        best_to_remove = min(candidates_to_remove, key=lambda x: (active_count[x], x))
        
        # Remove best_to_remove from current_set
        current_set.remove(best_to_remove)
        active[best_to_remove] = False
        
        # Update active_triples and active_count
        # All triples containing best_to_remove are no longer active (because one element is gone)
        triples_to_remove = set()
        for t in element_to_triples[best_to_remove]:
            if t in active_triples:
                triples_to_remove.add(t)
                
        for t in triples_to_remove:
            active_triples.remove(t)
            a, b, c = forbidden_triples[t]
            # Decrement counts for the other two elements in this triple
            # (best_to_remove is already removed, so we don't care about its count anymore, 
            #  but we should be consistent)
            for elem in [a, b, c]:
                if elem != best_to_remove and active[elem]:
                    active_count[elem] -= 1
        # Decrement count for best_to_remove itself? Not strictly necessary as it's removed,
        # but for consistency if we were to inspect it (we won't).
        active_count[best_to_remove] = 0

    return sorted([numbers[i] for i in current_set])
