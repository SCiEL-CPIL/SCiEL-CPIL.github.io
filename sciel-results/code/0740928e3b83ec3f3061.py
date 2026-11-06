import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 180
    elements = list(range(1, N + 1))
    
    # 1. Precompute all forbidden triples
    # 1/a = 1/b + 1/c  => c = ab / (b-a)
    # We need distinct a, b, c.
    triples = []
    # elem_to_triples: for each element, list of indices of triples it belongs to
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    # Sort the triple for consistent representation? 
                    # The equation is 1/a = 1/b + 1/c. 
                    # a is the smallest. b and c are larger.
                    # We store (a, b, c) where a < b and a < c.
                    # Note: b and c are symmetric in the sum, but distinct from a.
                    # We must ensure a, b, c are distinct.
                    t = (a, b, c)
                    idx = len(triples)
                    triples.append(t)
                    for x in t:
                        elem_to_triples[x].append(idx)
    
    # For fast lookup, store triples as frozensets? Or just tuples.
    # We need to check if a triple is fully contained in a set.
    
    def can_add(S, x):
        """Check if x can be added to S without creating a forbidden triple."""
        for idx in elem_to_triples[x]:
            a, b, c = triples[idx]
            # Check if the other two elements are in S
            if x == a:
                if b in S and c in S:
                    return False
            elif x == b:
                if a in S and c in S:
                    return False
            elif x == c:
                if a in S and b in S:
                    return False
        return True

    def run_greedy(randomize_order=False):
        """Construct a set by greedily adding elements."""
        S = set()
        candidates = list(elements)
        if randomize_order:
            random.shuffle(candidates)
        
        for x in candidates:
            if can_add(S, x):
                S.add(x)
        return S

    def local_search(S, time_budget=10.0):
        """
        Local search with 1-for-1 and 1-for-2 swaps.
        1-for-1: Remove y, add x if it allows more additions later? 
                 Actually, just try to improve size.
                 1-for-1 maintains size. It's useful for escaping local optima 
                 where 1-for-0 is not possible.
                 But 1-for-2 increases size.
        """
        S = set(S)
        removed = set(elements) - S
        
        # Phase 1: Add any element that can be added directly
        for x in list(removed):
            if time.time() > start_time + time_budget:
                break
            if can_add(S, x):
                S.add(x)
                removed.remove(x)
        
        # Phase 2: 1-for-2 Swaps
        # Try to remove one element y from S and add two elements x1, x2 from removed
        # such that the new set is valid.
        improved = True
        while improved and time.time() < start_time + time_budget:
            improved = False
            removed = list(set(elements) - S)
            random.shuffle(removed)
            
            for y in list(S):
                if time.time() > start_time + time_budget:
                    break
                
                # Temporarily remove y
                S.remove(y)
                
                # Find up to 2 elements from removed that can be added
                added = []
                for x in removed:
                    if len(added) == 2:
                        break
                    if can_add(S, x):
                        S.add(x)
                        added.append(x)
                
                if len(added) == 2:
                    # We increased size by 1
                    improved = True
                    # Update removed set
                    for x in added:
                        if x in removed:
                            removed.remove(x)
                    # y is now in removed
                    removed.append(y)
                else:
                    # Restore S
                    for x in added:
                        S.remove(x)
                    S.add(y)
                    # If we added 1, size is same. 
                    # We might still want to keep it if it helps future moves?
                    # For simplicity, only keep if size increases (len(added)==2) 
                    # or if it enables immediate further additions (complex).
                    # Let's stick to size-increasing moves for now to ensure progress.
        
        # Phase 3: 1-for-1 Swaps to potentially unlock 1-for-0 or 1-for-2
        # This is heuristic. We try swaps that reduce the number of "conflicts" 
        # or just random swaps to escape local minima.
        # Given time constraints, we will do a few random 1-for-1 swaps 
        # followed by a greedy add pass.
        
        for _ in range(50): # Limited number of random walks
            if time.time() > start_time + time_budget:
                break
            
            S_copy = set(S)
            removed_copy = list(set(elements) - S)
            
            if not removed_copy:
                break
                
            random.shuffle(removed_copy)
            
            # Pick a random y from S and x from removed
            y = random.choice(list(S))
            x = random.choice(removed_copy)
            
            S.remove(y)
            if can_add(S, x):
                S.add(x)
                # Try to add more
                for z in list(set(elements) - S):
                    if can_add(S, z):
                        S.add(z)
                
                # If size didn't increase, revert to avoid wandering?
                # Or keep it? 
                # If len(S) > len(S_copy):
                #    improved = True
                # else:
                #    S = S_copy
                pass # Keep the change to allow exploration
            
            # If we didn't add x, restore
            if x not in S:
                S.add(y)

        return S

    # Strategy:
    # 1. Run several randomized greedy initializations.
    # 2. Apply local search to the best ones.
    
    best_set = set()
    best_size = 0
    
    # Deterministic greedy (smallest to largest)
    s = run_greedy(randomize_order=False)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        
    # Randomized greedy initializations
    # We have ~50s. 
    # Each greedy is fast. Local search is slower.
    # Let's do 20 initializations, then local search on top 5.
    
    candidates_for_ls = []
    
    for i in range(20):
        if time.time() > deadline - 15: # Reserve 15s for local search
            break
        s = run_greedy(randomize_order=True)
        candidates_for_ls.append(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Sort candidates by size, keep top 5
    candidates_for_ls.sort(key=lambda x: len(x), reverse=True)
    top_candidates = candidates_for_ls[:5]
    
    # Apply local search to each top candidate
    for s in top_candidates:
        if time.time() > deadline - 2:
            break
        # Allocate remaining time / num_remaining
        remaining = deadline - time.time() - 1
        budget = remaining / max(1, len(top_candidates))
        # Cap budget to avoid one taking too long
        budget = min(budget, 10.0)
        
        S = local_search(s, time_budget=budget)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Final local search on the absolute best with remaining time
    if time.time() < deadline - 1:
        remaining = deadline - time.time() - 1
        S = local_search(best_set, time_budget=remaining)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return list(best_set)

# EVOLVE-BLOCK-END
