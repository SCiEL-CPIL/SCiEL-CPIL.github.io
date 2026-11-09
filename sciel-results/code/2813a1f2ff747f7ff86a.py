import time
import random

def construct():
    N = 256
    deadline = time.time() + 55.0  # Safety margin
    
    # Precompute all 3-term arithmetic progressions (a, b, c) with a < b < c
    # a + c = 2b
    aps = []
    for b in range(2, N):
        for a in range(1, b):
            c = 2 * b - a
            if c <= N:
                aps.append((a, b, c))
    
    # Helper to check if adding x to S is valid
    # S is a set of ints
    def can_add(S, x):
        for a, b, c in aps:
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

    # Helper to calculate score
    def score(S):
        return sum(1.0 / x for x in S)

    # Initial Greedy Solution
    S = set()
    for x in range(1, N + 1):
        if can_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_score = score(S)
    
    # Local Search
    # We will try to improve by removing one element and refilling greedily.
    # To make it faster, we can precompute for each x, which APs it is part of.
    # But N is small, so direct check is fine.
    
    # Optimization: Precompute neighbors?
    # For each x, list of APs containing x.
    ap_by_elem = {i: [] for i in range(1, N + 1)}
    for a, b, c in aps:
        ap_by_elem[a].append((a, b, c))
        ap_by_elem[b].append((a, b, c))
        ap_by_elem[c].append((a, b, c))

    def can_add_fast(S, x):
        for ap in ap_by_elem[x]:
            a, b, c = ap
            # Check if the other two are in S
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

    # Re-initialize S with fast check
    S = set()
    for x in range(1, N + 1):
        if can_add_fast(S, x):
            S.add(x)
    
    best_S = set(S)
    best_score = score(S)
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add any valid element
        # Since S is maximal in the previous iteration (or initially), we might have added some, 
        # but after a removal, we should refill.
        # Actually, the loop structure:
        # - If we just added an element, S is no longer maximal? No, we added one. 
        #   We should check if we can add more.
        
        # Refill Phase: Add all possible valid elements
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, N + 1):
                if x not in S and can_add_fast(S, x):
                    S.add(x)
                    changed = True
                    break # Restart scan to ensure smallest added first? 
                         # Or just continue? 
                         # Adding smallest first is good for weight.
                         # But if we add a small one, it might block a slightly larger one.
                         # Greedy by value is the heuristic.
        
        current_score = score(S)
        if current_score > best_score:
            best_score = current_score
            best_S = set(S)
        
        # If we didn't improve best, and we are stuck (maximal), try removals
        # Check if S is maximal
        is_maximal = True
        for x in range(1, N + 1):
            if x not in S and can_add_fast(S, x):
                is_maximal = False
                break
        
        if is_maximal:
            # Try removing each element
            # To speed up, we can prioritize removing elements with low 1/x (large x) 
            # or those involved in many conflicts?
            # Let's just iterate.
            elements_to_remove = list(S)
            random.shuffle(elements_to_remove)
            
            for x in elements_to_remove:
                if time.time() > deadline:
                    break
                
                S.remove(x)
                
                # Refill
                refill_changed = True
                while refill_changed and time.time() < deadline:
                    refill_changed = False
                    for y in range(1, N + 1):
                        if y not in S and can_add_fast(S, y):
                            S.add(y)
                            refill_changed = True
                            break
                
                new_score = score(S)
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_S = set(S)
                    improved = True
                    break # Restart the whole loop
                else:
                    # Restore
                    S.add(x)
                    # Note: Refilling after restore is not strictly necessary 
                    # if we assume the state was maximal before removal, 
                    # but removing x might have allowed additions which we 
                    # didn't take because we reverted. 
                    # Actually, we reverted S to S \ {x}. 
                    # The original S was maximal. 
                    # So S \ {x} is not necessarily maximal, but we don't need to refill 
                    # because we are just checking if removing x helps. 
                    # If it doesn't, we restore x and continue.
        
        if not improved:
            # No single removal helped. 
            # We can try a random restart or 2-swap.
            # For simplicity and safety, let's do a random restart with a bias towards small numbers.
            # Or just break if we've been stuck for a while?
            # Given time, let's do a few random restarts.
            # But we must ensure we return the best found so far.
            
            # Simple random restart:
            # Generate a new random valid set? 
            # Greedy with random tie-breaking?
            # Let's just do a greedy pass with some randomness in order?
            # No, greedy by value is strong.
            
            # Try 2-removal?
            # Might be too slow.
            
            # If we are stuck, we can try to "shake" the set.
            # Remove a random element, refill, accept if better.
            # If not better, try another.
            # If all single removals fail, try 2 removals.
            
            # To keep it simple and robust within time:
            # If no improvement, do a random restart of the greedy algorithm 
            # with a small probability of skipping a small number?
            # No, let's just do a random restart of the whole process 
            # but keep the best_S.
            
            # Actually, if no single removal improves, we are at a local optimum 
            # for 1-removal moves. 
            # We can try a "kick": remove 2 random elements, refill, accept if better.
            
            if len(S) > 2:
                # Pick 2 elements to remove
                # Prefer removing larger elements (lower weight)
                # Sort S by weight descending, pick from bottom half?
                sorted_S = sorted(S, key=lambda x: 1.0/x)
                # Pick 2 from the lower weight half
                half = len(sorted_S) // 2
                candidates = sorted_S[half:]
                if len(candidates) >= 2:
                    r1, r2 = random.sample(candidates, 2)
                    S.remove(r1)
                    S.remove(r2)
                    
                    refill_changed = True
                    while refill_changed and time.time() < deadline:
                        refill_changed = False
                        for y in range(1, N + 1):
                            if y not in S and can_add_fast(S, y):
                                S.add(y)
                                refill_changed = True
                                break
                    
                    new_score = score(S)
                    if new_score > current_score:
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_S = set(S)
                    else:
                        # Restore
                        S.add(r1)
                        S.add(r2)
                        # Refill to restore maximal state?
                        # Not strictly necessary if we just continue, 
                        # but good for consistency.
                        refill_changed = True
                        while refill_changed and time.time() < deadline:
                            refill_changed = False
                            for y in range(1, N + 1):
                                if y not in S and can_add_fast(S, y):
                                    S.add(y)
                                    refill_changed = True
                                    break
            else:
                # If S is very small, break?
                pass

    return list(best_S)
