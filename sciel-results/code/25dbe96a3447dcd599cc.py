import random
import time

def construct():
    N = 256
    # Initialize with a greedy set (similar to seed but let's make it robust)
    # We'll use a standard greedy: sort by 1/x descending? No, seed sorts by x ascending.
    # Let's start with the seed's result as a baseline, then improve.
    
    # Replicate seed logic to get initial S
    S = set()
    for x in range(1, N + 1):
        conflict = False
        # Check if x forms 3-AP with S
        # x is largest: a, b, x => 2b-x=a
        for b in S:
            if b >= x: continue
            a = 2 * b - x
            if a > 0 and a in S:
                conflict = True
                break
        if not conflict:
            S.add(x)
            
    # Convert to list for faster iteration in some contexts? Set is fine for membership.
    # We will use a boolean array for O(1) membership checks in the improvement loop.
    in_set = [False] * (N + 1)
    for x in S:
        in_set[x] = True
        
    def is_valid_addition(x, in_set, S):
        # Check if adding x to S keeps it 3-AP free
        # 1. x is middle: x-d, x+d in S
        for d in range(1, min(x, N - x) + 1):
            if in_set[x - d] and in_set[x + d]:
                return False
        # 2. x is largest: a, b, x => a = 2b - x
        for b in S:
            if b >= x: continue
            a = 2 * b - x
            if a > 0 and a < b and in_set[a]:
                return False
        # 3. x is smallest: x, b, c => c = 2b - x
        for b in S:
            if b <= x: continue
            c = 2 * b - x
            if c > x and c < b and in_set[c]:
                return False
        return True

    def get_score(S):
        return sum(1.0 / x for x in S)

    best_S = set(S)
    best_score = get_score(best_S)
    
    start_time = time.time()
    timeout = 55.0 # Leave 5s margin
    
    # Hill Climbing
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        # Try all possible moves: Add, Remove, Swap
        # To be fast, we can iterate through candidates
        
        # 1. Try adding any x not in S
        for x in range(1, N + 1):
            if in_set[x]:
                continue
            if is_valid_addition(x, in_set, best_S):
                # Calculate gain
                gain = 1.0 / x
                if gain > 0: # Always positive
                    # Check if this move improves total score? 
                    # Wait, adding x always increases score. 
                    # So if it's valid, we should take it? 
                    # Not necessarily, if removing a high value allows adding two lower values?
                    # But hill climbing takes best single move.
                    # Let's just collect all valid moves and pick the best delta.
                    pass
                    
        # Let's do a proper best-improvement hill climbing step.
        best_move = None
        best_delta = 0
        
        # Candidate additions
        for x in range(1, N + 1):
            if in_set[x]:
                continue
            if is_valid_addition(x, in_set, best_S):
                delta = 1.0 / x
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('add', x, None)
                    
        # Candidate removals (only useful if it enables a larger addition, 
        # but simple hill climbing usually only adds if valid. 
        # However, sometimes you must remove to add. 
        # Let's consider swaps: Remove y, Add x.
        # This is O(N^2) which is 256^2 = 65536 checks per step. 
        # Each check is O(N) or O(N^2) worst case? 
        # is_valid_addition is O(N) for 'largest' and 'smallest' checks.
        # 65536 * 256 * 256 operations is too slow for 55s in Python.
        
        # Optimization: 
        # 1. Greedy Add: Just add any valid x. Since 1/x > 0, any valid add is an improvement.
        #    So we can just add the one with largest 1/x (smallest x) that is valid?
        #    Or just the first valid one? 
        #    Actually, if we can add x, score increases. 
        #    So the only reason not to add is if it blocks a better future combination.
        #    But local search handles that via swaps later.
        #    So Step 1: Add best valid x.
        
        best_add_x = None
        best_add_val = -1
        for x in range(1, N + 1):
            if in_set[x]:
                continue
            if is_valid_addition(x, in_set, best_S):
                val = 1.0 / x
                if val > best_add_val:
                    best_add_val = val
                    best_add_x = x
                    
        if best_add_x is not None:
            best_S.add(best_add_x)
            in_set[best_add_x] = True
            improved = True
            # Continue to next iteration to see if more adds are possible
            # Actually, we can keep adding until no more valid adds exist.
            # This is just a greedy phase.
            # Let's loop until no add is possible.
            continue
            
        # If no add is possible, try swaps.
        # Swap: Remove y from S, Add x not in S.
        # Condition: S \ {y} is 3-AP free (it is, subset of 3-AP free set).
        # Condition: (S \ {y}) U {x} is 3-AP free.
        # Gain: 1/x - 1/y.
        # We want max gain > 0.
        
        # To speed up swap checking:
        # Precompute for each y in S, which x's become valid if y is removed?
        # This is complex. 
        # Given N=256, N^2 = 65k pairs. 
        # For each pair (x, y), check if (S - {y} + {x}) is valid.
        # Validity check: 
        #   x must not form AP with S \ {y}.
        #   This means x does not form AP with any pair in S \ {y}.
        #   Previously x might have formed AP with y.
        
        # Let's implement a fast validity check for a specific set state.
        # We can cache "conflicts" for x?
        
        # Simpler approach for swaps:
        # For each x not in S:
        #   Find all y in S such that x conflicts with y (i.e., removing y might make x valid).
        #   If no conflict, x is already valid (handled above).
        #   So x conflicts with at least one y.
        #   Let C_x be the set of y in S that conflict with x.
        #   If we remove y in C_x, is x valid?
        #   We need to check if x conflicts with any z in S \ {y}.
        #   If C_x == {y}, then removing y makes x valid (assuming no other conflicts).
        #   If |C_x| > 1, we might need to remove multiple, which is not a simple swap.
        
        # So, only consider swaps where x conflicts with exactly one element y in S.
        
        # Step 1: For each x not in S, find conflicts.
        potential_swaps = []
        
        for x in range(1, N + 1):
            if in_set[x]:
                continue
            # Find conflicts
            conflicts = set()
            
            # x is middle
            for d in range(1, min(x, N - x) + 1):
                if in_set[x - d] and in_set[x + d]:
                    conflicts.add(x - d)
                    conflicts.add(x + d)
                    
            # x is largest
            for b in best_S:
                if b >= x: continue
                a = 2 * b - x
                if a > 0 and a < b and in_set[a]:
                    conflicts.add(a)
                    conflicts.add(b)
                    
            # x is smallest
            for b in best_S:
                if b <= x: continue
                c = 2 * b - x
                if c > x and c < b and in_set[c]:
                    conflicts.add(b)
                    conflicts.add(c)
                    
            if len(conflicts) == 1:
                y = next(iter(conflicts))
                # Check if removing y resolves all conflicts for x
                # We need to verify x is valid in S \ {y}
                # This requires a full check, but we can optimize:
                # We know x only conflicted with y.
                # So if we remove y, x should be valid, UNLESS there are other conflicts we missed?
                # No, we found ALL conflicts. If the set of conflicting elements is {y}, then removing y leaves no conflicting elements.
                # So x becomes valid.
                
                gain = 1.0 / x - 1.0 / y
                if gain > best_delta:
                    best_delta = gain
                    best_move = ('swap', x, y)
                    
        if best_move is not None:
            if best_move[0] == 'swap':
                _, x, y = best_move
                best_S.remove(y)
                in_set[y] = False
                best_S.add(x)
                in_set[x] = True
            improved = True
            continue
            
        # If no add and no simple swap, try 2-swap? Or stop?
        # For now, stop.
        
    return list(best_S)
