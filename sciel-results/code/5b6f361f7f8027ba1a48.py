"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 85.0  # 5s safety margin before 90s kill

def _is_valid_addition(selected, candidate):
    """
    Check if adding 'candidate' to 'selected' maintains validity.
    Condition: No selected integer (including candidate) may divide the sum 
    of any 3 DISTINCT OTHER selected integers.
    
    This check needs to verify:
    1. For the new candidate: Does candidate divide the sum of any 3 distinct 
       integers from 'selected'?
    2. For each existing integer 'a' in 'selected': Does 'a' divide the sum of 
       any 3 distinct integers from ('selected' - {a}) + {candidate}?
       This can be split into:
       a. 'a' divides sum of 3 from 'selected' - {a} (Already true if 'selected' was valid)
       b. 'a' divides sum of (2 from 'selected' - {a}) + candidate
       c. 'a' divides sum of (1 from 'selected' - {a}) + 2*candidate? No, distinct OTHER summands.
          Wait, the summands must be DISTINCT.
          So for existing 'a', the new sums involving 'candidate' are:
          - candidate + x + y where x,y are distinct from 'selected' and x!=y, x!=candidate, y!=candidate.
          Since candidate is new, x,y are from 'selected' \ {a}.
          - Is it possible to have 3 summands involving candidate? 
            No, because candidate is only one element. The other two must be from 'selected'.
            So we only need to check sums of form: candidate + x + y.
    """
    # Optimization: Precompute sums? 
    # Let's do a direct check but optimized.
    
    # 1. Check candidate against existing 'selected'
    # candidate divides x+y+z for distinct x,y,z in selected
    # If len(selected) < 3, this is impossible.
    if len(selected) >= 3:
        sel_list = list(selected)
        # We can optimize by checking modulo
        # For each triple in selected, check if sum % candidate == 0
        # This is O(N^3) which is slow if N is large. 
        # But N is max 48. 48^3 ~ 110k. 
        # We are doing this for each candidate. 
        # Total candidates 192. 192 * 110k ~ 21M ops. Might be okay in Python if optimized? 
        # Actually, we can do better.
        
        # Let's use a set for O(1) lookups? No, we need sums.
        # Let's just iterate.
        for x in sel_list:
            for y in sel_list:
                if y == x: continue
                for z in sel_list:
                    if z == x or z == y: continue
                    if (x + y + z) % candidate == 0:
                        return False
    else:
        # If len < 3, no triple exists, so candidate is safe from this perspective
        pass

    # 2. Check existing 'a' against new sums involving candidate
    # For each a in selected:
    #   Does a divide (candidate + x + y) for distinct x,y in selected \ {a}?
    for a in selected:
        # We need distinct x, y from selected such that x != a, y != a, x != y
        # And sum = candidate + x + y
        # Check if (candidate + x + y) % a == 0
        
        # Optimization: 
        # (candidate + x + y) % a == 0  =>  (x + y) % a == (-candidate) % a
        # Let target = (-candidate) % a
        # We need to find if there exist distinct x, y in selected \ {a} such that (x + y) % a == target
        
        target = (-candidate) % a
        
        # Iterate x in selected \ {a}
        # For each x, we need y in selected \ {a, x} such that (x + y) % a == target
        # => y % a == (target - x) % a
        
        # We can precompute a map of residues for selected \ {a}?
        # Since this is inside a loop, let's just do it directly but carefully.
        
        # To speed up, we can create a list of valid y's for each residue?
        # But 'a' changes.
        
        # Let's just loop.
        # selected is small (<=48).
        # Loop x in selected:
        #   if x == a: continue
        #   needed_residue = (target - x) % a
        #   Loop y in selected:
        #      if y == a or y == x: continue
        #      if y % a == needed_residue:
        #          return False
        # This is O(N^2) per 'a', so O(N^3) total.
        # N=48, 48^3 = 110k. 
        # This is similar to the first check.
        
        # We can optimize the inner loop by precomputing residues of selected elements.
        # But 'a' is the modulus, so residues change.
        
        # Let's try to optimize the first check (candidate vs selected) too.
        # Sum of 3 distinct from selected.
        # If we sort selected, we can maybe prune?
        
        # For now, let's implement the O(N^3) checks but with early exit.
        
        # Re-implementing check 1 (candidate vs selected) more efficiently?
        # Actually, I already did check 1 above. Let's refine it.
        # The above check 1 was O(N^3).
        
        # Let's refine check 2 here.
        for x in selected:
            if x == a:
                continue
            needed_residue = (target - x) % a
            for y in selected:
                if y == a or y == x:
                    continue
                if y % a == needed_residue:
                    return False
    return True

def _greedy_with_backtracking():
    """
    Uses a greedy approach but with a twist: 
    Instead of just adding the largest, it tries to add elements that are "least likely" to cause conflicts.
    Actually, let's stick to a standard greedy but with the optimized validity check.
    The previous seed was greedy from 192 down to 1.
    Let's try greedy from 192 down to 1, but using the optimized _is_valid_addition.
    """
    selected = []
    # Try adding from largest to smallest
    for candidate in range(192, 0, -1):
        if _is_valid_addition(selected, candidate):
            selected.append(candidate)
    return selected

def _local_search(initial_set):
    """
    Try to improve the set by swapping elements or adding/removing.
    Since we want to maximize cardinality, we primarily want to ADD elements.
    If we can't add, maybe we can swap a "bad" element for two "good" ones? 
    Or remove an element that blocks many others and add more?
    
    Simple local search:
    1. Try to add any unselected element.
    2. If stuck, try to swap one selected element with two unselected ones?
       Or just remove one selected and try to add two unselected.
    """
    selected = list(initial_set)
    unselected = [x for x in range(1, 193) if x not in selected]
    
    improved = True
    while improved and (time.time() - _START_TIME) < _DEADLINE:
        improved = False
        
        # Phase 1: Try to add
        for c in unselected:
            if _is_valid_addition(selected, c):
                selected.append(c)
                unselected.remove(c)
                improved = True
                break
        
        if improved:
            continue
            
        # Phase 2: Try to swap (remove 1, add 2)
        # This is expensive. Let's try a simpler perturbation:
        # Remove a random element, then try to greedily add back as many as possible.
        if len(selected) > 0:
            # Pick a random element to remove
            idx = random.randint(0, len(selected) - 1)
            removed = selected.pop(idx)
            unselected.append(removed)
            
            # Try to add back
            added_any = False
            for c in unselected[:]:
                if _is_valid_addition(selected, c):
                    selected.append(c)
                    unselected.remove(c)
                    added_any = True
                    # Don't break, try to add more
                    if (time.time() - _START_TIME) > _DEADLINE:
                        break
            
            # If we didn't end up with more elements, we might have lost one.
            # We should only keep this change if len(selected) >= len(initial_set) or we found a better config?
            # Actually, local search usually keeps the move if it doesn't decrease objective.
            # Here objective is cardinality.
            # If len(selected) < len(initial_set), we should revert?
            # Or maybe this perturbation allows us to add MORE later?
            # Let's just keep the new state and see if we can recover.
            # To be safe, let's only accept if len(selected) >= len(initial_set) is not strictly required 
            # if we are in a random walk, but for maximizing, we should prefer larger sets.
            # Let's revert if we got smaller, unless we are doing a simulated annealing style.
            # For simplicity, let's just keep the best found so far.
            
            if len(selected) < len(initial_set):
                # Revert
                selected.remove(removed)
                unselected.remove(removed)
                selected.append(removed)
                # Actually, the logic above modified selected and unselected.
                # Let's just restart the local search from the best known state if we degrade.
                # For now, let's just continue with the current state and hope it recovers.
                # Better: keep track of best_set.
        
    return selected

def construct():
    start_time = time.time()
    best_set = []
    
    # Run multiple random restarts or different orderings
    # 1. Greedy Descending
    # 2. Greedy Ascending
    # 3. Random Greedy
    
    strategies = [
        lambda: _greedy_with_backtracking(),
        lambda: _greedy_with_backtracking_asc(),
    ]
    
    # Add random restarts
    for i in range(20):
        if (time.time() - start_time) > _DEADLINE - 5:
            break
        # Random permutation of candidates
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        selected = []
        for c in candidates:
            if _is_valid_addition(selected, c):
                selected.append(c)
        strategies.append(lambda s=selected: s)
        
    for strat in strategies:
        if (time.time() - start_time) > _DEADLINE - 10:
            break
        try:
            current_set = strat()
            if len(current_set) > len(best_set):
                best_set = current_set
            # Try local search on this result
            if (time.time() - start_time) < _DEADLINE - 2:
                improved_set = _local_search(current_set)
                if len(improved_set) > len(best_set):
                    best_set = improved_set
        except Exception:
            pass
            
    # Final validation and cleanup
    # Ensure no duplicates and within range
    best_set = list(set(best_set))
    best_set = [x for x in best_set if 1 <= x <= 192]
    
    return best_set

def _greedy_with_backtracking_asc():
    selected = []
    for candidate in range(1, 193):
        if _is_valid_addition(selected, candidate):
            selected.append(candidate)
    return selected

# EVOLVE-BLOCK-END
