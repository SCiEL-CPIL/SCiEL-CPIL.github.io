import time
import random
from itertools import combinations

START_TIME = time.time()
DEADLINE = 55.0  # seconds (safety margin from 60s)

def elapsed():
    return time.time() - START_TIME

def is_valid_set(sums_by_card):
    """Check that no sum appears in two different cardinality sets."""
    seen = {}
    for k, s in sums_by_card.items():
        for val in s:
            if val in seen and seen[val] != k:
                return False
            seen[val] = k
    return True

def build_sums_by_card(elements):
    """Build dictionary: cardinality -> set of achievable sums."""
    n = len(elements)
    sums_by_card = {k: set() for k in range(n + 1)}
    sums_by_card[0].add(0)
    
    for i, elem in enumerate(elements):
        # Update in reverse to avoid using the same element twice
        for k in range(i, -1, -1):
            for s in list(sums_by_card[k]):
                sums_by_card[k + 1].add(s + elem)
    return sums_by_card

def check_collision(sums_by_card, new_sums_by_card):
    """Check if new_sums_by_card introduces any cross-cardinality collision."""
    # We only need to check if any sum in new_sums_by_card[k] is in new_sums_by_card[j] for j != k
    # And since we only added one element, the new sums are in new_sums_by_card.
    # The old sums were valid.
    # So we just need to check if any sum in new_sums_by_card is in another cardinality's set.
    # Actually, we can just check the full validity of new_sums_by_card.
    seen = {}
    for k, s in new_sums_by_card.items():
        for val in s:
            if val in seen and seen[val] != k:
                return True
            seen[val] = k
    return False

def add_element_incremental(sums_by_card, elem):
    """
    Add an element to the existing sums_by_card structure.
    Returns a new sums_by_card dict.
    """
    n = len(sums_by_card) - 1
    new_sums = {k: set() for k in range(n + 2)}
    
    # Copy existing sums
    for k in range(n + 1):
        new_sums[k] = sums_by_card[k].copy()
        
    # Add new sums involving elem
    # Iterate k from n down to 0
    for k in range(n, -1, -1):
        for s in sums_by_card[k]:
            new_sums[k + 1].add(s + elem)
            
    return new_sums

def backtracking_search_incremental(elements, sums_by_card, target_size, max_value=256):
    """
    Backtracking search with incremental state.
    """
    if len(elements) >= target_size:
        return elements[:]
    
    if elapsed() > DEADLINE:
        return None
        
    used = set(elements)
    candidates = [v for v in range(1, max_value + 1) if v not in used]
    
    # Heuristic: Sort candidates.
    # We want to avoid collisions.
    # Let's score by how many new sums they create that are "close" to existing sums of other cardinalities.
    # Or simply try them and see.
    
    # To speed up, we can filter candidates that definitely cause collisions?
    # Hard to do without checking.
    
    # Let's sort by value. 
    # Strategy: Try large values first? Or small?
    # Large values push sums up, potentially avoiding lower cardinality sums.
    # The base set {226..256} uses large values.
    # So let's try large values first.
    
    # However, we need to ensure we can still fill the set.
    # If we pick too many large values, we might run out of small values to satisfy min-sum constraints?
    # Actually, the constraint is just disjointness.
    
    # Let's try a randomized restart approach with incremental state.
    
    # Score candidates: 
    # A candidate v is good if it creates few collisions.
    # We can estimate this by checking if v + s (for s in sums[k]) collides with sums[j].
    
    scored_candidates = []
    for v in candidates:
        # Quick check: 
        # If we add v, new sums are v + s_k.
        # Collision if v + s_k == s_j for j != k+1.
        # i.e., v == s_j - s_k.
        # We can precompute differences? Too expensive.
        
        # Just do the incremental update and check.
        new_sums = add_element_incremental(sums_by_card, v)
        
        # Check validity
        seen = {}
        valid = True
        for k, s in new_sums.items():
            for val in s:
                if val in seen and seen[val] != k:
                    valid = False
                    break
                seen[val] = k
            if not valid:
                break
        
        if valid:
            # Prefer larger v to keep sums high?
            # Or smaller v to keep sums low?
            # The base set is high. Let's stick with high.
            scored_candidates.append((v, new_sums))
            
    # Sort by v descending (larger first)
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    for v, new_sums in scored_candidates:
        if elapsed() > DEADLINE:
            return None
        result = backtracking_search_incremental(elements + [v], new_sums, target_size, max_value)
        if result is not None:
            return result
            
    return None

def construct():
    """Main entry point: construct the largest valid set possible."""
    if elapsed() > DEADLINE:
        return []
    
    # Start with the known valid construction of size 31
    base = list(range(226, 257))
    best = base
    
    # Verify base
    base_sums = build_sums_by_card(base)
    if not is_valid_set(base_sums):
        best = []
    
    # Try to extend using incremental backtracking
    # We will try to find a set of size 32, 33, etc.
    # Start from the base set.
    
    current_elements = base[:]
    current_sums = base_sums
    
    # Try to add one more element
    # We can use the backtracking search starting from the base state.
    # But backtracking_search_incremental expects to build up to target_size.
    # We can call it with target_size = len(base) + 1.
    
    # To increase the chance of success, we can try different starting points or heuristics.
    # The current heuristic in backtracking_search_incremental tries large values first.
    
    # Let's try to extend by 1, then 2, etc.
    # If we fail at 32, we might try a different base or a full search from scratch for 32.
    
    # First, try extending the base.
    result = backtracking_search_incremental(current_elements, current_sums, len(base) + 1, 256)
    if result is not None:
        best = result
        # Try to extend further
        best_sums = build_sums_by_card(best)
        for target in range(len(best) + 1, 40):
            if elapsed() > DEADLINE - 5:
                break
            res = backtracking_search_incremental(best, best_sums, target, 256)
            if res is not None:
                best = res
                best_sums = build_sums_by_card(best)
            else:
                break
    
    # If we didn't improve, try a search from scratch for size 32
    # This might find a different valid set of size 32 if the base extension failed.
    if len(best) < 32 and elapsed() < DEADLINE - 10:
        # Initialize empty state
        empty_sums = {0: {0}}
        res = backtracking_search_incremental([], empty_sums, 32, 256)
        if res is not None:
            best = res
            
    # Final check
    if best:
        if not is_valid_set(build_sums_by_card(best)):
            best = list(range(226, 257))
            
    return best
