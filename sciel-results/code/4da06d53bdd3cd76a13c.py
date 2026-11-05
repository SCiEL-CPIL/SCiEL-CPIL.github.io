import time
from fractions import Fraction

def construct():
    """
    Constructs a subset of [1, 256] with no non-constant 3-term arithmetic progression
    that maximizes the sum of reciprocals.
    
    Strategy:
    1. Generate a base 3-AP-free set using a greedy approach (adding small numbers first).
    2. Use local search to improve the reciprocal sum by:
       - Trying to add missing numbers (if valid).
       - Trying to swap existing numbers with missing ones to gain reciprocal value.
       - Trying to remove a number if it allows adding multiple others with higher total reciprocal value.
    3. Use a deadline guard to ensure the program finishes within the time limit.
    """
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    def is_valid_addition(x, selected_set):
        """Check if x can be added to selected_set without forming a 3-AP."""
        for a in selected_set:
            # Case 1: a, x, b is AP => b = 2x - a
            b1 = 2 * x - a
            if b1 in selected_set and b1 != x:
                return False
            
            # Case 2: a, b, x is AP => b = (a + x) / 2
            if (a + x) % 2 == 0:
                b2 = (a + x) // 2
                if b2 in selected_set and b2 != x and b2 != a:
                    return False
            
            # Case 3: b, a, x is AP => b = 2a - x
            b3 = 2 * a - x
            if b3 in selected_set and b3 != x and b3 != a:
                return False
        return True

    # Step 1: Greedy construction (add small numbers first to maximize reciprocal sum)
    selected = []
    selected_set = set()
    
    for x in range(1, n + 1):
        if is_valid_addition(x, selected_set):
            selected.append(x)
            selected_set.add(x)
    
    # Current sum
    current_sum = sum(Fraction(1, x) for x in selected)
    best_sum = current_sum
    best_set = set(selected)
    
    # Step 2: Local search
    # We'll try various improvements
    # To speed up, we precompute the set of "forbidden" pairs for each x
    # Actually, we can just check on the fly but optimize the loop
    
    # List of all numbers not in selected
    missing = [x for x in range(1, n + 1) if x not in selected_set]
    
    # Improvement 1: Try to add any missing number
    # Sort missing by 1/x (descending) to try the most valuable first
    missing_sorted = sorted(missing, key=lambda x: -1.0/x)
    
    # We'll do multiple passes
    # Since n is small (256), we can afford to be a bit exhaustive in local search
    
    # Let's try a more systematic local search
    # We'll maintain the set as a set for O(1) lookups
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        
        # 1. Try adding missing numbers
        for x in missing_sorted:
            if time.time() > deadline - 10.0:
                break
            if x in selected_set:
                continue
            if is_valid_addition(x, selected_set):
                # Check if adding x improves the sum (it always does since 1/x > 0)
                selected_set.add(x)
                selected.append(x)
                current_sum += Fraction(1, x)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_set = set(selected_set)
                improved = True
                # Remove from missing
                if x in missing:
                    missing.remove(x)
        
        # 2. Try swapping: remove y from selected, add x from missing
        # We want 1/x > 1/y, so x < y
        # But we need to check validity
        # This is more complex. Let's try removing a large y and adding a small x.
        
        # To make this efficient, we'll iterate over pairs (x, y) where x is missing, y is selected
        # and x < y.
        
        # We can limit the search to avoid O(n^2) being too slow if the set is large.
        # But n=256, so n^2 is ~65k, which is fine.
        
        # Let's try a few rounds of swapping
        # We'll pick the best swap if any exists
        
        best_swap_gain = Fraction(0)
        best_swap = None
        
        # To speed up, we can just check a subset or use heuristics
        # For now, let's check all pairs but break early if time is low
        
        for x in missing:
            if time.time() > deadline - 5.0:
                break
            # For each x, check if we can remove some y and add x
            # Actually, we need to check if x is valid *after* removing y.
            # This depends on y.
            
            # A simpler approach: just try to remove y and see if any x becomes valid.
            # Or, try to add x, and if it conflicts with y, check if removing y helps.
            
            # Let's stick to a simpler "add if possible" first, then "swap" later if needed.
            # The "add if possible" loop above already handles the easy wins.
            
            # For swaps, we need to be careful.
            # Let's try: for each y in selected, remove it, then check if any x in missing can be added.
            # If so, and 1/x > 1/y, do the swap.
            
            # This is O(|selected| * |missing| * cost_check).
            # cost_check is O(|selected|).
            # So O(n^3) in worst case, which is 256^3 ~ 16M. Might be too slow in Python.
            
            # Let's optimize by only considering y that are "large" (low reciprocal value)
            # and x that are "small" (high reciprocal value).
            
            pass
        
        # Let's implement a simpler swap logic
        # Iterate over y in selected (sorted by value, largest first)
        # For each y, temporarily remove it
        # Check if any x in missing (sorted by value, smallest first) can be added
        # If yes, and 1/x > 1/y, do the swap
        
        selected_list = sorted(selected_set, reverse=True)  # Largest first
        missing_list = sorted(missing)  # Smallest first
        
        for y in selected_list:
            if time.time() > deadline - 5.0:
                break
            
            # Temporarily remove y
            selected_set.remove(y)
            
            # Try to add some x
            found_swap = False
            for x in missing_list:
                if x >= y:  # We want 1/x > 1/y, so x < y
                    break
                if is_valid_addition(x, selected_set):
                    # Check if this swap improves the sum
                    # New sum = current_sum - 1/y + 1/x
                    # Since x < y, 1/x > 1/y, so it always improves
                    selected_set.add(x)
                    current_sum += Fraction(1, x) - Fraction(1, y)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = set(selected_set)
                    
                    # Update lists
                    selected_list.remove(y)
                    missing_list.remove(x)
                    # Re-sort? Or just update the set
                    selected_set = set(selected_list) | {x}
                    # Wait, I removed y and added x.
                    # Let's just update the set and lists properly.
                    
                    # Actually, let's just do the swap and mark improved
                    # The set is already updated (removed y, added x)
                    # But selected_list and missing_list are stale.
                    # Let's just break and restart the loop with updated lists.
                    
                    improved = True
                    found_swap = True
                    break
            
            if not found_swap:
                # Put y back
                selected_set.add(y)
            else:
                # y is already removed, x is added
                # We need to update selected_list and missing_list
                # This is getting messy. Let's simplify.
                
                # Just break out and let the outer loop handle the state.
                # But we need to ensure the state is consistent.
                
                # Let's restart the local search with the new state.
                break
            
        if found_swap:
            # Update selected and missing
            selected = list(selected_set)
            missing = [x for x in range(1, n + 1) if x not in selected_set]
            missing_sorted = sorted(missing, key=lambda x: -1.0/x)
            # Continue the while loop
    
    # Step 3: Final check
    # Ensure the best set is returned
    # The best_set might be different from the current selected_set if we didn't update it properly
    # Let's just return the set that gives the best_sum
    
    # But we need to be careful: the "best_set" might not be valid if we made a mistake in tracking.
    # Let's just return the current selected_set if it's valid, otherwise the best_set.
    
    # Actually, the current selected_set is always valid by construction.
    # And we update best_sum whenever current_sum improves.
    # So if current_sum == best_sum, selected_set is a valid set with the best sum.
    # If current_sum < best_sum, it means we found a better set earlier but then made a move that reduced the sum?
    # No, we only make moves that improve the sum (adding x always improves, swapping x for y with x<y always improves).
    # So current_sum should always be >= best_sum.
    # Wait, in the swap logic, I did `current_sum += Fraction(1, x) - Fraction(1, y)`.
    # Since x < y, this is positive. So current_sum increases.
    # So current_sum should always be the best sum found so far.
    
    # Therefore, we can just return the current selected_set.
    
    return list(selected_set)

# EVOLVE-BLOCK-START
# The code above is the main logic.
# I need to make sure the swap logic is correct and efficient.
# The current implementation of the swap loop is a bit messy with the list updates.
# Let's refine it.
# EVOLVE-BLOCK-END
