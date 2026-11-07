def _try_add_to_set(arr, sum_to_card, active, x, deadline):
    """Try to add x to arr. Returns (success, new_sum_to_card, new_active)."""
    if time.time() > deadline:
        return False, None, None
        
    valid = True
    new_sums = []
    for s in active:
        ns = s + x
        nk = sum_to_card[s] + 1
        if sum_to_card[ns] >= 0 and sum_to_card[ns] != nk:
            valid = False
            break
        new_sums.append((ns, nk))
    
    if not valid:
        return False, None, None
        
    # Update sum_to_card
    new_sum_to_card = sum_to_card[:] # Copy is expensive, but safe. 
    # Better to modify in place if we pass a copy or handle carefully.
    # Since we are in a search loop, let's just check validity and return.
    # The caller will update if successful.
    
    return True, new_sums
