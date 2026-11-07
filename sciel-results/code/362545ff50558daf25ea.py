def _greedy_array(candidates):
    MAX_SUM = 256 * 60 + 100
    card = [-1] * MAX_SUM
    card[0] = 0
    result = []
    
    for x in candidates:
        # Check if we can add x
        # We need to see if any new sum ns = s+x collides with an existing sum s_old of different cardinality.
        # Existing sums are in card.
        # New sums are s+x for all s where card[s] != -1.
        
        # Optimization: if x is too large, many sums will exceed MAX_SUM?
        # Max sum with current set is S_max. New max is S_max + x.
        # We just need to check indices.
        
        ok = True
        # We need to check all s such that card[s] != -1
        # And check if card[s+x] is -1 or card[s+x] == card[s]+1
        
        # Note: We cannot modify card while iterating.
        # So we just check.
        
        for s in range(MAX_SUM):
            if card[s] != -1:
                ns = s + x
                if ns >= MAX_SUM:
                    continue # Or handle out of bounds? 
                    # Actually if ns >= MAX_SUM, it's a new sum, no collision with existing (which are < MAX_SUM).
                    # But we should ensure we don't index out of bounds.
                    # If we assume MAX_SUM is large enough, this won't happen for valid sets.
                    # Let's just check bounds.
                if ns < MAX_SUM:
                    if card[ns] != -1 and card[ns] != card[s] + 1:
                        ok = False
                        break
                # If ns >= MAX_SUM, it's effectively "new" and doesn't collide with any existing sum < MAX_SUM.
                # But wait, if we have many numbers, sum can exceed MAX_SUM.
                # Let's increase MAX_SUM to 256*60 = 15360.
        
        if ok:
            # Update card
            for s in range(MAX_SUM):
                if card[s] != -1:
                    ns = s + x
                    if ns < MAX_SUM:
                        card[ns] = card[s] + 1
            result.append(x)
            
    return result
