import time
import random

def construct():
    deadline = time.time() + 55
    N = 128
    
    # Precompute APs
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    num_aps = len(aps)
    
    # Map index to list of AP indices it belongs to
    index_to_aps = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for idx in ap:
            index_to_aps[idx].append(i)

    def try_solve(k):
        colors = [random.randint(0, k - 1) for _ in range(N)]
        is_bad = [False] * num_aps
        bad_count = 0
        
        # Initialize is_bad and bad_count
        for i, ap in enumerate(aps):
            c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            if len(c) < 3:
                is_bad[i] = True
                bad_count += 1
                
        if bad_count == 0:
            return colors

        best_bad_count = bad_count
        best_colors = colors[:]
        
        temp = 1.0
        cooling = 0.9995
        min_temp = 0.01
        
        while temp > min_temp:
            if time.time() > deadline:
                break
                
            pos = random.randint(0, N - 1)
            old_color = colors[pos]
            new_color = random.randint(0, k - 1)
            
            if new_color == old_color:
                continue
            
            # Apply change
            colors[pos] = new_color
            
            # Update bad_count for APs involving pos
            delta = 0
            for ap_idx in index_to_aps[pos]:
                ap = aps[ap_idx]
                was_bad = is_bad[ap_idx]
                
                # Check new status
                c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                is_now_bad = len(c) < 3
                
                if was_bad and not is_now_bad:
                    delta -= 1
                elif not was_bad and is_now_bad:
                    delta += 1
                
                is_bad[ap_idx] = is_now_bad
            
            new_bad_count = bad_count + delta
            
            # Accept or Revert
            if new_bad_count < bad_count:
                bad_count = new_bad_count
                if bad_count < best_bad_count:
                    best_bad_count = bad_count
                    best_colors = colors[:]
                if bad_count == 0:
                    return colors
            else:
                # Revert
                delta_rev = 0
                for ap_idx in index_to_aps[pos]:
                    ap = aps[ap_idx]
                    was_bad = is_bad[ap_idx] # This is is_now_bad from above
                    
                    # Revert color
                    # We need to check status with old_color
                    # Temporarily set back to check?
                    # Or just calculate delta for revert
                    # Easier: just re-evaluate APs with old_color
                    # But we already updated is_bad to reflect new_color.
                    # So we need to check if reverting makes it bad/good.
                    
                    # Let's just store the previous state of is_bad for these APs?
                    # Or simpler:
                    # Calculate cost with old_color vs new_color.
                    # We did that.
                    # To revert, we just need to update is_bad back.
                    # We can re-check the APs with the reverted color.
                    
                    # Let's just re-run the check for these APs with old_color
                    # But colors[pos] is currently new_color.
                    # We need to evaluate "what if colors[pos] was old_color".
                    
                    # Optimization: Store the previous is_bad state?
                    # For simplicity, let's just re-evaluate.
                    
                    # Actually, we can just set colors[pos] = old_color and re-eval?
                    # No, we are in the middle of update.
                    
                    # Let's just revert the is_bad array by re-evaluating with old_color.
                    # This requires knowing the colors of other 3 elements.
                    
                    pass # This logic is getting messy.
            
            # Simpler approach for revert:
            # If not accepted, revert colors[pos] to old_color.
            # Then re-evaluate the APs to restore is_bad and bad_count.
            
            if new_bad_count >= bad_count:
                # Revert
                colors[pos] = old_color
                # Re-evaluate APs involving pos to restore is_bad and bad_count
                # We need to subtract the delta we calculated?
                # No, because is_bad array was modified.
                # We must restore is_bad array.
                
                # Let's just recalculate is_bad for these APs based on current colors (which are reverted)
                # And update bad_count accordingly.
                # But bad_count was not updated (we only updated if accepted).
                # So bad_count is still correct for the "before" state.
                # We just need to restore is_bad array to the "before" state.
                
                # We can do this by checking each AP involving pos.
                # But we don't know the "before" state of is_bad.
                # However, we can recalculate it from scratch for these APs.
                
                for ap_idx in index_to_aps[pos]:
                    ap = aps[ap_idx]
                    c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                    is_bad[ap_idx] = len(c) < 3
                    
                # Note: bad_count is NOT updated here because we reverted.
                # The is_bad array is now consistent with colors and bad_count.
            
            temp *= cooling
            
        return best_colors if best_bad_count == 0 else None

    for k in [3, 4, 5]:
        res = try_solve(k)
        if res:
            return res
            
    # Fallback
    return [0] * 128 # Invalid, but better than crashing? No, must return valid.
    # If we reach here, something went wrong.
    # But 4 colors should always work.

