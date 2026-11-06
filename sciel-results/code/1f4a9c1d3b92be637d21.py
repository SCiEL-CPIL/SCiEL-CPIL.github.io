import time
import random

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Generate divisors > 1
    divisors = [d for d in range(3, N + 1, 2) if N % d == 0]
    
    # Precompute masks for each (m, r)
    # masks[m][r] is an integer bitset where bit i is 1 if i % m == r
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            mask = 0
            # Set bits at r, r+m, r+2m, ...
            for x in range(r, N, m):
                mask |= (1 << x)
            masks[m][r] = mask
            
    full_mask = (1 << N) - 1
    
    # State
    selected = {} # m -> r
    available = set(divisors)
    covered_mask = 0
    
    def get_uncovered_count():
        return N - covered_mask.bit_count()
        
    def find_best_residue(m, current_covered):
        """Find r that maximizes gain for modulus m"""
        best_r = 0
        best_gain = -1
        inv_covered = ~current_covered & full_mask
        for r in range(m):
            gain = (masks[m][r] & inv_covered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_r = r
        return best_r, best_gain

    # 1. Greedy Initialization
    while available:
        if time.time() > deadline:
            break
            
        best_m = None
        best_r = None
        best_gain = 0
        
        for m in available:
            r, gain = find_best_residue(m, covered_mask)
            if gain > best_gain:
                best_gain = gain
                best_m = m
                best_r = r
                
        if best_m is None or best_gain == 0:
            break
            
        selected[best_m] = best_r
        covered_mask |= masks[best_m][best_r]
        available.remove(best_m)
        
        if get_uncovered_count() == 0:
            break

    # 2. Local Search
    # We try to improve by:
    # a) Changing residue of a selected modulus
    # b) Swapping a selected modulus with an unselected one
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_uncovered = get_uncovered_count()
        
        # a) Try changing residue for each selected modulus
        for m in list(selected.keys()):
            if time.time() > deadline:
                break
            old_r = selected[m]
            old_mask = masks[m][old_r]
            
            # Temporarily remove
            covered_mask_without = covered_mask & ~old_mask
            # Note: This is not strictly "removing" if other classes cover those bits.
            # We need to check if any bit in old_mask is NOT covered by other classes.
            # Actually, it's easier to just calculate the new covered mask if we change r.
            
            best_new_r = old_r
            best_new_uncovered = current_uncovered
            
            inv_covered_base = ~covered_mask & full_mask
            
            for new_r in range(m):
                if new_r == old_r:
                    continue
                new_mask = masks[m][new_r]
                # New covered mask = (covered_mask without old_mask) OR new_mask
                # But "covered_mask without old_mask" is tricky if overlaps exist.
                # Let's compute:
                # Covered by others = covered_mask & ~old_mask
                # New Covered = Covered_by_others | new_mask
                # This is O(1) bitwise ops.
                
                covered_by_others = covered_mask & ~old_mask
                new_covered = covered_by_others | new_mask
                new_uncovered = N - new_covered.bit_count()
                
                if new_uncovered < best_new_uncovered:
                    best_new_uncovered = new_uncovered
                    best_new_r = new_r
            
            if best_new_r != old_r:
                # Apply change
                old_mask = masks[m][old_r]
                new_mask = masks[m][best_new_r]
                # Update covered_mask
                # Remove old contribution that isn't covered by others
                # Actually, simply:
                # covered_mask = (covered_mask & ~old_mask) | new_mask
                # This works because if a bit was covered by old AND others, 
                # covered_mask & ~old_mask keeps it (if others cover it? No).
                # Wait. If bit is in old_mask AND other_mask, covered_mask has it.
                # covered_mask & ~old_mask clears it.
                # Then | new_mask. If new_mask doesn't have it, and other_mask does, we lost it?
                # No, `covered_mask` is the union of ALL selected masks.
                # If we remove old_r and add new_r, the new union is:
                # (Union of all other masks) | new_mask.
                # We don't have "Union of all other masks" stored directly, but we can derive it?
                # No, we can't easily get "Union of all other masks" without recomputing from scratch or maintaining counts.
                
                # Simpler approach for bitsets:
                # Just recompute covered_mask from scratch?
                # 47 moduli. Recomputing is O(47) big int ORs. Very fast.
                
                selected[m] = best_new_r
                # Recompute covered_mask
                cm = 0
                for mm, rr in selected.items():
                    cm |= masks[mm][rr]
                covered_mask = cm
                improved = True
                current_uncovered = get_uncovered_count()
                if current_uncovered == 0:
                    return [[m, r] for m, r in selected.items()]

        # b) Try swapping selected m_out with unselected m_in
        selected_list = list(selected.keys())
        for m_out in selected_list:
            if time.time() > deadline:
                break
            
            old_r_out = selected[m_out]
            old_mask_out = masks[m_out][old_r_out]
            
            # Compute covered mask without m_out
            covered_without_out = 0
            for mm, rr in selected.items():
                if mm != m_out:
                    covered_without_out |= masks[mm][rr]
            
            for m_in in available:
                if time.time() > deadline:
                    break
                
                # Find best r for m_in given covered_without_out
                r_in, gain_in = find_best_residue(m_in, covered_without_out)
                
                if gain_in > 0:
                    # Calculate new uncovered
                    new_covered = covered_without_out | masks[m_in][r_in]
                    new_uncovered = N - new_covered.bit_count()
                    
                    if new_uncovered < current_uncovered:
                        # Perform swap
                        del selected[m_out]
                        available.remove(m_in)
                        available.add(m_out)
                        selected[m_in] = r_in
                        
                        # Recompute covered_mask
                        covered_mask = 0
                        for mm, rr in selected.items():
                            covered_mask |= masks[mm][rr]
                        
                        improved = True
                        current_uncovered = new_uncovered
                        if current_uncovered == 0:
                            return [[m, r] for m, r in selected.items()]
                        break # Restart swap loop for this m_out? Or just continue?
                        # Continuing is fine, but the state changed.
                        # To be safe, break inner and restart outer? 
                        # Or just mark improved and let the while loop restart.
                        # Since we modified `selected`, the `selected_list` is stale.
                        # Better to break out of loops and restart the `while improved` loop.
                        break
            else:
                continue
            break # Break out of m_out loop to restart while loop

    return [[m, r] for m, r in selected.items()]
