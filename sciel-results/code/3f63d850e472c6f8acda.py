import time
import random
import sys

N = 256
DEADLINE = 75.0  # safety margin before 90s kill

def _build_structures():
    """
    Returns:
    four_aps: list of tuples (a, b, c, d)
    elem_4aps: list of lists, elem_4aps[x] = list of indices of 4-APs containing x
    four_ap_masks: list of int, mask for each 4-AP
    four_ap_others: list of dicts? No, for each 4-AP index, we need mask of others for each element in it.
                    Actually, for can_add(x), we need to check 4-APs containing x.
                    For a 4-AP ap containing x, we need the mask of the other 3 elements.
                    We can precompute: for each x, a list of (ap_idx, others_mask).
    
    three_aps: list of tuples (a, b, c)
    elem_3aps: list of lists, elem_3aps[x] = list of indices of 3-APs containing x
    three_ap_masks: list of int, mask for each 3-AP
    """
    four_aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_4aps = len(four_aps)
    four_ap_masks = [0] * num_4aps
    for i, ap in enumerate(four_aps):
        m = 0
        for x in ap:
            m |= (1 << x)
        four_ap_masks[i] = m
        
    # Precompute for can_add: for each x, list of (other_mask) for each 4-AP containing x
    # If mask & other_mask == other_mask, then adding x creates a 4-AP.
    elem_4ap_others = [[] for _ in range(N + 1)]
    for i, ap in enumerate(four_aps):
        full_mask = four_ap_masks[i]
        for x in ap:
            bit_x = 1 << x
            others_mask = full_mask ^ bit_x
            elem_4ap_others[x].append(others_mask)

    three_aps = []
    for a in range(1, N + 1):
        for c in range(a + 2, N + 1, 2):
            b = (a + c) // 2
            three_aps.append((a, b, c))
            
    num_3aps = len(three_aps)
    three_ap_masks = [0] * num_3aps
    for i, ap in enumerate(three_aps):
        m = 0
        for x in ap:
            m |= (1 << x)
        three_ap_masks[i] = m

    elem_3aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_3aps[x].append(i)

    return four_aps, elem_4ap_others, three_aps, three_ap_masks, elem_3aps

def construct():
    start_time = time.time()
    random.seed(42)
    
    four_aps, elem_4ap_others, three_aps, three_ap_masks, elem_3aps = _build_structures()
    
    def can_add(x, mask):
        """Check if adding x to mask creates a 4-AP."""
        bit_x = 1 << x
        for others_mask in elem_4ap_others[x]:
            if (mask & others_mask) == others_mask:
                return False
        return True

    def count_3aps(mask):
        """Count 3-APs fully contained in mask."""
        cnt = 0
        for m in three_ap_masks:
            if (mask & m) == m:
                cnt += 1
        return cnt

    def calc_gain(x, mask):
        """Calculate number of new 3-APs formed by adding x to mask."""
        gain = 0
        bit_x = 1 << x
        new_mask = mask | bit_x
        for ap_idx in elem_3aps[x]:
            m = three_ap_masks[ap_idx]
            # If the AP was not complete in mask, but is complete in new_mask, it's a gain.
            # Since x is in the AP, if it wasn't complete before, it must be due to x missing.
            # So we just check if the other elements are in mask.
            others_mask = m ^ bit_x
            if (mask & others_mask) == others_mask:
                gain += 1
        return gain

    def calc_loss(x, mask):
        """Calculate number of 3-APs destroyed by removing x from mask."""
        loss = 0
        bit_x = 1 << x
        for ap_idx in elem_3aps[x]:
            m = three_ap_masks[ap_idx]
            others_mask = m ^ bit_x
            # If the AP was complete in mask, it must have had x.
            # So we check if the others were in mask.
            if (mask & others_mask) == others_mask:
                loss += 1
        return loss

    best_mask = 0
    best_score = 0

    # Greedy Phase: Multiple random restarts
    # We can try to be smarter: order elements by potential 3-AP contribution?
    # For now, random shuffle and greedy add.
    for trial in range(200):
        if time.time() - start_time > DEADLINE - 10:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        mask = 0
        for x in order:
            if can_add(x, mask):
                mask |= (1 << x)
        
        score = count_3aps(mask)
        if score > best_score:
            best_score = score
            best_mask = mask

    current_mask = best_mask
    current_score = best_score
    
    # Local Search Phase
    # Try swaps: remove x, add y.
    # We need to ensure validity.
    
    no_improve = 0
    max_no_improve = 500 # Can increase if faster
    
    while time.time() - start_time < DEADLINE - 5:
        improved = False
        
        # Try all single additions
        # This is O(N * N_4APs_per_x) which is small.
        # If we find an add that increases score, do it.
        # Note: Adding an element might block future additions, but if it increases current score, it's good greedily.
        # However, we might want to skip if it doesn't increase score immediately?
        # The problem asks to maximize total 3-APs.
        
        # Let's try additions first.
        for x in range(1, N + 1):
            if time.time() - start_time > DEADLINE - 5:
                break
            if not (current_mask & (1 << x)):
                if can_add(x, current_mask):
                    gain = calc_gain(x, current_mask)
                    if gain > 0:
                        current_mask |= (1 << x)
                        current_score += gain
                        improved = True
                        no_improve = 0
                        break # Restart search
        
        if not improved:
            # Try swaps
            # To speed up, we can restrict candidates.
            # But N=256 is small enough to try all pairs if the inner loop is fast.
            # Inner loop: check validity of y, calc net gain.
            # Validity of y in (mask without x):
            # can_add(y, mask ^ bit_x)
            
            # We can precompute a list of elements not in mask.
            # And list of elements in mask.
            in_mask = [x for x in range(1, N + 1) if (current_mask & (1 << x))]
            not_in_mask = [x for x in range(1, N + 1) if not (current_mask & (1 << x))]
            
            random.shuffle(in_mask)
            random.shuffle(not_in_mask)
            
            for x_out in in_mask:
                if time.time() - start_time > DEADLINE - 5:
                    break
                bit_out = 1 << x_out
                mask_without_out = current_mask ^ bit_out
                
                # Calculate loss of x_out
                # We already know loss = number of 3-APs containing x_out that are complete in current_mask.
                loss = calc_loss(x_out, current_mask)
                
                # If loss is high, maybe skip? No, we need to check if gain compensates.
                
                for x_in in not_in_mask:
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    # Check if x_in can be added to mask_without_out
                    if can_add(x_in, mask_without_out):
                        # Calculate gain of x_in in mask_without_out
                        gain = calc_gain(x_in, mask_without_out)
                        
                        if gain > loss:
                            current_mask = mask_without_out | (1 << x_in)
                            current_score += (gain - loss)
                            improved = True
                            no_improve = 0
                            break
                if improved:
                    break
                    
        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                break
                
    # Convert mask to sorted list
    result = [x for x in range(1, N + 1) if (current_mask & (1 << x))]
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Elements: {result}")
