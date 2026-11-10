import random
import time

def construct():
    N = 512
    start_time = time.time()
    timeout = 55.0 # Safety margin
    
    # Precompute 4-APs
    # An AP is (a, a+d, a+2d, a+3d)
    # Store as tuples of indices (0-based or 1-based? 1-based for set)
    aps = []
    for d in range(1, (N-1)//3 + 1):
        for a in range(1, N - 3*d + 1):
            b = a + d
            c = a + 2*d
            dd = a + 3*d
            aps.append((a, b, c, dd))
            
    # Map number to list of AP indices it belongs to
    num_to_aps = {i: [] for i in range(1, N+1)}
    for idx, ap in enumerate(aps):
        for num in ap:
            num_to_aps[num].append(idx)

    def is_valid_addition(x, current_set, ap_membership):
        # Check if adding x creates a 4-AP
        # x is in some APs. If any of those APs have the other 3 members in current_set, it's invalid.
        for ap_idx in num_to_aps[x]:
            ap = aps[ap_idx]
            # Check if other 3 are in current_set
            others = [y for y in ap if y != x]
            if all(y in current_set for y in others):
                return False
        return True

    best_set = set()
    
    # Strategy 1: Randomized Greedy
    # Try multiple random orders
    for restart in range(1000):
        if time.time() - start_time > timeout:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        current_set = set()
        for x in order:
            if is_valid_addition(x, current_set, num_to_aps):
                current_set.add(x)
        
        if len(current_set) > len(best_set):
            best_set = set(current_set)
            
    # Strategy 2: Local Search / Swap on the best set found
    # We try to improve best_set
    current_set = set(best_set)
    improved = True
    while improved and (time.time() - start_time < timeout):
        improved = False
        # Find candidates to add
        candidates = [x for x in range(1, N+1) if x not in current_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > timeout:
                break
            
            # Check if x is blocked
            if is_valid_addition(x, current_set, num_to_aps):
                current_set.add(x)
                improved = True
                # Restart candidate check? Or just continue
                # Adding x might block others, but we just want to grow
                continue
            
            # x is blocked. Find which APs block it.
            blocking_aps = []
            for ap_idx in num_to_aps[x]:
                ap = aps[ap_idx]
                others = [y for y in ap if y != x]
                if all(y in current_set for y in others):
                    blocking_aps.append(ap)
            
            if not blocking_aps:
                continue
                
            # Try to remove one element from one of the blocking APs to allow x
            # We want to see if removing y allows adding x (and maybe more)
            # Simple swap: remove y, add x. Size stays same. 
            # But maybe removing y unblocks other additions?
            # Let's try a 1-for-1 swap first to see if it leads to growth.
            # Actually, if we just swap, size is constant. 
            # We need to find a sequence that increases size.
            # Heuristic: Try removing a 'high degree' element? Or random?
            
            # Let's try to find a y in the blocking APs such that removing y 
            # allows adding x AND maybe another element z?
            # This is getting complex. 
            # Let's just do random swaps if we get stuck on additions.
            
            # Pick a random blocking AP
            ap = random.choice(blocking_aps)
            others = [y for y in ap if y != x]
            y = random.choice(others)
            
            # Tentatively remove y and add x
            new_set = set(current_set)
            new_set.remove(y)
            new_set.add(x)
            
            # Check if new_set is valid (it should be, since we removed a blocker)
            # But removing y might not break other APs? 
            # Wait, removing y can only reduce violations. Adding x might create violations 
            # other than the ones we fixed? 
            # No, we only added x. The violations involving x are fixed by removing y 
            # IF y was in ALL blocking APs? 
            # No, y is in ONE blocking AP. x might be blocked by MULTIPLE APs.
            # We need to remove elements from ALL blocking APs.
            
            # So, a single swap is only valid if x is blocked by exactly one AP.
            if len(blocking_aps) == 1:
                # Remove y, add x
                # Check validity of new_set
                # Since we removed y, the AP is broken.
                # Are there other APs involving x?
                # If x was blocked by only this one AP, then removing y fixes it.
                # Does adding x create other APs?
                # No, because we checked is_valid_addition initially. 
                # The only reason it failed was this AP.
                # So new_set is valid.
                current_set = new_set
                improved = True
                # Size is same, but we might have unlocked other additions
                # Break to restart candidate loop with new set
                break
                
    return sorted(current_set)

# EVOLVE-BLOCK-START
def construct():
    import random
    import time
    import numpy as np

    N = 512
    start_time = time.time()
    timeout = 55.0

    # Precompute 4-APs
    aps = []
    for d in range(1, (N-1)//3 + 1):
        for a in range(1, N - 3*d + 1):
            b = a + d
            c = a + 2*d
            dd = a + 3*d
            aps.append((a, b, c, dd))
            
    num_to_aps = {i: [] for i in range(1, N+1)}
    for idx, ap in enumerate(aps):
        for num in ap:
            num_to_aps[num].append(idx)

    def is_valid_addition(x, current_set):
        for ap_idx in num_to_aps[x]:
            ap = aps[ap_idx]
            others = [y for y in ap if y != x]
            if all(y in current_set for y in others):
                return False
        return True

    best_set = set()
    
    # Randomized Greedy Restarts
    rng = random.Random(42)
    
    for restart in range(500):
        if time.time() - start_time > timeout:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        current_set = set()
        for x in order:
            if is_valid_addition(x, current_set):
                current_set.add(x)
        
        if len(current_set) > len(best_set):
            best_set = set(current_set)
            
    # Local Search: additions, 1-swaps (multi-AP aware), and 2-swaps
    current_set = set(best_set)
    improved = True
    while improved and (time.time() - start_time < timeout):
        improved = False
        candidates = [x for x in range(1, N+1) if x not in current_set]
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > timeout:
                break
            
            if is_valid_addition(x, current_set):
                current_set.add(x)
                improved = True
                break
            
            # Find all blocking APs for x
            blocking_aps = []
            for ap_idx in num_to_aps[x]:
                ap = aps[ap_idx]
                others = [y for y in ap if y != x]
                if all(y in current_set for y in others):
                    blocking_aps.append(ap)
            
            if not blocking_aps:
                continue
            
            # Try 1-swap: remove one element from one blocking AP
            # This works if x is blocked by exactly one AP
            if len(blocking_aps) == 1:
                ap = blocking_aps[0]
                others = [y for y in ap if y != x]
                for y in others:
                    if time.time() - start_time > timeout:
                        break
                    # Check: after removing y, is x addable?
                    # y is in this AP. If x is only blocked by this AP, removing y fixes it.
                    # But we need to verify no other AP involving x becomes complete.
                    # Since x was only blocked by this one AP, removing any element from it
                    # should allow x to be added.
                    new_set = set(current_set)
                    new_set.remove(y)
                    new_set.add(x)
                    # Verify validity of new_set (paranoia check)
                    valid = True
                    for ap_idx2 in num_to_aps[x]:
                        ap2 = aps[ap_idx2]
                        others2 = [z for z in ap2 if z != x]
                        if all(z in new_set for z in others2):
                            valid = False
                            break
                    if valid:
                        current_set = new_set
                        improved = True
                        break
                if improved:
                    break
            
            # Try 2-swap: remove two elements (one from each of two blocking APs)
            # This handles the case where x is blocked by 2 APs
            elif len(blocking_aps) == 2:
                ap1, ap2 = blocking_aps[0], blocking_aps[1]
                others1 = [y for y in ap1 if y != x]
                others2 = [y for y in ap2 if y != x]
                # Try all pairs
                found = False
                for y1 in others1:
                    if time.time() - start_time > timeout:
                        break
                    for y2 in others2:
                        if y1 == y2:
                            continue
                        new_set = set(current_set)
                        new_set.discard(y1)
                        new_set.discard(y2)
                        new_set.add(x)
                        # Verify
                        valid = True
                        for ap_idx2 in num_to_aps[x]:
                            ap2_ = aps[ap_idx2]
                            others2_ = [z for z in ap2_ if z != x]
                            if all(z in new_set for z in others2_):
                                valid = False
                                break
                        if valid:
                            current_set = new_set
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break

    return sorted(current_set)
# EVOLVE-BLOCK-END
