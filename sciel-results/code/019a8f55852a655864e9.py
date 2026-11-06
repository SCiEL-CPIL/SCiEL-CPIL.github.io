import random
import time

def creates_ap(S, x, n=512):
    # Same as before, but optimized with set lookups
    # x is 4th: a, a+d, a+2d, x
    for d in range(1, x // 3 + 1):
        a = x - 3 * d
        if a >= 1 and a in S and (a + d) in S and (a + 2 * d) in S:
            return True
    # x is 3rd: a, a+d, x, a+3d
    for d in range(1, x // 2 + 1):
        a = x - 2 * d
        if a >= 1:
            b3 = a + 3 * d
            if b3 <= n and a in S and (a + d) in S and b3 in S:
                return True
    # x is 2nd: a, x, a+2d, a+3d
    for d in range(1, x):
        a = x - d
        if a >= 1:
            b2 = a + 2 * d
            b3 = a + 3 * d
            if b3 <= n and a in S and b2 in S and b3 in S:
                return True
    # x is 1st: x, x+d, x+2d, x+3d
    for d in range(1, (n - x) // 3 + 1):
        b1 = x + d
        b2 = x + 2 * d
        b3 = x + 3 * d
        if b3 <= n and b1 in S and b2 in S and b3 in S:
            return True
    return False

def construct():
    n = 512
    start_time = time.time()
    max_time = 55.0
    
    # 1. Initial Random Greedy
    best_set = set()
    best_size = 0
    
    # Run a few random greedy passes to find a good starting point
    for _ in range(10):
        if time.time() - start_time > max_time - 5: break
        order = list(range(1, n + 1))
        random.shuffle(order)
        current_set = set()
        for x in order:
            if not creates_ap(current_set, x, n):
                current_set.add(x)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    # 2. Local Search / Iterative Addition
    # We try to add elements to best_set
    # If we can't add, we try swaps to "unlock" additions?
    # Simpler: Just try to add any valid element repeatedly.
    # If stuck, try to remove a random element and see if we can add two?
    # Or just try swaps: remove x, add y.
    
    current_set = best_set
    current_size = best_size
    available = set(range(1, n + 1)) - current_set
    
    # Phase 2: Greedy Addition on the best found so far
    # We can iterate through available elements and add them if valid.
    # Do this in random order to avoid bias.
    while time.time() - start_time < max_time - 2:
        progress = False
        # Shuffle available to check
        avail_list = list(available)
        random.shuffle(avail_list)
        
        for y in avail_list:
            if time.time() - start_time > max_time - 2: break
            if not creates_ap(current_set, y, n):
                current_set.add(y)
                available.remove(y)
                current_size += 1
                progress = True
                # Reset loop to check new availability? 
                # Or just continue? Adding y might block others, but doesn't unlock.
                # So we can just continue checking other y's.
                # But creating a new avail_list is expensive.
                # Let's just break and restart the loop to be safe?
                # No, if we add y, we just removed it from available.
                # We can continue checking the rest of avail_list.
        
        if not progress:
            # Stuck. Try swaps.
            # Pick a random x in current_set, remove it.
            # Try to add a random y from available.
            # If valid, keep it. (Size stays same, but configuration changes)
            # This might allow adding more later.
            
            # To avoid infinite loops on swaps without gain, we limit swap attempts.
            swaps_made = 0
            while swaps_made < 100 and time.time() - start_time < max_time - 2:
                if not current_set: break
                x = random.choice(list(current_set))
                current_set.remove(x)
                available.add(x)
                
                # Try to add some y
                y_added = False
                # Try a few random y's
                for _ in range(10):
                    if not available: break
                    y = random.choice(list(available))
                    if not creates_ap(current_set, y, n):
                        current_set.add(y)
                        available.remove(y)
                        y_added = True
                        break
                        
                if y_added:
                    # We swapped x for y. Size is same.
                    # But now we might be able to add more.
                    # Let's exit swap loop and try adding again.
                    break
                else:
                    # Could not add y. Put x back? Or keep x removed and try another?
                    # If we put x back, we wasted time.
                    # If we keep x removed, size decreases. We don't want that.
                    # So we must revert if we don't add y.
                    current_set.add(x)
                    available.remove(x)
                swaps_made += 1
            
            # If we made swaps, loop back to try adding again.
            # If we didn't make swaps (or failed to improve), we might be stuck.
            # To prevent infinite loop, we can break if no progress in a long time.
            # For now, just break the outer loop if no additions were made in the last pass.
            # But we did swaps. So we should try adding again.
            # The 'progress' flag was False for additions.
            # We need to distinguish "stuck, no swaps" vs "swapped, try again".
            
            # If we just swapped, we should go back to the top of the while loop to try adding.
            # If we couldn't swap effectively, we might be at a local optimum.
            # We can break here to save time, or continue with more aggressive swaps.
            # Given time constraints, breaking is safer if we can't add.
            # But maybe a different swap works.
            # Let's just break the outer loop if we didn't add anything in the last full pass.
            # The swaps were just to change state.
            
            # Actually, if we swapped, we should retry adding.
            # So we don't break. We just continue the while loop.
            # But we need to ensure we don't loop forever swapping.
            # We can track 'stall' counter.
            
            # For simplicity and safety:
            # If no addition was made in the last pass, and we've done some swaps, 
            # we can break if we feel we've plateaued.
            # Let's just break if no addition was made in the last pass to ensure termination.
            # The swaps might have helped, but we check again next pass.
            # If the next pass also yields no addition, we break.
            
            # To be safe, let's just break here. The random greedy + one pass of addition 
            # is likely sufficient for this N.
            break
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set

    return sorted(best_set)
