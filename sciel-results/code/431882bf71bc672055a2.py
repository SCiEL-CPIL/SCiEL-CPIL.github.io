import time
import random

def _can_add(current_reach, v, target):
    """
    Check if adding v to the set with reachability bitmask 'current_reach' 
    results in 'target' being reachable.
    Returns (True, new_reach) if safe, (False, current_reach) if not.
    """
    # If v itself is target, it's unsafe (unless target is 0, but target is 200)
    if v == target:
        return False, current_reach
    
    # new_reach = current_reach | (current_reach << v)
    # We only care about bits up to target.
    # We can mask to keep the integer small, though Python handles big ints well.
    # Mask: (1 << (target + 1)) - 1
    mask = (1 << (target + 1)) - 1
    
    new_reach = current_reach | ((current_reach << v) & mask)
    
    # Check if target bit is set in new_reach
    if (new_reach >> target) & 1:
        return False, current_reach
    
    return True, new_reach

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    target = 200
    universe = list(range(1, 101))
    
    best_set = None
    best_size = 0
    
    # Strategies: Different orderings to build the set
    # 1. Ascending (small numbers first)
    # 2. Descending (large numbers first)
    # 3. Random (multiple iterations)
    
    # Strategy 1: Ascending
    if time.time() < deadline:
        order = sorted(universe)
        S = set()
        reach = 1  # Bit 0 is set (sum 0 is reachable)
        mask = (1 << (target + 1)) - 1
        
        for v in order:
            safe, new_reach = _can_add(reach, v, target)
            if safe:
                S.add(v)
                reach = new_reach
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    # Strategy 2: Descending
    if time.time() < deadline:
        order = sorted(universe, reverse=True)
        S = set()
        reach = 1
        mask = (1 << (target + 1)) - 1
        
        for v in order:
            safe, new_reach = _can_add(reach, v, target)
            if safe:
                S.add(v)
                reach = new_reach
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    # Strategy 3: Randomized Greedy with Local Search
    # We run this multiple times if time permits.
    # The bitset approach makes local search very fast.
    
    while time.time() < deadline - 5.0:
        # Randomized Greedy Construction
        order = universe.copy()
        random.shuffle(order)
        
        S = set()
        reach = 1
        mask = (1 << (target + 1)) - 1
        
        for v in order:
            safe, new_reach = _can_add(reach, v, target)
            if safe:
                S.add(v)
                reach = new_reach
        
        # Local Search: Try to add back removed elements
        # We maintain the 'reach' bitmask.
        # If we add an element, we update reach.
        # If we fail to add, we skip.
        
        # To do local search effectively, we should try to add elements 
        # that were skipped. 
        # However, with the greedy bitset approach, adding an element 
        # that was previously rejected might become possible if we 
        # had removed some other elements? No, we are only adding.
        # So "Local Search" here means: try a different random order.
        
        # Actually, a better local search is: 
        # Take the current S. 
        # Try to add every element not in S.
        # If we can add it, we do.
        # Since we only add, the set grows.
        
        # Let's refine: 
        # The greedy pass might miss elements because of the order.
        # If we process [1, 2, 3...], we might reject 100 because 
        # 100 + (subset of 1..99 summing to 100) = 200? 
        # Actually, if we have {1..10}, sum is 55. 200-100=100. 
        # Can we make 100 from 1..99? Yes. So 100 is rejected.
        
        # If we process [100, 99, ...], we accept 100. 
        # Then 99: 200-99=101. Can we make 101 from {100}? No. So accept 99.
        # Then 98: 200-98=102. From {100, 99}? No. Accept 98.
        # Descending tends to keep large numbers.
        
        # Let's try a "Hybrid" or "Shuffled" approach with a second pass.
        # Pass 1: Build S.
        # Pass 2: Iterate through all elements not in S. 
        #         Try to add them. If successful, add and update reach.
        #         This is equivalent to just running the greedy loop on 
        #         (universe - S) in some order.
        
        # Since the bitset check is O(1) (big int shift), we can afford 
        # to try adding elements in a few different orders in the local search phase.
        
        # Let's just do one more pass on the remaining elements in random order.
        remaining = [x for x in universe if x not in S]
        random.shuffle(remaining)
        
        for v in remaining:
            if time.time() > deadline - 2.0:
                break
            safe, new_reach = _can_add(reach, v, target)
            if safe:
                S.add(v)
                reach = new_reach
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
            # If we hit a new best, we might want to try to improve it further 
            # by shuffling the *entire* universe again? 
            # Or just continue the loop.
            
    if best_set is None:
        best_set = {100}
        best_size = 1

    return list(best_set)
