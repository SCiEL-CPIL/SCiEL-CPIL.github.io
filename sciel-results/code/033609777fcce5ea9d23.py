import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    max_val = 256
    max_size = 60
    
    def try_order(order):
        """
        Greedy construction with bitset DP for fast validity checks.
        """
        # masks[k] is a bitset where bit s is 1 if sum s is reachable with k elements
        masks = [0] * (max_size + 1)
        masks[0] = 1  # Sum 0 with 0 elements
        
        current_set = []
        
        for x in order:
            if len(current_set) >= max_size:
                break
            if time.time() > deadline:
                break
            
            # Check if adding x is valid
            # For each source cardinality i, new sums are masks[i] << x.
            # These new sums will be part of cardinality i+1.
            # They must not collide with masks[j] for any j != i+1.
            # Collision with masks[i+1] is allowed (same cardinality).
            
            valid = True
            # i is the source cardinality index in old masks
            # Max source index is len(current_set)
            for i in range(len(current_set) + 1):
                if masks[i] == 0:
                    continue
                
                new_sums = masks[i] << x
                target_k = i + 1
                
                # Check collision with all other cardinalities j != target_k
                # j ranges from 0 to len(current_set)
                for j in range(len(current_set) + 1):
                    if j == target_k:
                        continue
                    if masks[j] != 0 and (new_sums & masks[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Update masks
                # Update in reverse order to avoid using x multiple times
                # masks[k] |= masks[k-1] << x
                for k in range(len(current_set) + 1, 0, -1):
                    masks[k] |= (masks[k-1] << x)
                current_set.append(x)
        
        return current_set

    def local_search(initial_set, time_limit):
        """
        Attempt to improve the set by swapping or adding elements.
        """
        if time.time() > time_limit:
            return initial_set
        
        current_set = list(initial_set)
        current_size = len(current_set)
        
        # Build masks for current_set
        masks = [0] * (max_size + 1)
        masks[0] = 1
        for x in current_set:
            for k in range(current_size, 0, -1):
                if masks[k-1]:
                    masks[k] |= (masks[k-1] << x)
        
        used = set(current_set)
        candidates = [x for x in range(1, max_val + 1) if x not in used]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try adding a new element
            if current_size < max_size:
                for x in candidates:
                    if time.time() > time_limit:
                        break
                    
                    # Check if x can be added
                    valid = True
                    for i in range(current_size + 1):
                        if masks[i] == 0:
                            continue
                        new_sums = masks[i] << x
                        target_k = i + 1
                        for j in range(current_size + 1):
                            if j == target_k:
                                continue
                            if masks[j] != 0 and (new_sums & masks[j]):
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Add x
                        for k in range(current_size + 1, 0, -1):
                            masks[k] |= (masks[k-1] << x)
                        current_set.append(x)
                        current_size += 1
                        used.add(x)
                        candidates.remove(x)
                        improved = True
                        break
            
            # Try swapping an element for a larger one to free up space?
            # This is complex. Let's stick to adding for now, or simple swaps.
            # Actually, the greedy already does a full pass. Local search might not add much 
            # unless we restart with different orders.
            
            # Let's try swapping: remove one, try to add a different one?
            # Or just rely on multiple greedy passes with different orders.
            
        return current_set

    best_set = []
    
    # Strategy 1: Descending order
    order_desc = list(range(max_val, 0, -1))
    res = try_order(order_desc)
    if len(res) > len(best_set):
        best_set = res
    
    # Strategy 2: Ascending order
    if time.time() < deadline - 5:
        order_asc = list(range(1, max_val + 1))
        res = try_order(order_asc)
        if len(res) > len(best_set):
            best_set = res
    
    # Strategy 3: Randomized orders
    # Use a fixed seed for reproducibility within the run, but vary across attempts
    for attempt in range(200):
        if time.time() > deadline - 2.0:
            break
        
        # Create a biased random order
        # Larger numbers are generally better for spreading sums
        # Let's try a mix: 70% chance to pick from top half, 30% from bottom?
        # Or just shuffle with a seed.
        random.seed(attempt * 12345 + 67890)
        
        # Method: Generate a random permutation, but sort by a key that mixes value and randomness
        # Or simply:
        candidates = list(range(1, max_val + 1))
        
        # Bias towards larger numbers
        # Assign a weight to each number, e.g., value itself
        # Then sample without replacement using these weights?
        # Simpler: Shuffle, then reverse? No, that's just descending.
        
        # Let's try: Take the top 50% numbers, shuffle them. Take the bottom 50%, shuffle them.
        # Concatenate?
        top_half = list(range(max_val // 2 + 1, max_val + 1))
        bot_half = list(range(1, max_val // 2 + 1))
        
        random.shuffle(top_half)
        random.shuffle(bot_half)
        
        # Interleave or concatenate?
        # Concatenating top then bottom might be good.
        order = top_half + bot_half
        
        res = try_order(order)
        if len(res) > len(best_set):
            best_set = res
            
            # If we found a large set, maybe do local search on it?
            # But local search implementation above is basic.
            # Let's just keep trying different orders.
            
        # Also try pure shuffle
        if time.time() < deadline - 2.0:
            random.seed(attempt * 999 + 1)
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
            res = try_order(candidates)
            if len(res) > len(best_set):
                best_set = res

    # Ensure we return a valid list of ints
    if not best_set:
        # Fallback: use powers of 2 which always work
        best_set = [2**i for i in range(9) if 2**i <= 256]
    
    return [int(x) for x in best_set]
