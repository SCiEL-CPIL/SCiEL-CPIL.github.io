import time
import random

def construct():
    start = time.time()
    deadline = start + 55.0  # 55s safety margin
    
    def get_subset_sums(elements):
        """Return set of all nonempty subset sums."""
        sums = set()
        n = len(elements)
        # Bitmask iteration is fast enough for n <= 20
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums

    def is_valid(sums):
        """Check if any two distinct sums divide each other."""
        if not sums:
            return True
        sum_list = sorted(sums)
        max_sum = sum_list[-1]
        # For each sum, check if any multiple is also in the set
        for s in sum_list:
            # Check multiples 2s, 3s, ...
            # Since we only care if another sum divides this one or vice versa,
            # and sum_list is sorted, if s_i divides s_j, then s_j is a multiple of s_i.
            # So checking if multiples of s_i are in the set is sufficient.
            multiple = 2 * s
            while multiple <= max_sum:
                if multiple in sums:
                    return False
                multiple += s
        return True

    def check_set(elements):
        sums = get_subset_sums(elements)
        return is_valid(sums)

    # Initial set: Try to find a good starting point
    # A set of large numbers is less likely to have small divisors in its sums?
    # Actually, small numbers create small sums which are more likely to divide larger sums.
    # So we prefer larger numbers.
    
    candidates = list(range(1, 257))
    random.shuffle(candidates)
    
    best_set = []
    best_size = 0
    
    # Try to build a set greedily from large to small?
    # Or just random restarts.
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        # Start with a random set of size 15-18
        # Or start empty and add?
        # Adding is hard to backtrack.
        # Let's start with a random set of size 16 and try to expand/swap.
        
        current_set = set(random.sample(candidates, 16))
        # Ensure it's valid? If not, shrink it or restart.
        # For speed, we might just assume a random set of 16 is invalid and use local search to fix/expand.
        # Better: Start with a known valid small set and grow.
        
        # Let's try a deterministic start:
        # Powers of 2 are bad. 
        # Primes?
        
        # Strategy: Simulated Annealing on set size and composition.
        # State: Set of elements.
        # Objective: Maximize size, subject to validity.
        # If invalid, penalize heavily.
        
        # Simpler: Random restarts with greedy insertion.
        # Sort candidates by some heuristic? 
        # If we pick large numbers, sums are large.
        # Let's try picking from high end.
        
        current = []
        current_sums = set()
        
        # Greedy insertion
        # Try adding elements in a random order, but biased towards larger values?
        # Or just random order.
        
        shuffled_candidates = candidates[:]
        random.shuffle(shuffled_candidates)
        
        for c in shuffled_candidates:
            if len(current) >= 30:
                break
            # Try adding c
            new_elements = current + [c]
            
            # Efficient sum update
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + c)
            new_sums.add(c)
            
            # Check validity
            if is_valid(new_sums):
                current = new_elements
                current_sums = new_sums
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 20: # If we get good, maybe stop early? No, keep searching.
                pass

    # Local search improvement on best_set
    # Try to swap elements to increase size?
    # Actually, if we found a set of size K, can we find K+1?
    # We can try to replace pairs of elements with single elements? No, that reduces size.
    # We want to ADD elements.
    # If greedy got stuck at size K, it means no single element from candidates could be added.
    # But maybe swapping one element allows adding two?
    
    # Let's do a simple local search on the best set found.
    if best_set:
        current = best_set[:]
        current_sums = get_subset_sums(current)
        
        # Try to add elements by swapping
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            # Try to add a random element
            c = random.choice(candidates)
            if c in current:
                continue
            
            # If we just add, size increases by 1.
            # If it's valid, we accept.
            new_elements = current + [c]
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + c)
            new_sums.add(c)
            
            if is_valid(new_sums):
                current = new_elements
                current_sums = new_sums
                best_size = len(current)
                best_set = current[:]
                continue
            
            # If adding fails, try swapping an element with c
            # Swap i with c
            i = random.randint(0, len(current) - 1)
            old_elem = current[i]
            new_elements = current[:]
            new_elements[i] = c
            
            # Recalculate sums? Expensive.
            # But we can just check validity of the new set.
            # To speed up, we can check if the new set is valid.
            # If valid, we accept the swap (size stays same, but composition changes).
            # This might allow adding an element later.
            
            # Optimization: Only swap if it doesn't decrease "potential"?
            # Hard to define potential.
            # Just accept any valid swap with some probability?
            # Or accept if it leads to a state where we can add?
            
            # Let's just accept valid swaps to explore the space.
            if is_valid(get_subset_sums(new_elements)):
                # Update sums efficiently?
                # Removing old_elem and adding c.
                # Sums containing old_elem need to be updated.
                # This is complex. Let's just recompute if size is small.
                # For size ~18, 2^18 = 260k sums. Recomputing is ~0.1s.
                # 5000 swaps * 0.1s = 500s. Too slow.
                
                # We need incremental sum update.
                # Let's stick to just trying to ADD.
                # If we can't add, we are stuck.
                # Maybe random restarts are better.
                
                pass # Skip swaps for now to keep code simple and fast, rely on restarts.

    return best_set
