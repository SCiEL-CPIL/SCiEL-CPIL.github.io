import time
import random

def _is_valid(s):
    """
    Check if the set s has cardinality-separated subset sums.
    Returns True if valid, False otherwise.
    """
    n = len(s)
    # sums[k] is a set of all possible sums using exactly k elements
    sums = [set() for _ in range(n + 1)]
    sums[0].add(0)
    
    for x in s:
        # Update from high k to low k to avoid using x multiple times in one step
        # Actually, we need to update based on the PREVIOUS state.
        # So we should copy the previous state or iterate carefully.
        # Since n is small (<= 60), and we do this inside a search loop, 
        # efficiency matters.
        pass

    # Efficient incremental check for a single set
    # We'll compute all sums for the final set once.
    # For the search, we need fast updates.
    
    # Let's implement a fast full check for the candidate set.
    # If n is ~15, max sum ~ 4000.
    # 15 sets of size up to C(15, 7) ~ 6435.
    # Total elements in all sets ~ 2^15 = 32768.
    # This is very fast in Python.
    
    sums = [set() for _ in range(n + 1)]
    sums[0].add(0)
    for x in s:
        # We need to update sums in reverse order
        # But we must use the sums from the PREVIOUS iteration.
        # So, we can't just do sums[k].update(sums[k-1] + x) in-place without copies.
        # We'll use a temporary list of sets or copy.
        # Given n is small, copying is fine.
        
        # Actually, we can just compute from scratch for each candidate in the search?
        # No, that's too slow if we do 10k iterations.
        # But 32k operations is ~1ms. 10k iterations = 10s. Might be tight.
        # Let's optimize the check.
        
        pass

    # Let's write the full check properly.
    sums = [set() for _ in range(n + 1)]
    sums[0].add(0)
    for x in s:
        # Iterate backwards
        for k in range(n, 0, -1):
            # We need sums[k-1] from the previous state.
            # If we update in place, sums[k-1] might already contain x.
            # So we must snapshot or use a new list.
            # Since n is small, let's just use a new list for each x.
            pass

    # Correct efficient implementation:
    sums = [set() for _ in range(n + 1)]
    sums[0].add(0)
    for x in s:
        # Create a copy of the current sums state
        # Or, iterate k from n down to 1, and use a temporary set for additions?
        # No, standard 0/1 knapsack DP:
        # new_sums[k] = sums[k] U {s+x | s in sums[k-1]}
        # Since we want to check validity of the FINAL set, we can just build it up.
        
        # To be safe and simple:
        # We'll maintain the list of sets.
        # For each x, we update.
        # We must use the state BEFORE adding x.
        # So:
        # prev_sums = [set(s) for s in sums]
        # for k in range(1, n+1):
        #    for s in prev_sums[k-1]:
        #        sums[k].add(s+x)
        # This is O(n * 2^n). For n=15, 15 * 32768 = 500k ops.
        # 1000 restarts = 500M ops. Too slow in Python.
        
        # We need a faster validity check or a smarter search.
        # Observation: If we construct the set greedily with large gaps, it's likely valid.
        # Let's try a **Deterministic Construction** first.
        
        # Strategy:
        # 1. Start with empty set.
        # 2. Try to add the smallest possible integer > last_added that maintains validity.
        # 3. If stuck, backtrack or use a different heuristic.
        
        pass

    # Let's implement the check correctly and efficiently enough for a limited search.
    # We will use a bitset (Python int) for the sums if possible?
    # Max sum 256*15 = 3840.
    # A Python int of 3840 bits is fast.
    # sums[k] is a bitset.
    # sums[k] |= (sums[k-1] << x)
    # Then check if sums[k] & sums[j] == 0 for k != j.
    
    sums = [0] * (n + 1)
    sums[0] = 1  # bit 0 is set (sum 0)
    
    for x in s:
        # Update in reverse
        for k in range(n, 0, -1):
            sums[k] |= (sums[k - 1] << x)
            
    # Check disjointness
    # We need to ensure no bit is set in two different sums[k]
    # We can OR all of them and check if the sum of popcounts equals the popcount of the union?
    # Or simply:
    union = 0
    for k in range(n + 1):
        if union & sums[k]:
            return False
        union |= sums[k]
        
    return True

def construct():
    deadline = time.time() + 55  # 55s margin
    best_set = list(range(1, 9)) # Powers of 2: 1,2,4,8,16,32,64,128? No, 1..8 are not powers of 2.
    # 1, 2, 4, 8, 16, 32, 64, 128.
    best_set = [1, 2, 4, 8, 16, 32, 64, 128]
    if not _is_valid(best_set):
        best_set = [1, 2, 4, 8, 16, 32, 64, 128] # This should be valid.
        
    max_val = 256
    target_size = 14 # Try to reach 14
    
    # Try to improve best_set
    # Strategy: Random Local Search
    # State: a list of unique ints in 1..256
    
    current = best_set[:]
    current_size = len(current)
    
    # We will try to find a valid set of size K, starting from K=9 up to 20.
    # If we find a valid set of size K, we update best.
    
    for target_k in range(current_size + 1, 21):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 2000 # Limit per size
        
        while attempts < max_attempts and not found:
            attempts += 1
            if time.time() > deadline:
                break
            
            # Generate a random candidate of size target_k
            # Bias towards larger numbers? Or random?
            # Let's try random sampling from 1..256.
            # To ensure distinctness, use sample.
            # But random sets are rarely valid.
            # Better: Start from a valid smaller set and add/swap.
            
            # Let's try a "Greedy with Randomization" approach.
            # 1. Start with a random valid set of size target_k - 1 (or just random).
            # 2. Try to add a random element.
            # 3. If invalid, try to swap one element.
            
            # Simpler: Random Restart Hill Climbing on the set of size target_k.
            # Objective: Minimize number of collisions.
            
            # Let's define a collision counter.
            # We'll just use _is_valid.
            
            # To make it faster, we can try to construct from a "seed" structure.
            # E.g., start with powers of 2 and add random elements, then clean up.
            
            # Let's try:
            # 1. Take best_set (valid).
            # 2. Add random elements until size target_k.
            # 3. Run local search: pick two elements, swap one with a random new one.
            #    If validity improves (or is maintained), keep it.
            
            cand = best_set[:]
            # Add random elements
            while len(cand) < target_k:
                # Pick a random number not in cand
                candidates = [i for i in range(1, max_val + 1) if i not in cand]
                if not candidates:
                    break
                cand.append(random.choice(candidates))
            
            # Now we have a set of size target_k (maybe with duplicates? No, we checked).
            # It is likely invalid.
            # Local search:
            # We want to minimize "invalidity".
            # Since _is_valid is boolean, we can't easily minimize.
            # We can count collisions.
            
            def count_collisions(s):
                n = len(s)
                sums = [0] * (n + 1)
                sums[0] = 1
                for x in s:
                    for k in range(n, 0, -1):
                        sums[k] |= (sums[k - 1] << x)
                
                # Count bits that are in multiple sums
                # This is hard with bitsets.
                # Alternative: Just use _is_valid.
                # If we use _is_valid, we can do a "First Fit" search.
                # Swap an element with a random other element.
                # If _is_valid(new_cand) is True, return.
                
                return 0 # Placeholder
            
            # Since _is_valid is fast (~1ms), we can do many swaps.
            # But 2000 attempts * 1ms = 2s per target_k.
            # 10 target_k values = 20s. Feasible.
            
            # Better: Use a "Cooling" schedule or just random swaps.
            # Let's do random swaps.
            
            for _ in range(50): # 50 swaps per attempt
                if len(cand) < target_k:
                    break
                i = random.randint(0, len(cand) - 1)
                old_val = cand[i]
                # Pick a new value
                candidates = [j for j in range(1, max_val + 1) if j not in cand]
                if not candidates:
                    break
                new_val = random.choice(candidates)
                cand[i] = new_val
                
                if _is_valid(cand):
                    best_set = cand[:]
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            # If we couldn't find a valid set of size target_k, 
            # it might be too hard or we ran out of time.
            # We continue to next target_k? No, if we can't do K, we probably can't do K+1.
            # But maybe a different random seed would work.
            # We'll just stop trying larger sizes if we fail here, 
            # unless we have lots of time.
            # Actually, let's just break to save time for final verification.
            # Or try a few more restarts?
            # For now, break.
            pass

    # Final check and conversion
    # Ensure all are Python ints
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-START
import time
import random

def _is_valid(s):
    """
    Check if the set s has cardinality-separated subset sums.
    Returns True if valid, False otherwise.
    Uses bitset DP.
    """
    n = len(s)
    if n == 0:
        return True
        
    # sums[k] is a bitset where bit j is 1 if sum j is achievable with k elements
    # Max sum: n * 256. For n=20, max sum 5120.
    # Python ints are arbitrary precision, so this is fine.
    
    sums = [0] * (n + 1)
    sums[0] = 1  # Sum 0 with 0 elements
    
    for x in s:
        # Update in reverse to avoid using x multiple times
        for k in range(n, 0, -1):
            sums[k] |= (sums[k - 1] << x)
            
    # Check if any sum is reachable with different cardinalities
    union = 0
    for k in range(n + 1):
        if union & sums[k]:
            return False
        union |= sums[k]
        
    return True

def construct():
    deadline = time.time() + 55  # 55s margin
    
    # Initial valid set: Powers of 2
    # 1, 2, 4, 8, 16, 32, 64, 128
    best_set = [1, 2, 4, 8, 16, 32, 64, 128]
    
    max_val = 256
    
    # Try to increase the size
    # We will attempt to find valid sets of size 9, 10, 11, ...
    
    current_best_size = len(best_set)
    
    for target_k in range(current_best_size + 1, 21):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        # Increase attempts if we have time
        # Start with 100 attempts, increase if successful?
        # Let's use a fixed budget per size.
        max_attempts = 500
        
        while attempts < max_attempts and not found:
            attempts += 1
            if time.time() > deadline:
                break
            
            # Generate candidate
            # Start from best_set and add random elements
            cand = best_set[:]
            
            # Add random unique elements
            available = [i for i in range(1, max_val + 1) if i not in cand]
            random.shuffle(available)
            
            for val in available:
                if len(cand) >= target_k:
                    break
                cand.append(val)
            
            if len(cand) < target_k:
                continue
                
            # Local Search: Random Swaps
            # We try to fix the set by swapping elements.
            # We accept a swap if it leads to a valid set.
            # Since validity is boolean, we just search for a valid configuration.
            
            # To make it smarter, we could try to swap elements that are "causing" collisions.
            # But identifying collisions is complex.
            # Random swaps might work if the space of valid sets is not too sparse.
            
            # Let's do 20 swaps per attempt.
            for _ in range(20):
                i = random.randint(0, len(cand) - 1)
                old_val = cand[i]
                
                # Pick a new value
                # To avoid duplicates, we pick from 1..256 excluding current cand values
                # But we can just pick a random 1..256 and check if it's in cand
                # If it is, skip or pick another.
                
                # Efficient way:
                # Create a list of all possible values not in cand
                # This is O(256). Fast.
                cand_set = set(cand)
                cand_set.remove(old_val)
                candidates = [j for j in range(1, max_val + 1) if j not in cand_set]
                
                if not candidates:
                    break
                    
                new_val = random.choice(candidates)
                cand[i] = new_val
                
                if _is_valid(cand):
                    best_set = cand[:]
                    found = True
                    break
        
        if found:
            # Update best_set to the new valid set
            # We continue to next target_k
            pass
        else:
            # If we failed to find a valid set of size target_k,
            # we might still try target_k+1? Unlikely to help.
            # But maybe a different random start?
            # We'll just break to save time.
            # Actually, let's try a few more times if we have time?
            # No, break is safer.
            break

    # Ensure return type is list of ints
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
