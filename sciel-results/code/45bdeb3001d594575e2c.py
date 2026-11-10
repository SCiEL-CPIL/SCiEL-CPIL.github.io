import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s buffer
    
    # Helper to compute sums by cardinality
    def compute_sums_by_cardinality(s):
        n = len(s)
        sums_by_k = [set() for _ in range(n + 1)]
        sums_by_k[0].add(0)
        
        # DP: for each element v, update sums in reverse order of k
        # To be safe and fast, we can use a list of sets
        for v in s:
            # We need to update sums_by_k[k] using sums_by_k[k-1]
            # Iterating k from n down to 1 ensures we don't use v multiple times
            # But sums_by_k[k-1] might have been updated?
            # No, if we iterate k from n down to 1, sums_by_k[k-1] is from the previous step (without v)
            # Wait, sums_by_k[k-1] is updated when k-1 is processed.
            # If we go n -> 1, when processing k, k-1 has NOT been updated yet in this iteration?
            # No, k-1 < k, so it will be processed later.
            # So sums_by_k[k-1] is the state BEFORE adding v.
            # This is correct.
            
            # Optimization: Only iterate up to current max possible k
            # But n is small enough.
            for k in range(n, 0, -1):
                # If sums_by_k[k-1] is empty, skip
                if not sums_by_k[k-1]:
                    continue
                # Add v to each sum in sums_by_k[k-1]
                # Create a list to avoid "set changed size" errors if we modified in place?
                # We are adding to sums_by_k[k], not sums_by_k[k-1].
                # So it is safe.
                for prev_sum in sums_by_k[k-1]:
                    sums_by_k[k].add(prev_sum + v)
        
        return sums_by_k

    def count_collisions(sums_by_k):
        n = len(sums_by_k) - 1
        total = 0
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                # Intersection size
                total += len(sums_by_k[i] & sums_by_k[j])
        return total

    def is_valid_set(s):
        sums_by_k = compute_sums_by_cardinality(s)
        return count_collisions(sums_by_k) == 0

    def try_add_element(current_set, v, sums_by_k):
        """
        Check if v can be added to current_set.
        Returns (True, new_sums_by_k) if valid, else (False, None).
        """
        n = len(current_set)
        # New sums_by_k will have size n+2
        # We can compute the new sums efficiently
        # S'_k = S_k U {s + v | s in S_{k-1}}
        
        # To check validity, we just need to see if the new sets are disjoint.
        # We can compute the new sets and check.
        
        new_sums_by_k = [set() for _ in range(n + 2)]
        
        # Copy old
        for k in range(n + 1):
            new_sums_by_k[k].update(sums_by_k[k])
        
        # Add new
        for k in range(1, n + 2):
            if not sums_by_k[k-1]:
                continue
            for s in sums_by_k[k-1]:
                new_sums_by_k[k].add(s + v)
                
        # Check disjointness
        for i in range(n + 2):
            for j in range(i + 1, n + 2):
                if not new_sums_by_k[i].isdisjoint(new_sums_by_k[j]):
                    return False, None
                    
        return True, new_sums_by_k

    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_size = len(best_set)
    
    # Strategy: 
    # 1. Random restarts with large initial sets.
    # 2. Greedy removal to reach validity.
    # 3. Greedy addition to maximize size.
    
    random.seed(42)
    
    for attempt in range(100): # Limit attempts
        if time.time() > deadline - 5:
            break
            
        # Random initial set
        # Start with larger sets to have more room to prune
        size = random.randint(30, 60)
        if size > 256: size = 256
        random_set = random.sample(range(1, 257), size)
        
        current_set = random_set
        sums_by_k = compute_sums_by_cardinality(current_set)
        current_collisions = count_collisions(sums_by_k)
        
        # Phase 1: Remove elements to reduce collisions
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            best_v = None
            best_collisions = current_collisions
            best_new_set = None
            best_new_sums = None
            
            n = len(current_set)
            if n == 0:
                break
                
            # Try removing each element
            # Optimization: If n is large, sampling might be better, but n<=60 is small
            for idx, v in enumerate(current_set):
                new_set = current_set[:idx] + current_set[idx+1:]
                new_sums = compute_sums_by_cardinality(new_set)
                new_collisions = count_collisions(new_sums)
                
                if new_collisions < best_collisions:
                    best_collisions = new_collisions
                    best_v = v
                    best_new_set = new_set
                    best_new_sums = new_sums
                    improved = True
                    
            if improved:
                current_set = best_new_set
                sums_by_k = best_new_sums
                current_collisions = best_collisions
                
        # Phase 2: Add elements to maximize size
        if current_collisions == 0:
            # Try adding any remaining element
            remaining = [v for v in range(1, 257) if v not in current_set]
            # Sort remaining by some heuristic? 
            # Smaller numbers are harder to add? Or larger?
            # Let's just try all
            added = True
            while added and time.time() < deadline - 5:
                added = False
                for v in remaining:
                    if time.time() > deadline - 5:
                        break
                    valid, new_sums = try_add_element(current_set, v, sums_by_k)
                    if valid:
                        current_set.append(v)
                        remaining.remove(v)
                        sums_by_k = new_sums
                        added = True
                        break # Restart loop with new state
        
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(best_set)
            
    # Final check and return
    # Ensure types are correct
    return [int(x) for x in best_set]

