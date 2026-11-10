import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s buffer for 60s limit
    
    # Initial best known valid set
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_size = len(best_set)
    
    def compute_sums_by_cardinality(s):
        """
        Compute sums_by_k for a given set s using bitsets for speed.
        Returns a list of sets.
        """
        n = len(s)
        if n == 0:
            return [{0}]
        
        max_sum = sum(s)
        # Use list of bitsets (Python integers) where bit i is set if sum i is achievable
        # sums_by_k[k] is a bitset
        
        sums_by_k = [0] * (n + 1)
        sums_by_k[0] = 1  # bit 0 is set (sum 0)
        
        for v in s:
            # Update in reverse order
            for k in range(n, 0, -1):
                sums_by_k[k] |= (sums_by_k[k-1] << v)
        
        # Convert bitsets to sets for intersection counting
        result = [set() for _ in range(n + 1)]
        for k in range(n + 1):
            mask = sums_by_k[k]
            # Extract set bits
            # Efficient way: while mask: lsb = mask & -mask; idx = lsb.bit_length()-1; add idx; mask ^= lsb
            # But for large max_sum, this might be slow if many bits.
            # Alternative: just check each possible sum? No, bitset is faster for intersection.
            # Let's keep them as bitsets for collision counting.
            # Wait, find_collisions uses set intersection.
            # Let's return bitsets and modify find_collisions to use bitwise AND.
            
        return sums_by_k # Return bitsets

    def count_collisions_bitset(sums_by_k):
        """
        Count total cross-cardinality collisions using bitwise AND.
        """
        n = len(sums_by_k) - 1
        total = 0
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                inter = sums_by_k[i] & sums_by_k[j]
                if inter:
                    # Count bits in inter
                    total += bin(inter).count('1')
        return total

    def global_conflict_removal_bitset(initial_set, time_limit):
        """
        Start with initial_set, iteratively remove the element whose removal
        reduces the total number of cross-cardinality collisions the most.
        Uses bitset DP for speed.
        """
        current_set = list(initial_set)
        sums_by_k = compute_sums_by_cardinality(current_set)
        current_collisions = count_collisions_bitset(sums_by_k)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_idx = -1
            best_collisions = current_collisions
            best_new_set = None
            best_new_sums = None
            
            n = len(current_set)
            if n == 0:
                break
                
            # For each element, try removing it
            for idx in range(n):
                v = current_set[idx]
                new_set = current_set[:idx] + current_set[idx+1:]
                new_sums = compute_sums_by_cardinality(new_set)
                new_collisions = count_collisions_bitset(new_sums)
                
                if new_collisions < best_collisions:
                    best_collisions = new_collisions
                    best_idx = idx
                    best_new_set = new_set
                    best_new_sums = new_sums
                    improved = True
                    
            if improved:
                current_set = best_new_set
                sums_by_k = best_new_sums
                current_collisions = best_collisions
                
        return current_set, current_collisions

    def is_valid_set(s):
        """
        Check if the set is valid (no cross-cardinality sum collisions).
        """
        sums_by_k = compute_sums_by_cardinality(s)
        return count_collisions_bitset(sums_by_k) == 0

    # Strategy 1: Greedy construction
    # Start with empty set, add elements that maintain validity.
    # Order candidates by some heuristic (e.g., larger values first to spread sums).
    
    def greedy_construct():
        current = []
        candidates = list(range(1, 257))
        # Sort candidates? Maybe larger first helps separate layers?
        # Or smaller first?
        # Let's try a few orderings.
        
        # Ordering 1: Descending
        for order in [sorted(candidates, reverse=True), sorted(candidates)]:
            curr = []
            for v in order:
                # Check if adding v keeps it valid
                new_curr = curr + [v]
                if is_valid_set(new_curr):
                    curr = new_curr
                if time.time() > deadline - 10:
                    break
            if len(curr) > best_size:
                return curr
        return current # fallback

    # Try greedy construction
    g_set = greedy_construct()
    if len(g_set) > best_size:
        best_set = g_set
        best_size = len(best_set)

    # Strategy 2: Global conflict removal from random large sets
    # This is often more effective for finding larger valid sets if they exist.
    
    random.seed(42)
    
    # Try different sizes and random samples
    # Start with large sets (e.g., 30-50) and remove conflicts.
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
            
        size = random.randint(25, 45)
        if size > 256: size = 256
        random_set = random.sample(range(1, 257), size)
        
        # Run global conflict removal with a time limit per attempt
        # We want to leave some time for other attempts
        # Total time budget for this phase: ~40s
        # 50 attempts * 0.8s each = 40s
        
        result_set, final_collisions = global_conflict_removal_bitset(random_set, time.time() + 0.8)
        
        if final_collisions == 0:
            if len(result_set) > best_size:
                best_set = result_set
                best_size = len(result_set)
                
    # Strategy 3: Local search / Swap
    # If we have a valid set, try to improve it by swapping elements or adding new ones.
    # But we need to be careful not to break validity.
    
    # Let's try to extend the best set found so far by trying to add elements.
    # We can use a "repair" strategy: add an element, if invalid, remove the most conflicting element.
    
    def extend_set(current_set):
        """
        Try to add elements to current_set.
        If adding an element causes a conflict, remove the element that minimizes total collisions.
        """
        curr = list(current_set)
        candidates = [v for v in range(1, 257) if v not in curr]
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            best_new_set = None
            best_new_collisions = None
            
            # Compute current collisions (should be 0 if valid)
            sums_curr = compute_sums_by_cardinality(curr)
            curr_coll = count_collisions_bitset(sums_curr)
            
            for v in candidates:
                new_set = curr + [v]
                new_sums = compute_sums_by_cardinality(new_set)
                new_coll = count_collisions_bitset(new_sums)
                
                if new_coll == 0:
                    # Valid extension
                    if best_new_set is None or len(new_set) > len(best_new_set):
                        best_new_set = new_set
                        best_new_collisions = 0
                        improved = True
                        # We can break early if we found a valid extension?
                        # No, we might find a better one? No, size is same (len(curr)+1).
                        # So first valid one is fine.
                        break
                else:
                    # Invalid. Try to repair by removing one element from new_set.
                    # This is expensive. Let's just skip for now or do a limited repair.
                    # For now, just skip invalid additions.
                    pass
            
            if improved:
                curr = best_new_set
                # Update candidates
                candidates = [v for v in range(1, 257) if v not in curr]
        
        return curr

    # Try to extend the best set found
    if best_size > 0:
        extended = extend_set(best_set)
        if len(extended) > best_size:
            best_set = extended
            best_size = len(best_set)

    # Ensure types are correct
    return [int(x) for x in best_set]
