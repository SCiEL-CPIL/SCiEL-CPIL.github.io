import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    N = 1000
    best_size = 0
    best_set = []
    
    # We represent the set as a boolean array for fast membership and sum computation
    # However, for the specific constraint "at most 2 representations", 
    # maintaining exact counts is crucial.
    # We will use a dictionary for sum_counts for O(1) updates, but 
    # we will use a different search strategy: 
    # 1. Start with a random subset of size ~700 (denser than greedy typically finds)
    # 2. Iteratively find "violating" sums (count > 2)
    # 3. Remove an element involved in the most violations to repair the set
    # 4. Try to add random elements back if possible (reconstruction)
    # This "repair" phase is the key difference from pure greedy.
    
    def get_sum_counts(S):
        """Compute sum counts for a given set S. O(|S|^2) but used sparingly or incrementally."""
        counts = {}
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            # a + a
            s = 2 * a
            counts[s] = counts.get(s, 0) + 1
            # a + b for b > a
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return counts

    def is_valid(S, sum_counts):
        """Check if all counts are <= 2."""
        for v in sum_counts.values():
            if v > 2:
                return False
        return True

    def repair(S, sum_counts):
        """
        Remove elements to make the set valid.
        Strategy: While there is a sum with count > 2, 
        identify all pairs (a,b) in S that sum to that value.
        Remove the element that is part of the most such "bad" pairs.
        """
        S = set(S)
        # Rebuild sum_counts if it's not passed correctly or to ensure consistency
        # For efficiency, we pass it, but we must update it as we remove.
        
        while True:
            # Find a violating sum
            violating_sum = None
            max_count = 0
            for s, c in sum_counts.items():
                if c > 2:
                    violating_sum = s
                    max_count = c
                    break
            
            if violating_sum is None:
                break
            
            # Find all pairs in S that sum to violating_sum
            # Note: We need to be careful with a <= b.
            # Pairs are {a, b} such that a + b = violating_sum and a, b in S.
            # If a == b, it's a single element contributing 1 to the count.
            # If a != b, it's a pair.
            
            elements_in_conflict = {}
            for a in S:
                b = violating_sum - a
                if b in S and a <= b:
                    # This pair (a,b) contributes to the count
                    # Increment conflict count for a and b
                    elements_in_conflict[a] = elements_in_conflict.get(a, 0) + 1
                    if a != b:
                        elements_in_conflict[b] = elements_in_conflict.get(b, 0) + 1
            
            if not elements_in_conflict:
                # Should not happen if violating_sum is in sum_counts
                break
            
            # Pick the element with the highest conflict count
            # Tie-breaker: random to avoid determinism loops
            max_conflicts = max(elements_in_conflict.values())
            candidates = [x for x, c in elements_in_conflict.items() if c == max_conflicts]
            x_remove = random.choice(candidates)
            
            # Remove x_remove from S and update sum_counts
            S.discard(x_remove)
            for y in S:
                s = x_remove + y
                if s in sum_counts:
                    sum_counts[s] -= 1
                    if sum_counts[s] == 0:
                        del sum_counts[s]
            # Handle x_remove + x_remove
            s_double = 2 * x_remove
            if s_double in sum_counts:
                sum_counts[s_double] -= 1
                if sum_counts[s_double] == 0:
                    del sum_counts[s_double]
            
            if time.time() > deadline:
                break
        
        return S, sum_counts

    def try_add(S, sum_counts, candidate):
        """Try to add candidate to S. Returns True if successful."""
        if candidate in S:
            return False
        
        # Check if adding candidate violates constraints
        # New sums created: candidate + y for y in S, and candidate + candidate
        for y in S:
            s = candidate + y
            if sum_counts.get(s, 0) >= 2:
                return False
        s_double = 2 * candidate
        if sum_counts.get(s_double, 0) >= 2:
            return False
        
        # It's safe to add
        S.add(candidate)
        sum_counts[s_double] = sum_counts.get(s_double, 0) + 1
        for y in S:
            if y == candidate:
                continue
            s = candidate + y
            sum_counts[s] = sum_counts.get(s, 0) + 1
        return True

    # Main Loop: Multiple restarts with Repair-Then-Grow strategy
    while time.time() < deadline:
        # 1. Initialize with a random set. 
        # Density estimate: If we pick 500 random numbers, sums are roughly uniform?
        # Let's start with a moderately dense random set, say 400-500 elements.
        # Greedy usually gets ~600-700. Random repair might start lower but recover?
        # Actually, let's start with a random set of size 600.
        initial_size = 600
        S = set(random.sample(range(1, N + 1), initial_size))
        
        # 2. Compute initial sum counts
        sum_counts = get_sum_counts(S)
        
        # 3. Repair phase: Remove elements to satisfy constraints
        S, sum_counts = repair(S, sum_counts)
        
        current_size = len(S)
        
        # 4. Grow phase: Try to add more elements
        # We can do this in a few passes.
        # To speed up, we can keep a list of candidates not in S.
        candidates = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(candidates)
        
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            # Try to add as many as possible in this pass
            # We iterate through candidates. If we add one, the list of valid candidates changes.
            # For simplicity and speed, we just iterate once.
            for c in candidates:
                if c in S:
                    continue
                if try_add(S, sum_counts, c):
                    added_any = True
                    # Note: We don't remove c from candidates list immediately 
                    # because try_add checks membership. 
                    # But we should probably break or continue carefully.
                    # Since try_add is O(|S|), and we do this for many candidates, 
                    # it can be slow. 
                    # Optimization: If we add one, the set changes. 
                    # Let's just do one pass.
        
        # 5. Second Repair? (Sometimes adding elements creates new conflicts if we didn't check perfectly? 
        # No, try_add ensures validity. So S is valid.)
        
        # 6. Local Search / Swap to improve size?
        # Try to swap elements to allow more additions?
        # Simple swap: Remove a random element, see if we can add 2 new ones?
        # This is complex. Let's stick to the repair-grow loop.
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(S)
            
        # If we are close to deadline, stop
        if time.time() > deadline - 5:
            break

    return best_set
