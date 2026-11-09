import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 55.0  # seconds, leave margin for overhead
    
    # We want to maximize size.
    # Start with a known valid set of size 9?
    # Or just start empty and grow.
    # Starting empty is safer for backtracking.
    
    max_val = 256
    candidates = list(range(1, max_val + 1))
    
    # We will try to find a set of size target_n.
    # We iterate target_n from high to low? No, we want the MAX.
    # So we try 12, then 11, etc?
    # If we find 12, we return it.
    # If we fail 12, we try 11.
    # This is efficient if 12 is impossible.
    # But if 12 is possible but hard to find, we might waste time.
    # Given the small search space (256 elements), backtracking should be fast if pruned well.
    
    def is_valid_addition(current_sums, candidate, current_size):
        """
        Check if adding 'candidate' to a set of size 'current_size' 
        with existing sums 'current_sums' results in a valid configuration.
        
        current_sums: list of sets. current_sums[k] is the set of sums for subsets of size k.
        Returns: new_sums if valid, None otherwise.
        """
        new_sums = [set(s) for s in current_sums]
        
        # Update sums
        # Adding candidate creates new subsets of size k+1 from old subsets of size k.
        # new_sums[k+1] = old_sums[k+1] U {s + candidate for s in old_sums[k]}
        
        # We must check collisions between new_sums[k+1] and new_sums[j] for j != k+1.
        # Note: Collisions within new_sums[k+1] are allowed.
        
        # Efficient check:
        # For each k from current_size down to 0:
        #   new_elements = {s + candidate for s in current_sums[k]}
        #   new_sums[k+1].update(new_elements)
        #   Check if any element in new_elements is in new_sums[j] for j != k+1.
        #   Actually, we can check against all existing sums in new_sums[j] for j != k+1.
        
        # Let's do it carefully.
        # We update new_sums in place.
        # We need to check collisions of the *newly added* elements against *all* other cardinalities.
        
        for k in range(current_size, -1, -1):
            new_elements = set()
            for s in current_sums[k]:
                ns = s + candidate
                new_elements.add(ns)
            
            # Check collisions with other cardinalities
            # The new elements go into cardinality k+1.
            # They must not collide with sums in cardinality j for j != k+1.
            
            # Note: new_sums[k+1] already contains old sums for k+1.
            # We are adding new_elements to it.
            # So we check new_elements against:
            # 1. new_sums[j] for j < k+1 (these are unchanged from current_sums, except if j=k? No, we go down)
            #    Wait, if we iterate k down, new_sums[k+1] is being built.
            #    Let's just check against the final new_sums sets?
            #    No, that's circular.
            
            # Better approach:
            # 1. Calculate all new sums for all cardinalities.
            # 2. Check for cross-cardinality collisions.
            
            pass # This logic is getting messy in-place. Let's do a clean check.
        
        # Clean check:
        # 1. Construct the full new_sums list.
        # 2. Check for collisions.
        
        new_sums = [set(s) for s in current_sums]
        for k in range(current_size, -1, -1):
            for s in current_sums[k]:
                new_sums[k+1].add(s + candidate)
        
        # Now check for collisions
        # For each pair of distinct cardinalities i, j, check if new_sums[i] & new_sums[j] is empty.
        # This is O(K^2 * |S|). K is small (<15). |S| is small.
        
        for i in range(len(new_sums)):
            for j in range(i+1, len(new_sums)):
                if new_sums[i] and new_sums[j]:
                    # Check intersection
                    # Using set intersection is fast
                    if new_sums[i] & new_sums[j]:
                        return None
        
        return new_sums

    def backtrack(depth, current_sums, used, target_size):
        if time.time() > deadline:
            return None
        
        if depth == target_size:
            return list(used)
        
        # Heuristic: Try candidates in random order to avoid systematic bias?
        # Or sorted?
        # Let's try random order for diversity.
        
        # To improve speed, we can filter candidates that are obviously too large or too small?
        # For now, try all unused candidates.
        
        candidates_to_try = [c for c in candidates if c not in used]
        random.shuffle(candidates_to_try)
        
        for c in candidates_to_try:
            new_sums = is_valid_addition(current_sums, c, depth)
            if new_sums is not None:
                used.add(c)
                result = backtrack(depth + 1, new_sums, used, target_size)
                if result is not None:
                    return result
                used.remove(c)
        
        return None

    # Try to find the largest set
    # Max possible size?
    # Theoretical bound: sums must be distinct.
    # Let's try from 12 down to 9.
    
    best_set = None
    
    # Initial state: empty set
    # current_sums[0] = {0}
    initial_sums = [set() for _ in range(15)] # Max size 14?
    initial_sums[0] = {0}
    
    # We will try target sizes from high to low.
    # If we find a valid set of size 12, we return it.
    # If not, try 11, etc.
    
    # To speed up, we can use a "warm start" or better heuristic.
    # But backtracking with 256 candidates might be slow if it explores many dead ends.
    # Let's limit the search depth or use a timeout.
    
    for target_n in range(12, 8, -1):
        if time.time() > deadline:
            break
            
        # Reset used
        used = set()
        
        # To avoid getting stuck in the same local minima, we can shuffle the global candidates list?
        # The shuffle is inside backtrack.
        
        result = backtrack(0, initial_sums, used, target_n)
        
        if result is not None:
            best_set = result
            break # Found the max for this target, which is the global max since we go down
    
    if best_set is None:
        # Fallback: Powers of 2
        best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Ensure types are correct
    return [int(x) for x in best_set]
