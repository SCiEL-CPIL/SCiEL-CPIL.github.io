"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
import time

def construct():
    """
    Find a large Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses randomized backtracking with bitsets for sum tracking.
    """
    required = [1, 2, 5, 11]
    required_set = set(required)
    
    # Initialize used sums bitset
    # Sums range: 2 to 568
    def get_initial_sums(elements):
        sums = 0
        n = len(elements)
        for i in range(n):
            for j in range(i, n):
                sums |= 1 << (elements[i] + elements[j])
        return sums

    initial_sums = get_initial_sums(required)
    
    # Candidates
    all_candidates = [x for x in range(1, 285) if x not in required_set]
    
    best_result = list(required)
    best_size = len(required)
    
    def can_add(used_sums, current_elements, x):
        """
        Check if x can be added to current_elements without sum collision.
        Returns True if valid.
        """
        new_sums = 1 << (2 * x)
        for e in current_elements:
            new_sums |= 1 << (x + e)
        return (new_sums & used_sums) == 0

    def backtrack(current_elements, used_sums, candidates, target_size, start_time, time_limit):
        """
        Recursive backtracking.
        current_elements: list of selected elements
        used_sums: bitset of used sums
        candidates: list of remaining candidate values
        target_size: desired total size of the set
        """
        if time.time() - start_time > time_limit:
            return None
        
        if len(current_elements) == target_size:
            return list(current_elements)
        
        # Pruning: not enough candidates left
        needed = target_size - len(current_elements)
        if len(candidates) < needed:
            return None
        
        # Filter valid candidates
        valid_candidates = []
        for x in candidates:
            if can_add(used_sums, current_elements, x):
                valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        # Heuristic: Shuffle to randomize search path
        # Optionally, we could sort by some heuristic, e.g., value
        # Random is usually good for finding *a* solution if one exists
        random.shuffle(valid_candidates)
        
        # Try each valid candidate
        for i, x in enumerate(valid_candidates):
            new_elements = current_elements + [x]
            # Update used sums
            new_sums = used_sums
            new_sums |= 1 << (2 * x)
            for e in current_elements:
                new_sums |= 1 << (x + e)
            
            # New candidates: remove x from candidates
            # To avoid copying the whole list every time, we pass a filtered list
            # But filtering is O(N). N is ~280. Depth ~16. 280*16 is small.
            new_candidates = [c for c in candidates if c != x]
            
            result = backtrack(new_elements, new_sums, new_candidates, target_size, start_time, time_limit)
            if result is not None:
                return result
        
        return None

    # Try to find a set of size 20, then 19, etc.
    max_possible = 20
    min_possible = len(required)
    
    start_total_time = time.time()
    total_time_limit = 110  # Leave 10s for overhead/validation
    
    for target_size in range(max_possible, min_possible - 1, -1):
        if time.time() - start_total_time > total_time_limit:
            break
            
        # Try multiple restarts for this target size
        # The number of restarts can be adaptive
        restarts = 0
        max_restarts = 20 if target_size > 18 else 100
        
        while restarts < max_restarts:
            if time.time() - start_total_time > total_time_limit:
                break
            
            # Shuffle initial candidates to randomize the search space
            shuffled_candidates = list(all_candidates)
            random.shuffle(shuffled_candidates)
            
            result = backtrack(list(required), initial_sums, shuffled_candidates, target_size, time.time(), 1.0)
            
            if result is not None:
                best_result = result
                best_size = target_size
                break
            
            restarts += 1
        
        # If we found a solution for this size, we can stop (since we are going downwards)
        if best_size == target_size:
            break

    # Final validation and cleanup
    result = sorted(best_result)
    
    # Validate
    sums = set()
    valid = True
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    
    if not valid or not all(x in result for x in required):
        # Fallback to required set
        result = [1, 2, 5, 11]
        
    return result

# EVOLVE-BLOCK-END
