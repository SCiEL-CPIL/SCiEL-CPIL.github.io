import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    timeout = 58.0  # Safety margin for 60s limit
    
    candidates = list(range(1, 23))
    
    # Precompute reciprocal values as Fractions
    reciprocals = {x: Fraction(1, x) for x in candidates}
    
    best_set = []
    best_size = 0
    
    def is_dissociated(subset):
        """Check if a subset is dissociated."""
        sums = {Fraction(0)}
        for x in subset:
            v = reciprocals[x]
            # Check for collision: s + v in sums?
            # We can do this by generating new sums and checking intersection
            new_sums = {s + v for s in sums}
            if new_sums & sums:
                return False
            sums.update(new_sums)
        return True

    def try_build_greedy(order):
        """Build a set greedily following the given order."""
        current_set = []
        sums = {Fraction(0)}
        
        for x in order:
            v = reciprocals[x]
            # Check collision
            collision = False
            # Optimization: check intersection of sums and sums+v
            # This is O(|sums|)
            for s in sums:
                if (s + v) in sums:
                    collision = True
                    break
            
            if not collision:
                current_set.append(x)
                # Update sums
                # We must be careful to use the old sums for generation
                new_sums = [s + v for s in sums]
                sums.update(new_sums)
        
        return current_set

    # 1. Try several deterministic heuristics
    orders = [
        sorted(candidates, reverse=True), # Large denominators (small values) first
        sorted(candidates),               # Small denominators (large values) first
        [x for x in candidates if x > 10],
        [x for x in candidates if all(x % d != 0 for d in range(2, int(x**0.5)+1))], # Primes
        [x for x in candidates if x % 2 != 0], # Odds
    ]
    
    for order in orders:
        s = try_build_greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
        
        if time.time() - start_time > timeout:
            break

    # 2. Randomized local search / restarts
    # We want to maximize size.
    # We can try to improve the current best set by swapping elements.
    # Or just random restarts with greedy.
    
    # To find larger sets, we might need to be smarter than simple greedy.
    # Backtracking for specific sizes is more robust.
    
    def find_set_of_size(target_size, candidates_list):
        """
        Use backtracking to find a dissociated set of exactly target_size.
        candidates_list should be a list of indices/values to consider.
        """
        if not candidates_list:
            return [] if target_size == 0 else None
        
        # Pruning: if not enough candidates left
        if len(candidates_list) < target_size:
            return None
            
        # Heuristic: try candidates that are "independent" early?
        # Just iterate
        for i, x in enumerate(candidates_list):
            v = reciprocals[x]
            # Check if x can be added to an empty set? Yes.
            # We need to build the set.
            # This simple backtracking is exponential. 
            # Let's use a recursive helper.
            
            pass # Placeholder, will implement below if needed
        
        return None

    # Better backtracking:
    def backtrack(remaining, target_size, current_set, current_sums):
        if len(current_set) == target_size:
            return current_set[:]
        
        if not remaining or len(remaining) < target_size - len(current_set):
            return None
        
        # Heuristic: order remaining?
        # We'll just iterate through remaining
        for i, x in enumerate(remaining):
            v = reciprocals[x]
            
            # Check collision with current_sums
            collision = False
            for s in current_sums:
                if (s + v) in current_sums:
                    collision = True
                    break
            
            if not collision:
                # Add x
                new_set = current_set + [x]
                new_sums = set(current_sums)
                new_sums.update(s + v for s in current_sums)
                
                # Recurse
                result = backtrack(remaining[i+1:], target_size, new_set, new_sums)
                if result is not None:
                    return result
            else:
                # Skip x
                pass # The loop continues to next x, effectively skipping x in this branch
                
        return None

    # Try to find sets of size larger than best_size
    # Start from best_size + 1 up to 22
    # We need to be careful with time.
    # Backtracking can be very slow.
    # Let's limit the search depth or use it only for small improvements.
    
    # Since N=22 is small, we can try to find the global max using randomized greedy first,
    # then if we have time, try to prove optimality or find larger via backtracking.
    
    # Let's do more random restarts with different orderings
    for _ in range(200):
        if time.time() - start_time > timeout:
            break
            
        random.shuffle(candidates)
        s = try_build_greedy(candidates)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
            # If we found a very large set, we might stop early or try to verify
            if best_size >= 15:
                # High bar, likely optimal or near-optimal
                # We can break if we want, but let's keep checking a bit
                pass

    # If we have time, try to improve best_set via local search
    # Try swapping elements in best_set with elements not in best_set
    if best_size < 22 and time.time() - start_time < timeout - 5:
        current_best = set(best_set)
        improvements_found = True
        while improvements_found and time.time() - start_time < timeout - 2:
            improvements_found = False
            # Try adding an element not in current_best
            for x in candidates:
                if x in current_best:
                    continue
                
                # Check if adding x to current_best creates a collision
                # We need the sums of current_best
                # Re-calculating sums for every check is expensive.
                # Let's just re-run greedy on the combined set? No.
                
                # Efficient check:
                sums = {Fraction(0)}
                for y in current_best:
                    v = reciprocals[y]
                    new_sums = {s + v for s in sums}
                    # If collision, break
                    if new_sums & sums:
                        break
                    sums.update(new_sums)
                else:
                    # No collision in current_best (should be true)
                    # Now check adding x
                    v = reciprocals[x]
                    new_sums_with_x = {s + v for s in sums}
                    if not (new_sums_with_x & sums):
                        # We can add x!
                        current_best.add(x)
                        best_size += 1
                        best_set = list(current_best)
                        improvements_found = True
                        break # Restart loop with new best

    # Final fallback
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    return best_set
