"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    # Fallback known valid set of 16
    fallback_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Candidates are 2..40
    candidates = list(range(2, 41))
    
    # Sort candidates: prefer prime powers and smaller numbers first?
    # Actually, sparsity is key.
    # We'll just iterate through candidates in a randomized order with restarts.
    
    def can_add(current_prods, x):
        # Check if adding x to the set of numbers corresponding to current_prods
        # creates a collision.
        # current_prods is the set of all subset products of the current set.
        # New products are P * x for P in current_prods.
        # If any P * x is in current_prods, it's a conflict.
        # Also, P * x must be distinct from Q * x for P != Q, which is true if P != Q.
        # So we only need to check intersection with current_prods.
        for p in current_prods:
            if p * x in current_prods:
                return False
        return True

    def search_for_size(target_size):
        # Backtracking search
        # We maintain the list of selected numbers and the set of their subset products.
        # To speed up, we use a recursive function.
        
        # Heuristic: sort candidates by number of prime factors (ascending)
        # This favors pure powers.
        
        def prime_factor_count(n):
            count = 0
            d = 2
            while d * d <= n:
                while n % d == 0:
                    count += 1
                    n //= d
                d += 1
            if n > 1:
                count += 1
            return count

        sorted_candidates = sorted(candidates, key=lambda x: (prime_factor_count(x), x))
        
        # We will try to build a set of size target_size.
        # We use a recursive backtracker.
        # To avoid deep recursion and overhead, we might use a greedy approach with randomization.
        
        best_set = []
        
        def backtrack(idx, current_set, current_prods):
            nonlocal best_set
            if len(current_set) >= target_size:
                if len(current_set) > len(best_set):
                    best_set = current_set[:]
                return True
            
            if time.time() > deadline:
                return False
                
            # Try adding candidates from idx to end
            # Optimization: if remaining candidates + current size < target_size, prune
            remaining = len(sorted_candidates) - idx
            if len(current_set) + remaining < target_size:
                return False

            for i in range(idx, len(sorted_candidates)):
                x = sorted_candidates[i]
                if can_add(current_prods, x):
                    # Generate new products
                    new_prods = set(current_prods)
                    for p in current_prods:
                        new_prods.add(p * x)
                    
                    # Recurse
                    if backtrack(i + 1, current_set + [x], new_prods):
                        return True
            
            return False

        # Initialize
        backtrack(0, [], {1})
        
        return best_set

    # Try to find 18
    # Then 17
    # If not found, return fallback
    
    # Since search can be expensive, we limit restarts or time per attempt.
    # For now, try 17 first as it's more likely.
    
    # Randomize candidate order for multiple attempts?
    # The sorted order above is deterministic.
    # Let's try a few random permutations if the first fails.
    
    found_set = search_for_size(17)
    
    if len(found_set) < 17 and time.time() < deadline - 10:
        # Try 18 with a different heuristic or randomization
        # Actually, 18 is very hard. Let's stick to 17 if possible.
        # If 17 failed, try 18 with a quick attempt.
        pass

    if len(found_set) >= 17:
        # Verify and return
        # The search ensures validity, but a final check is cheap.
        return found_set
    else:
        return fallback_set

# EVOLVE-BLOCK-END
