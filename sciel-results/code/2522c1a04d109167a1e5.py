"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# EVOLVE-BLOCK-END

def construct():
    """
    Find 12 distinct positive integers with all subset sums distinct, minimizing the largest element.
    
    Strategy:
    1. Use greedy construction to get a good initial solution.
    2. Use local search to try to improve the largest element.
    3. Return the best solution found within the time budget.
    """
    start_time = time.time()
    deadline = 85  # seconds, with safety margin before 90s limit
    
    def subset_sums_set(a):
        """Compute all subset sums of list a as a set."""
        sums = {0}
        for x in a:
            new_sums = {s + x for s in sums}
            sums = sums | new_sums
        return sums
    
    def is_valid(a):
        """Check if all subset sums are distinct."""
        return len(subset_sums_set(a)) == (1 << len(a))
    
    def greedy_construct():
        """Greedy: add smallest possible element at each step."""
        result = []
        current_sums = {0}
        
        for i in range(12):
            candidate = 1
            while True:
                new_sums = current_sums | {s + candidate for s in current_sums}
                if len(new_sums) == (1 << (i + 1)):
                    result.append(candidate)
                    current_sums = new_sums
                    break
                candidate += 1
                if candidate > 1048576:
                    return None
                if time.time() - start_time > deadline:
                    return None
        return result
    
    # Step 1: Get a valid solution via greedy
    best = greedy_construct()
    if best is None:
        # Fallback: powers of 2
        return [1 << i for i in range(12)]
    
    best_max = max(best)
    best_sums = subset_sums_set(best)
    
    # Step 2: Try to improve via local search
    # We'll try to reduce the largest element by replacing it with smaller values
    # and checking validity
    
    def try_improve(a, max_steps=100000):
        """Try to improve the solution by local search."""
        current = list(a)
        current_max = max(current)
        current_sums = subset_sums_set(current)
        steps = 0
        
        while steps < max_steps:
            if time.time() - start_time > deadline:
                break
            steps += 1
            
            # Find the largest element
            max_idx = current.index(current_max)
            
            # Try to replace the largest element with smaller values
            # Start from current_max - 1 and go down
            for new_val in range(current_max - 1, 0, -1):
                if new_val in current:
                    continue
                
                # Check if replacing current_max with new_val keeps validity
                # Remove current_max's contribution from sums
                # This is complex, so just rebuild
                test = list(current)
                test[max_idx] = new_val
                test_sums = subset_sums_set(test)
                
                if len(test_sums) == (1 << 12):
                    # Valid! Update current
                    current = test
                    current_max = new_val
                    current_sums = test_sums
                    break
            else:
                # No improvement found with this largest element
                break
        
        return current, current_max
    
    # Try local search
    improved, improved_max = try_improve(best)
    if improved_max < best_max:
        best = improved
        best_max = improved_max
    
    # Step 3: Try a different approach - start with a random valid set and optimize
    # Actually, let's try a smarter local search: try to reduce each element
    
    def local_search_optimize(a, iterations=50000):
        """Local search to minimize the maximum element."""
        current = list(a)
        current_max = max(current)
        current_sums = subset_sums_set(current)
        
        for _ in range(iterations):
            if time.time() - start_time > deadline:
                break
            
            # Try to reduce a random element
            idx = random.randint(0, 11)
            old_val = current[idx]
            
            # Try smaller values
            for new_val in range(old_val - 1, 0, -1):
                if new_val in current:
                    continue
                
                test = list(current)
                test[idx] = new_val
                test_sums = subset_sums_set(test)
                
                if len(test_sums) == (1 << 12):
                    # Valid! Check if it improves the max
                    new_max = max(test)
                    if new_max < current_max:
                        current = test
                        current_max = new_max
                        current_sums = test_sums
                    break
        
        return current, current_max
    
    # Run local search optimization
    optimized, opt_max = local_search_optimize(best)
    if opt_max < best_max:
        best = optimized
        best_max = opt_max
    
    return best

# EVOLVE-BLOCK-END
