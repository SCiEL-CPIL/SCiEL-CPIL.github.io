import random
import time

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    prescribed = [1, 2, 5, 11]
    max_val = 284
    
    best_size = 0
    best_set = set(prescribed)
    
    # Helper to check validity and count conflicts
    def get_conflicts(s):
        sums = {}
        conflicts = 0
        items = list(s)
        items.sort()
        n = len(items)
        for i in range(n):
            for j in range(i, n):
                val = items[i] + items[j]
                if val in sums:
                    conflicts += 1
                else:
                    sums[val] = True
        return conflicts

    # We try to find a set of size N
    # We will iterate N from high to low
    # For each N, we run a local search
    
    def find_set_of_size(N, time_limit):
        # Initialize with prescribed elements and random others
        # To ensure diversity, we can use different random seeds or strategies
        
        # Strategy: Start with a greedy set, then improve
        # Or start with random
        
        # Let's use a hybrid: start with a random set, then local search
        # Local search move: swap one non-prescribed element with a random outside element
        
        p_set = set(prescribed)
        candidates = [x for x in range(1, max_val + 1) if x not in p_set]
        
        # We need N - len(p_set) random elements
        if N < len(p_set):
            return None
        
        num_random = N - len(p_set)
        if num_random > len(candidates):
            return None
            
        current_set = set(prescribed)
        current_set.update(random.sample(candidates, num_random))
        
        # Precompute conflicts for current set
        # To be efficient, we should track sums incrementally, but for N<=20, 
        # full recalculation is O(N^2) = 400 ops, which is fast enough.
        
        current_conflicts = get_conflicts(current_set)
        
        if current_conflicts == 0:
            return current_set
        
        # Local Search / Simulated Annealing
        # We'll use a simple hill climbing with random restarts if stuck, 
        # or SA to escape local minima.
        
        # Given the small N, a simple randomized hill climbing with "kick" 
        # (random perturbation) works well.
        
        t = time.time()
        max_iter = 1000000  # Cap iterations to avoid infinite loops if time check is slow
        
        # We will use a temperature-like variable for SA
        # Or just random restarts of hill climbing
        
        # Let's do a few restarts
        max_restarts = 50
        
        for r in range(max_restarts):
            # Reset to a new random state? Or continue from current?
            # Let's try to continue from current best so far for this N
            # But if stuck, restart from scratch
            
            # For simplicity and robustness, let's do a SA loop
            
            T = 10.0
            cooling = 0.999
            min_T = 0.01
            
            # Reset state for this restart? 
            # If we restart, we pick a new random set.
            if r > 0:
                 current_set = set(prescribed)
                 current_set.update(random.sample(candidates, num_random))
                 current_conflicts = get_conflicts(current_set)
                 if current_conflicts == 0:
                     return current_set
                 T = 10.0
            
            # SA Loop
            while T > min_T and (time.time() - t) < time_limit:
                # Pick a random non-prescribed element in the set
                non_prescribed_in_set = [x for x in current_set if x not in p_set]
                if not non_prescribed_in_set:
                    break # Should not happen if N > 4
                
                x_old = random.choice(non_prescribed_in_set)
                
                # Pick a random element outside the set
                # To be efficient, we can't generate all outside elements each time.
                # But max_val is 284, so it's small.
                outside = [x for x in range(1, max_val + 1) if x not in current_set]
                if not outside:
                    break
                x_new = random.choice(outside)
                
                # Apply move
                current_set.remove(x_old)
                current_set.add(x_new)
                
                new_conflicts = get_conflicts(current_set)
                delta = new_conflicts - current_conflicts
                
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    current_conflicts = new_conflicts
                else:
                    # Revert
                    current_set.remove(x_new)
                    current_set.add(x_old)
                
                T *= cooling
                
                if current_conflicts == 0:
                    return current_set
                
                # Safety check
                if time.time() - t > time_limit:
                    break
                    
        return None

    # Try sizes from 20 down to 10
    for N in range(20, 9, -1):
        if time.time() > deadline:
            break
            
        # Allocate remaining time to this size
        # Give each size a chunk of time, but prioritize larger N?
        # Actually, if we find a valid set of size N, we return it immediately?
        # No, we want the MAXIMUM size.
        # So we should try 20 first. If we find one, return it.
        # If we don't find one within a time budget, try 19, etc.
        
        # How much time per N?
        # Total 115s. 10 sizes. ~11s each?
        # But finding 20 is harder.
        # Let's give 20 more time.
        
        time_budget = 10.0
        if N == 20:
            time_budget = 20.0
        elif N == 19:
            time_budget = 15.0
        elif N == 18:
            time_budget = 10.0
        else:
            time_budget = 5.0
            
        # Check remaining time
        remaining = deadline - time.time()
        if remaining <= 0:
            break
            
        actual_budget = min(time_budget, remaining)
        
        found_set = find_set_of_size(N, actual_budget)
        
        if found_set is not None:
            # We found a valid set of size N
            # Since we are iterating from high to low, this is the best possible
            # (unless we can find a larger one, but we started at 20)
            return list(found_set)
            
        # If not found, continue to N-1
        
    # If no large set found, return the best we have (prescribed set)
    # The seed program returns a list.
    return list(best_set)
