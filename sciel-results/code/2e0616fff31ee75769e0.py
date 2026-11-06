import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def try_add(x, S_set):
        """Check if x can be added to S_set while keeping it sum-free. Returns True if valid."""
        if x + x in S_set:
            return False
        for a in S_set:
            if x + a in S_set:
                return False
            b = x - a
            if b in S_set:
                return False
        return True

    def local_search(S, initial, time_limit):
        """
        Local search to improve sum-free subset size.
        Uses add and swap moves.
        """
        current = list(initial)
        current_set = set(current)
        best = list(current)
        best_size = len(current)
        
        S_list = S
        no_improve_count = 0
        max_no_improve = 500  # Increased patience to allow more exploration
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try to add any element
            # We prioritize adding elements that might "unlock" others? 
            # For now, random shuffle is a decent heuristic for exploration.
            candidates = [x for x in S_list if x not in current_set]
            if candidates:
                random.shuffle(candidates)
                for add in candidates:
                    if try_add(add, current_set):
                        current.append(add)
                        current_set.add(add)
                        if len(current) > best_size:
                            best = list(current)
                            best_size = len(current)
                        improved = True
                        no_improve_count = 0
                        break
            
            if improved:
                continue
            
            # 2. Try swap: remove one, add one (net size change 0, but might enable future adds)
            # Or remove one, add two? 
            # Standard local search for max independent set / max sum-free often uses swaps.
            # If we can't add, we try to swap an element out for another that allows an add.
            
            # To make this efficient, we sample removal candidates.
            # If n is small, we can iterate all.
            rem_candidates = list(current)
            random.shuffle(rem_candidates)
            
            for rem in rem_candidates:
                if time.time() > time_limit:
                    break
                
                trial_set = set(current_set)
                trial_set.discard(rem)
                
                # Try to add any candidate to the trial set
                add_candidates = [x for x in S_list if x not in trial_set]
                # Heuristic: prioritize candidates that are "compatible" with many remaining elements?
                # For now, random.
                random.shuffle(add_candidates)
                
                added_any = False
                for add in add_candidates:
                    if try_add(add, trial_set):
                        # Perform swap
                        # Update current
                        current = [x for x in current if x != rem]
                        current.append(add)
                        current_set = set(current)
                        
                        # Check if we can add MORE now
                        # Loop back to "Try to add" logic essentially, 
                        # but since we broke out of the main loop structure, 
                        # we just mark improved and let the main loop handle the next add.
                        improved = True
                        no_improve_count = 0
                        added_any = True
                        break
                
                if improved:
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                    
        return best

    def solve_instance(S, time_limit):
        n = len(S)
        best = []
        best_size = 0
        
        # Strategy 1: Odds (Strong structural prior)
        odds = [x for x in S if x % 2 != 0]
        if len(odds) > best_size:
            best = odds
            best_size = len(odds)
            
        # Strategy 2: Positives upper half
        pos = sorted([x for x in S if x > 0])
        if pos:
            half = len(pos) // 2
            pos_large = pos[half:]
            if len(pos_large) > best_size:
                best = pos_large
                best_size = len(pos_large)
                
        # Strategy 3: Negatives lower half
        neg = sorted([x for x in S if x < 0])
        if neg:
            half = len(neg) // 2
            neg_small = neg[:half]
            if len(neg_small) > best_size:
                best = neg_small
                best_size = len(neg_small)
                
        # Strategy 4: Random Greedy restarts
        # Allocate some time for these
        greedy_time_limit = time.time() + min(2.0, (time_limit - time.time()) * 0.1)
        
        while time.time() < greedy_time_limit:
            order = list(S)
            random.shuffle(order)
            current = []
            current_set = set()
            for x in order:
                if try_add(x, current_set):
                    current.append(x)
                    current_set.add(x)
            if len(current) > best_size:
                best = current
                best_size = len(current)
                
        # Strategy 5: Local search from best solution
        # Allocate remaining time to local search
        # We might do multiple local search runs from different starts if time permits
        
        remaining_time = time_limit - time.time()
        
        # Run local search from the current best
        if remaining_time > 0.5:
            improved = local_search(S, best, time_limit - 0.5) # Leave 0.5s buffer
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
                
        # If there's still significant time left, try local search from other diverse starts
        # e.g., from the odds set, from the pos_large set
        # But local_search already does add/swap, so starting from a good structural prior is key.
        
        # Let's do a few more local search restarts if time allows
        # We need to be careful not to exceed time_limit.
        
        time_left = time_limit - time.time()
        
        # Restart from Odds
        if time_left > 5.0:
            odds = [x for x in S if x % 2 != 0]
            improved = local_search(S, odds, time_limit - 2.0)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
                
        # Restart from Positives
        time_left = time_limit - time.time()
        if time_left > 5.0:
            pos = sorted([x for x in S if x > 0], reverse=True)
            # Greedy first to get a valid set
            current = []
            current_set = set()
            for x in pos:
                if try_add(x, current_set):
                    current.append(x)
                    current_set.add(x)
            improved = local_search(S, current, time_limit - 1.0)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
                
        return best

    # Solve all three instances
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    # Allocate time proportionally to size, but with a minimum
    # 60, 120, 240 -> sum = 420
    # Weights: 1, 2, 4 ? Or just proportional to size?
    # Let's use proportional to size for fairness, but ensure small ones get enough time.
    total_size = sum(s for s, _ in instances)
    time_budget = DEADLINE - (time.time() - start_time) - 2.0 # 2s buffer for verification/overhead
    
    for i, (size, seed) in enumerate(instances):
        S = instance(size, seed)
        
        # Calculate time slice
        # Proportional: size / total_size
        # But maybe give the largest one more?
        # Let's stick to proportional for now, it's simple.
        # Actually, local search scales superlinearly? 
        # Let's just divide remaining time equally? 
        # No, larger sets need more search.
        # Let's use square root of size?
        # Let's just use size.
        
        # Dynamic allocation: 
        # If we have done 0 instances, give 1/3? 
        # Simpler: give each instance a slice of the total time_budget based on its size.
        # But we need to account for time already spent.
        
        # Let's just split the remaining time (time_budget) by the number of remaining instances,
        # but weighted by size.
        
        # Actually, simpler and robust:
        # Assign fixed fractions?
        # 60: 20%, 120: 30%, 240: 50%
        
        if i == 0:
            fraction = 0.2
        elif i == 1:
            fraction = 0.3
        else:
            fraction = 0.5
            
        instance_time_limit = start_time + time_budget * fraction
        
        # Ensure we don't go past the global deadline
        if instance_time_limit > start_time + DEADLINE - 1.0:
            instance_time_limit = start_time + DEADLINE - 1.0
            
        # Ensure minimum time
        if instance_time_limit < time.time() + 2.0:
            instance_time_limit = time.time() + 2.0

        result = solve_instance(S, instance_time_limit)
        results.append(result)
        
        # Check time
        if time.time() > start_time + DEADLINE - 2.0:
            # Out of time, break and use quick fallback for remaining
            break
            
    # Fill any missing results with quick greedy
    while len(results) < len(instances):
        size, seed = instances[len(results)]
        S = instance(size, seed)
        # Quick greedy: Odds
        result = [x for x in S if x % 2 != 0]
        results.append(result)

    # Final verification
    for i, result in enumerate(results):
        if not is_sum_free(result):
            S = instance(instances[i][0], instances[i][1])
            # Fallback: Odds
            results[i] = [x for x in S if x % 2 != 0]
            # Verify odds again? Odds are always sum-free.
            # a+b=c with a,b,c odd -> odd+odd=even != odd.
            # So odds are always sum-free.

    return results
