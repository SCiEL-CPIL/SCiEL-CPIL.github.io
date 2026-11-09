"""
Strategy: Spectral Decomposition + Simulated Annealing
1. Partition the input set into 'Large' (high magnitude) and 'Small' (low magnitude).
2. Initialize a sum-free set using a greedy approach on 'Large' elements first, 
   as large elements are less likely to form a+b=c with other large elements 
   unless signs are mixed carefully.
3. Refine the set using Simulated Annealing to escape local optima.
   - Moves: Add an element, remove an element, or swap.
   - Cost function: Penalty for violations (should be 0) + Reward for size.
   - Actually, we maintain a valid sum-free set. 
   - Moves: Try to add a random element. If it violates, try to remove a conflicting element.
   - Or: Start with a valid set, try to add elements. If blocked, try to remove one existing 
     element to allow addition of a potentially 'better' (larger) one? 
     Since we just maximize size, any addition is good.
     So, the search is: Can we add x? If not, which element y in current set conflicts with x?
     If we remove y, can we add x? This is a 1-swap move.
     We can also just try to add x if possible.
"""

import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_sum_free(values):
        if not values:
            return []
        
        n = len(values)
        # Sort by absolute value descending. 
        # Intuition: Large numbers are 'safer' to include because a+b=c 
        # requires c to be large. If a,b are large, c is very large, 
        # likely outside the set or less frequent.
        # Actually, if we pick all positive large numbers, a+b is even larger.
        # If we pick all negative large numbers, a+b is very negative.
        # The danger is mixing signs or small numbers.
        
        # Heuristic Initialization:
        # 1. Try taking all elements with |x| > threshold.
        # 2. Greedy add remaining.
        
        # Let's use a more robust heuristic:
        # Partition into Pos, Neg, Zero (no zero).
        # Sum-free sets often consist of all numbers in an interval (n/2, n] (positive) 
        # or similar structures.
        
        # Strategy A: Greedy by Magnitude
        # Sort by abs value descending
        sorted_vals = sorted(values, key=lambda x: abs(x), reverse=True)
        
        best_set = set()
        for x in sorted_vals:
            # Check if x can be added
            conflict = False
            for y in best_set:
                if x + y in best_set:
                    conflict = True
                    break
                if x + x in best_set:
                    conflict = True
                    break
            if not conflict:
                best_set.add(x)
        
        # Strategy B: Simulated Annealing / Local Search
        # We have a valid set S.
        # We want to maximize |S|.
        # Move 1: Add x not in S. If valid, do it.
        # Move 2: Remove y from S, Add x not in S. If valid, do it.
        # Move 3: Remove y from S. (Only if it allows adding multiple others? Hard to track).
        
        # Since we want max size, we should never remove unless we can add more.
        # So, focus on "Add if possible" and "Swap if beneficial".
        # Beneficial swap: Remove y, Add x. Net change 0. 
        # When is it beneficial? If removing y allows adding x AND z? 
        # Or if x is 'better' than y? But we only care about count.
        # So a swap is only useful if it enables further additions later.
        
        # Let's implement a simple local search:
        # 1. Try to add any element not in set.
        # 2. If stuck, try swaps: remove y, add x. If valid, accept with some probability (annealing).
        
        S = set(best_set)
        remaining = [x for x in values if x not in S]
        
        # Pre-compute conflicts? 
        # For n=240, checking validity is O(n^2). 
        # We can maintain a set of S for O(1) lookup.
        
        current_size = len(S)
        
        # Time budget for SA
        t_start = time.time()
        sa_limit = 20  # 20 seconds per instance? 
        # Total time 150s. 3 instances. 45s each is safe. 
        # Let's give 40s each.
        
        temp = 1.0
        cooling = 0.999
        
        # To speed up conflict check, we can precompute sums?
        # No, S changes.
        
        # Optimization: 
        # When adding x, we only need to check x+y in S for y in S.
        # And x+x in S.
        
        # Let's run a loop
        no_improve_count = 0
        max_no_improve = 5000 # Reset temp if stuck?
        
        while time.time() - t_start < sa_limit:
            if not remaining and not S:
                break
                
            move_made = False
            best_delta = 0
            best_move = None
            
            # 1. Try simple additions
            # Randomly sample from remaining to avoid O(N^2) scan every time?
            # Or scan all? N=240, S~100. 240*100 = 24k ops. Fast.
            
            # Let's try to find ANY valid addition first.
            added = False
            for x in remaining:
                valid = True
                for y in S:
                    if x + y in S or x + x in S:
                        valid = False
                        break
                if valid:
                    S.add(x)
                    remaining.remove(x)
                    added = True
                    break # Just one per iteration to keep it fast? Or all?
                    # If we add one, the set changes, so we should restart scan.
            
            if added:
                no_improve_count = 0
                continue # Restart loop with new state
                
            # 2. If no simple addition, try swaps
            # Pick a random x from remaining
            if remaining:
                x = random.choice(remaining)
                # Find conflicts: y in S such that x+y in S or x+x in S
                # If we remove y, does it resolve the conflict?
                # Conflict types:
                # a) x + y = z (z in S). Removing y or z helps.
                # b) x + x = z (z in S). Removing z helps.
                
                # Let's identify all y in S that block x.
                blockers = []
                for y in S:
                    if x + y in S:
                        blockers.append(y)
                        blockers.append(x + y) # The sum itself is also a blocker if we remove it
                    if x + x in S:
                        blockers.append(x + x)
                
                if not blockers:
                    # Should have been added in step 1
                    continue

                # Try removing one blocker b
                # If we remove b, is x valid?
                # We need to check if x conflicts with S \ {b}
                
                # To be efficient, we can just try a random blocker
                if blockers:
                    b = random.choice(blockers)
                    # Try swap: S' = S - {b} + {x}
                    # Check validity of x in S - {b}
                    valid = True
                    for y in S:
                        if y == b:
                            continue
                        if x + y in S:
                            # Wait, x+y might be b?
                            # If x+y == b, then b is in S, so x+y is in S. 
                            # But we are removing b. So x+y is NOT in S'.
                            # So we must check if x+y is in S AND x+y != b.
                            if x + y != b:
                                valid = False
                                break
                        if x + x in S:
                            if x + x != b:
                                valid = False
                                break
                    
                    if valid:
                        # Accept swap? 
                        # Size stays same. 
                        # Why do it? To break deadlock.
                        # Use Metropolis criterion? 
                        # Since size is same, we accept with prob exp(0/T) = 1?
                        # Or just accept to explore.
                        S.remove(b)
                        S.add(x)
                        remaining.remove(x)
                        remaining.append(b)
                        move_made = True
                        # Small temperature decay
                        temp *= cooling
                        no_improve_count += 1 # Size didn't increase
                        if no_improve_count > max_no_improve:
                            temp = 1.0 # Reheat
                            no_improve_count = 0
            else:
                # No remaining elements
                break
                
        return list(S)

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE:
            # Fallback: simple heuristic
            vals = instance(size, seed)
            # Just take positives?
            results.append([x for x in vals if x > 0])
            continue
        
        vals = instance(size, seed)
        
        # Time allocation: 
        # 60: 10s
        # 120: 30s
        # 240: 50s
        # Total 90s. Safe.
        
        # We modify solve_sum_free to take a time limit
        # Let's wrap it
        
        t_start = time.time()
        # Pass time limit
        # The function above uses a fixed 20s. 
        # Let's adjust it to use remaining time.
        
        # Refactor solve_sum_free to accept time_limit
        def solve_with_limit(values, time_limit):
            if not values:
                return []
            n = len(values)
            sorted_vals = sorted(values, key=lambda x: abs(x), reverse=True)
            
            S = set()
            for x in sorted_vals:
                conflict = False
                for y in S:
                    if x + y in S or x + x in S:
                        conflict = True
                        break
                if not conflict:
                    S.add(x)
            
            remaining = [x for x in values if x not in S]
            
            t_start = time.time()
            temp = 1.0
            cooling = 0.999
            no_improve_count = 0
            max_no_improve = 2000
            
            while time.time() - t_start < time_limit:
                added = False
                for x in remaining:
                    valid = True
                    for y in S:
                        if x + y in S or x + x in S:
                            valid = False
                            break
                    if valid:
                        S.add(x)
                        remaining.remove(x)
                        added = True
                        break
                
                if added:
                    no_improve_count = 0
                    continue
                
                if remaining:
                    x = random.choice(remaining)
                    blockers = []
                    for y in S:
                        if x + y in S:
                            blockers.append(y)
                            blockers.append(x + y)
                        if x + x in S:
                            blockers.append(x + x)
                    
                    if blockers:
                        b = random.choice(blockers)
                        valid = True
                        for y in S:
                            if y == b:
                                continue
                            if x + y in S and x + y != b:
                                valid = False
                                break
                            if x + x in S and x + x != b:
                                valid = False
                                break
                        
                        if valid:
                            S.remove(b)
                            S.add(x)
                            remaining.remove(x)
                            remaining.append(b)
                            temp *= cooling
                            no_improve_count += 1
                            if no_improve_count > max_no_improve:
                                temp = 1.0
                                no_improve_count = 0
                else:
                    break
                    
            return list(S)
        
        # Allocate time
        remaining_time = DEADLINE - time.time()
        # Distribute remaining time among remaining instances
        instances_left = 3 - len(results)
        time_per_instance = remaining_time / instances_left - 1 # 1s buffer
        
        sol = solve_with_limit(vals, time_per_instance)
        
        # Verify
        if not is_sum_free(sol):
            # Fallback
            sol = [x for x in vals if x > 0]
        
        results.append(sol)
        
    return results

# EVOLVE-BLOCK-END
