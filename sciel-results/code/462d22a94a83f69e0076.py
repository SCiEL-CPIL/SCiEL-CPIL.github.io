import time
import random

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
    
    def branch_and_bound(S, time_limit):
        """
        Find maximum sum-free subset using branch and bound.
        S is a sorted list of distinct integers.
        """
        n = len(S)
        S_set = set(S)
        
        # Precompute for each element, which other elements it "blocks"
        # If we include x, then for any a in our set, x+a cannot be included
        # and a+x cannot be included, and x+x cannot be included
        
        # Sort by absolute value or some heuristic
        # Actually, let's try sorting by value
        
        # Upper bound: for remaining elements, at most all can be added
        # Better bound: use the fact that sum-free sets have structure
        
        best_size = 0
        best_subset = []
        
        # Try the "large half" heuristic first: elements > n/2 in magnitude
        # Actually, a known result: the set of elements in (n/3, n] is sum-free
        # But our elements are arbitrary, not 1..n
        
        # Let's use a different approach: try to find a large sum-free subset
        # by considering the "upper half" - elements whose absolute value is large
        
        # Strategy: branch and bound with good ordering
        # Order elements so that "conflicting" elements are close together
        
        # For each element x, compute its "conflict degree"
        # x conflicts with y if there exists z in S such that x+y=z, or x+z=y, or y+z=x
        
        # Actually, let me use a simpler but effective approach:
        # 1. Try the set of all "positive large" elements
        # 2. Try the set of all "negative large" elements  
        # 3. Use local search to improve
        
        # Let me implement a proper branch and bound
        
        def can_add(x, current_set):
            """Check if x can be added to current_set while maintaining sum-free."""
            for a in current_set:
                if a + x in current_set:
                    return False
                if x + x in current_set:
                    return False
                # Check if x + a = b for some b in current_set
                # This is covered by a + x in current_set since addition is commutative
                # Wait, we need: for all a,b in current_set ∪ {x}, a+b not in current_set ∪ {x}
                # So we need: x + a not in current_set for all a in current_set
                # and x + x not in current_set
                # and a + b not in {x} for all a,b in current_set (but this is already satisfied if current_set is sum-free)
                # Actually no: we need a + b != x for all a,b in current_set
                # That's a different check!
                pass
            
            # Check: for all a, b in current_set, a + b != x
            for a in current_set:
                for b in current_set:
                    if a + b == x:
                        return False
            
            # Check: for all a in current_set, x + a not in current_set
            for a in current_set:
                if x + a in current_set:
                    return False
            
            # Check: x + x not in current_set
            if x + x in current_set:
                return False
            
            return True
        
        # This is O(k^2) per check where k is current set size, too slow
        # Let me use a set for current_set
        
        def can_add_fast(x, current_set, current_set_list):
            """Check if x can be added to current_set (a set) while maintaining sum-free."""
            # Check: x + x not in current_set
            if x + x in current_set:
                return False
            
            # Check: for all a in current_set, x + a not in current_set
            for a in current_set:
                if x + a in current_set:
                    return False
            
            # Check: for all a, b in current_set, a + b != x
            # This is expensive. Let me precompute or use a different approach
            # Actually, if current_set is sum-free, then a+b is never in current_set for a,b in current_set
            # But we need a+b != x, not a+b not in current_set
            # So I do need to check this
            
            # Alternative: maintain a set of all pairwise sums of current_set
            # But that changes as we add elements
            
            # For now, do the O(k^2) check but with early termination
            for a in current_set:
                for b in current_set:
                    if a + b == x:
                        return False
            
            return True
        
        # The O(k^2) check is too slow. Let me use a different strategy.
        # 
        # Key insight: if I maintain the set of all pairwise sums of the current set,
        # I can check in O(1) whether x is a sum of two elements.
        # But maintaining this set is O(k) per addition.
        
        # Let me try a different approach: use the fact that for random sets,
        # the "large elements" strategy works well.
        
        # Strategy 1: Take all elements with |x| > threshold
        # Strategy 2: Local search from good initial solutions
        
        # For local search, I'll use a faster violation check
        
        def count_violations(S):
            S_set = set(S)
            count = 0
            for a in S:
                for b in S:
                    if a + b in S_set:
                        count += 1
            return count
        
        def try_add(x, S_set):
            """Check if x can be added to S_set while keeping it sum-free. Returns True if valid."""
            # Check x + x
            if x + x in S_set:
                return False
            # Check x + a for a in S_set
            for a in S_set:
                if x + a in S_set:
                    return False
            # Check a + b = x for a, b in S_set
            # This requires checking all pairs, which is O(k^2)
            # But we can optimize: for each a in S_set, check if x - a is in S_set
            # and a != x - a (unless we allow a=b, which we do)
            for a in S_set:
                b = x - a
                if b in S_set:
                    # We found a + b = x, so x cannot be added
                    return False
            return True
        
        # This is O(k) per check! Much better.
        
        def greedy_solve(S, order):
            """Greedy: add elements in given order if they don't create violations."""
            result = []
            result_set = set()
            for x in order:
                if try_add(x, result_set):
                    result.append(x)
                    result_set.add(x)
            return result
        
        def local_search(S, initial, max_iters=10000):
            """Local search: try to improve by removing one element and adding another."""
            current = list(initial)
            current_set = set(current)
            best = list(current)
            best_size = len(best)
            
            for _ in range(max_iters):
                if time.time() > time_limit:
                    break
                
                improved = False
                
                # Try single additions
                candidates = [x for x in S if x not in current_set]
                random.shuffle(candidates)
                for add in candidates:
                    if try_add(add, current_set):
                        current.append(add)
                        current_set.add(add)
                        if len(current) > best_size:
                            best = list(current)
                            best_size = len(best)
                        improved = True
                        break
                
                if improved:
                    continue
                
                # Try single removals + additions (swap)
                # Remove one element, then try to add others
                for i in range(len(current)):
                    if time.time() > time_limit:
                        break
                    
                    rem = current[i]
                    trial = current[:i] + current[i+1:]
                    trial_set = set(trial)
                    
                    candidates = [x for x in S if x not in trial_set]
                    random.shuffle(candidates)
                    
                    added_any = False
                    for add in candidates:
                        if try_add(add, trial_set):
                            # Check if this improves size
                            if len(trial) + 1 > best_size:
                                new_current = trial + [add]
                                best = new_current
                                best_size = len(best)
                                current = new_current
                                current_set = set(current)
                                improved = True
                                added_any = True
                                break
                    
                    if improved:
                        break
                
                if not improved:
                    break
            
            return best
        
        # Generate multiple initial solutions and improve them
        best = []
        best_size = 0
        
        # Strategy 1: Greedy with random orderings
        for restart in range(50):
            if time.time() > time_limit:
                break
            order = list(S)
            random.shuffle(order)
            candidate = greedy_solve(S, order)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
        
        # Strategy 2: Greedy with "large absolute value" order
        order_large = sorted(S, key=lambda x: -abs(x))
        candidate = greedy_solve(S, order_large)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
        
        # Strategy 3: Greedy with "positive large" then "negative large"
        pos_large = sorted([x for x in S if x > 0], key=lambda x: -x)
        neg_large = sorted([x for x in S if x < 0], key=lambda x: x)
        order = pos_large + neg_large
        candidate = greedy_solve(S, order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
        
        # Strategy 4: Local search from best greedy solution
        if time.time() < time_limit - 5:
            candidate = local_search(S, best, max_iters=5000)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
        
        # Strategy 5: Multiple local search restarts from different greedy solutions
        for restart in range(10):
            if time.time() > time_limit - 2:
                break
            order = list(S)
            random.shuffle(order)
            candidate = greedy_solve(S, order)
            candidate = local_search(S, candidate, max_iters=2000)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
        
        return best
    
    # Solve all three instances
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        S = instance(size, seed)
        time_remaining = DEADLINE - (time.time() - start_time)
        if time_remaining < 5:
            # Not enough time, use quick greedy
            order = list(S)
            random.shuffle(order)
            result = []
            result_set = set()
            for x in order:
                if try_add(x, result_set):
                    result.append(x)
                    result_set.add(x)
            results.append(result)
        else:
            result = branch_and_bound(S, time.time() + time_remaining - 1)
            results.append(result)
    
    # Final verification
    for i, result in enumerate(results):
        if not is_sum_free(result):
            S = instance(instances[i][0], instances[i][1])
            results[i] = [x for x in S if x % 2 != 0]
    
    return results
