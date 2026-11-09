import time
from random import Random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
        
    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values):
        n = len(values)
        if n == 0:
            return []
        
        # Build conflict graph representation
        # conflict[i] is a set of indices j such that values[i] and values[j] cannot both be in a sum-free set.
        # Wait, the condition is "no a+b=c". 
        # If we pick a and b, we cannot pick c=a+b.
        # This is a 3-uniform hypergraph independent set problem, or we can view it as a graph if we fix one variable?
        # Actually, it's simpler to check validity incrementally.
        
        # Let's use a greedy + local search approach.
        # State: a set of indices included.
        
        # Precompute sums?
        # For a given set of included indices S, an index k is valid to add if 
        # for all i, j in S, values[i] + values[j] != values[k] AND values[i] + values[k] != values[j] AND values[k] + values[j] != values[i].
        
        # To speed up, we can maintain the set of values in S.
        
        best_set = set()
        
        # Greedy Initialization
        # Try a few random orderings
        rng = Random(42)
        for _ in range(10):
            order = list(range(n))
            rng.shuffle(order)
            current = set()
            current_vals = set()
            for idx in order:
                val = values[idx]
                # Check if val conflicts with any pair in current
                # Conflict if:
                # 1. val + x = y for x,y in current
                # 2. x + val = y for x,y in current
                # 3. x + y = val for x,y in current
                
                ok = True
                # Optimize: iterate through current
                # Since n is small, O(|current|^2) is fine for initialization
                for x in current_vals:
                    if val + x in current_vals:
                        ok = False
                        break
                    if x + val in current_vals: # same as above
                        pass
                    # Check if x + y = val
                    # This requires checking pairs in current_vals
                    # Let's do a separate check for sum-to-val
                if not ok:
                    continue
                # Check if any pair in current_vals sums to val
                # This is O(|current|^2)
                # We can optimize by keeping a set of sums?
                # For now, simple check.
                for x in current_vals:
                    for y in current_vals:
                        if x + y == val:
                            ok = False
                            break
                    if not ok:
                        break
                
                if ok:
                    current.add(idx)
                    current_vals.add(val)
            
            if len(current) > len(best_set):
                best_set = current
        
        best_vals = {values[i] for i in best_set}
        
        # Local Search
        # We try to improve the set.
        # Move: Remove one element, then greedily add as many as possible.
        # Or: Try to swap.
        
        # A better local search for this specific constraint:
        # 1. Identify "bad" elements? No, the set is valid.
        # 2. Try removing an element v. This might allow adding multiple new elements.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_set = set(best_set)
            current_vals = set(best_vals)
            
            # Try removing each element in the current set
            for v_idx in list(current_set):
                v_val = values[v_idx]
                
                # Remove v
                temp_set = current_set - {v_idx}
                temp_vals = current_vals - {v_val}
                
                # Greedily add elements not in temp_set
                # Sort candidates by some heuristic? Maybe by degree of conflict?
                # For now, just iterate through all not in temp_set
                # To make it deterministic and fast, we can just try adding all possible ones
                # that don't conflict.
                
                added_any = False
                # We need to check if adding x is valid given temp_set
                # valid(x) if no a,b in temp_vals s.t. a+b=x, a+x=b, b+x=a
                
                # To speed up, we can just try to add elements from the complement
                # in a random or sorted order.
                
                # Let's just try to add all valid elements from the complement
                # This might be slow if done naively for every removal.
                # Optimization: Only try removal if it potentially helps.
                # Given the small N, let's just do it.
                
                new_set = set(temp_set)
                new_vals = set(temp_vals)
                
                # Iterate through all indices not in new_set
                # Sort by value? Or just index order?
                # Let's use index order for determinism in the greedy fill
                for idx in range(n):
                    if idx in new_set:
                        continue
                    val = values[idx]
                    
                    # Check validity
                    ok = True
                    for x in new_vals:
                        if val + x in new_vals:
                            ok = False
                            break
                        # Check if x + y = val
                        # This inner loop is O(|new_vals|)
                        # Total O(|new_vals|^2) per check.
                        # With N=240, this is heavy.
                        # Let's optimize the check.
                    
                    if not ok:
                        continue
                    
                    # Check if sum of two elements in new_vals equals val
                    # We can precompute sums? No, new_vals changes.
                    # Let's just do the double loop but break early.
                    found_sum = False
                    for x in new_vals:
                        if val - x in new_vals:
                            found_sum = True
                            break
                    if found_sum:
                        continue
                        
                    new_set.add(idx)
                    new_vals.add(val)
                    added_any = True
                
                if len(new_set) > len(best_set):
                    best_set = new_set
                    best_vals = new_vals
                    improved = True
                    # Break out of removal loop to restart with new best?
                    # Or continue checking other removals?
                    # Restarting is safer to find local maxima.
                    break
            # End while loop if no improvement
        
        return [values[i] for i in sorted(best_set)]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Fallback to odd numbers if time runs out
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
            
        vals = instance(size, seed)
        # For small N, simple heuristic might be enough, but let's use the solver
        # If N is very small, we can even do exhaustive? No, 2^60 is too big.
        # The local search above should work.
        
        # Optimization: The local search above is O(N^3) roughly per iteration.
        # For N=240, this might be slow in Python.
        # Let's refine the local search to be faster or use a better heuristic.
        
        # Alternative: Use the "large interval" property?
        # Sum-free sets often contain large numbers.
        # A very strong heuristic: Take all numbers with absolute value > N/2?
        # If a, b > N/2, then a+b > N. If the max value is M, and a+b > M, it can't be in the set.
        # But the set is bounded by 10000, not N.
        
        # Let's stick to the local search but optimize the inner check.
        # We can represent the set as a boolean array or set of values.
        
        # Let's try a different, faster local search:
        # 1. Start with a greedy set.
        # 2. Repeat:
        #    - Try to add any element that fits.
        #    - If none, try removing one element and see if we can add more.
        
        # Given the time constraints and Python speed, let's use a slightly smarter greedy:
        # Sort by absolute value descending. Large numbers are less likely to be sums of two others in the set (if the set is sparse or large).
        
        def fast_solve(vals):
            n = len(vals)
            
            def can_add(val, s):
                """Check if val can be added to set s without violating sum-free property."""
                for x in s:
                    if val + x in s:
                        return False
                    if val - x in s:
                        return False
                return True
            
            def greedy_fill(order, start_set=None, start_vals=None):
                """Greedy fill given an order of indices."""
                if start_set is None:
                    current = set()
                    current_vals = set()
                else:
                    current = set(start_set)
                    current_vals = set(start_vals)
                
                for idx in order:
                    if idx in current:
                        continue
                    val = vals[idx]
                    if can_add(val, current_vals):
                        current.add(idx)
                        current_vals.add(val)
                return current, current_vals
            
            # Multiple greedy strategies
            strategies = []
            
            # Strategy 1: Descending absolute value
            order1 = sorted(range(n), key=lambda i: abs(vals[i]), reverse=True)
            strategies.append(order1)
            
            # Strategy 2: Positive values first (descending), then negative (ascending)
            pos_idx = [i for i in range(n) if vals[i] > 0]
            neg_idx = [i for i in range(n) if vals[i] < 0]
            order2 = sorted(pos_idx, key=lambda i: vals[i], reverse=True) + \
                     sorted(neg_idx, key=lambda i: vals[i])
            strategies.append(order2)
            
            # Strategy 3: Random shuffles
            rng = __import__('random').Random(12345)
            for _ in range(5):
                order3 = list(range(n))
                rng.shuffle(order3)
                strategies.append(order3)
            
            # Find best greedy solution
            best_set = set()
            best_vals = set()
            best_size = 0
            
            for order in strategies:
                if time.time() > deadline - 10:
                    break
                current, current_vals = greedy_fill(order)
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = current
                    best_vals = current_vals
            
            # Local search: try removing one element and refilling
            # Use multiple restarts with different removal orders
            improved = True
            iteration = 0
            max_iterations = 50 if n <= 100 else 25
            
            while improved and iteration < max_iterations and time.time() < deadline - 5:
                improved = False
                iteration += 1
                
                # Try each element for removal
                removal_order = list(best_set)
                if iteration % 2 == 0:
                    removal_order.reverse()
                
                for v_idx in removal_order:
                    if time.time() > deadline - 5:
                        break
                    
                    v_val = vals[v_idx]
                    temp_set = best_set - {v_idx}
                    temp_vals = best_vals - {v_val}
                    
                    # Refill with multiple strategies
                    best_new_set = set(temp_set)
                    best_new_vals = set(temp_vals)
                    best_new_size = len(temp_set)
                    
                    # Strategy A: fill in original index order
                    order_a = [i for i in range(n) if i not in temp_set]
                    new_set_a, new_vals_a = greedy_fill(order_a, temp_set, temp_vals)
                    if len(new_set_a) > best_new_size:
                        best_new_set = new_set_a
                        best_new_vals = new_vals_a
                        best_new_size = len(new_set_a)
                    
                    # Strategy B: fill by descending abs value
                    order_b = sorted([i for i in range(n) if i not in temp_set], 
                                    key=lambda i: abs(vals[i]), reverse=True)
                    new_set_b, new_vals_b = greedy_fill(order_b, temp_set, temp_vals)
                    if len(new_set_b) > best_new_size:
                        best_new_set = new_set_b
                        best_new_vals = new_vals_b
                        best_new_size = len(new_set_b)
                    
                    if best_new_size > best_size:
                        best_set = best_new_set
                        best_vals = best_new_vals
                        best_size = best_new_size
                        improved = True
                        break  # Restart with new best
                
                if not improved:
                    break
            
            return [vals[i] for i in sorted(best_set)]

        results.append(fast_solve(vals))

    return results
