import time
from random import Random
import sys
import numpy as np

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

    def fast_solve(vals):
        n = len(vals)
        
        # Precompute conflict information
        # For each index i, conflict_pairs[i] is a set of pairs (j, k) such that vals[i] + vals[j] == vals[k] or vals[i] + vals[k] == vals[j]
        # Actually, it's better to precompute for each value v, which other values x make v+x also in the set.
        # Let's precompute a dictionary: val_to_idx mapping.
        val_to_idx = {v: i for i, v in enumerate(vals)}
        
        # For each index i, let's compute the set of indices j such that adding i and j would conflict if some k is also present.
        # Conflict condition: a+b=c.
        # If we have a and b, we cannot have c.
        # If we have a and c, we cannot have b (if b = c-a).
        # If we have b and c, we cannot have a (if a = c-b).
        
        # So for a candidate value v, it conflicts with a set S if:
        # 1. There exists x, y in S such that x + y == v.
        # 2. There exists x in S such that v + x in S.
        # 3. There exists x in S such that v - x in S. (Since x + (v-x) = v)
        
        # We can speed up the check by precomputing for each value v, the list of values x such that v+x is in the universe.
        # But the universe is fixed.
        
        # Let's try a different approach: Simulated Annealing or Hill Climbing with a good move generator.
        
        # State: binary vector x of length n.
        # Objective: maximize sum(x) subject to sum-free constraint.
        
        # Let's use a greedy initialization with multiple orders, then local search.
        
        def can_add(val, s):
            """Check if val can be added to set s (set of values) without violating sum-free property."""
            # s is a set of values
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
        rng = Random(12345)
        for _ in range(10):
            order3 = list(range(n))
            rng.shuffle(order3)
            strategies.append(order3)
        
        # Find best greedy solution
        best_set = set()
        best_vals = set()
        best_size = 0
        
        for order in strategies:
            if time.time() > deadline - 20:
                break
            current, current_vals = greedy_fill(order)
            if len(current) > best_size:
                best_size = len(current)
                best_set = current
                best_vals = current_vals
        
        # Local Search: 1-move (remove one, add one) and 2-move (remove one, add two)
        # We will use a simple hill climbing with restarts.
        
        improved = True
        iteration = 0
        max_iterations = 100
        
        while improved and iteration < max_iterations and time.time() < deadline - 10:
            improved = False
            iteration += 1
            
            # Try removing each element in the current set
            # To make it faster, we can shuffle the order of removals
            removal_order = list(best_set)
            rng.shuffle(removal_order)
            
            for v_idx in removal_order:
                if time.time() > deadline - 10:
                    break
                
                v_val = vals[v_idx]
                temp_set = best_set - {v_idx}
                temp_vals = best_vals - {v_val}
                
                # Try to add elements from the complement
                # We want to add as many as possible.
                # Let's try to add elements in a specific order.
                # Maybe sort by how many conflicts they have?
                
                # For simplicity, let's try to add all valid elements in index order.
                # This is O(N^3) worst case but N is small.
                
                new_set = set(temp_set)
                new_vals = set(temp_vals)
                
                # We can optimize the "can_add" check by maintaining the set of sums?
                # No, let's just use the simple check.
                
                # To find the best addition, we can iterate through all candidates.
                # Let's try to add candidates in a random order to escape local minima?
                # Or just index order for determinism.
                
                for idx in range(n):
                    if idx in new_set:
                        continue
                    val = vals[idx]
                    if can_add(val, new_vals):
                        new_set.add(idx)
                        new_vals.add(val)
                
                if len(new_set) > best_size:
                    best_set = new_set
                    best_vals = new_vals
                    best_size = len(new_set)
                    improved = True
                    break  # Restart with new best
        
        return [vals[i] for i in sorted(best_set)]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Fallback to odd numbers if time runs out
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
            
        vals = instance(size, seed)
        results.append(fast_solve(vals))

    return results
