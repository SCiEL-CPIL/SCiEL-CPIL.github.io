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
        
        val_to_idx = {v: i for i, v in enumerate(vals)}
        
        def can_add(val, s):
            """Check if val can be added to set s (set of values) without violating sum-free property."""
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
        for _ in range(20):
            order3 = list(range(n))
            rng.shuffle(order3)
            strategies.append(order3)
        
        # Find best greedy solution
        best_set = set()
        best_vals = set()
        best_size = 0
        
        for order in strategies:
            if time.time() > deadline - 30:
                break
            current, current_vals = greedy_fill(order)
            if len(current) > best_size:
                best_size = len(current)
                best_set = current
                best_vals = current_vals
        
        def local_search(start_set, start_vals):
            current_set = set(start_set)
            current_vals = set(start_vals)
            current_size = len(current_set)
            improved = True
            iter_count = 0
            max_iters = 50
            
            while improved and iter_count < max_iters and time.time() < deadline - 5:
                improved = False
                iter_count += 1
                removal_order = list(current_set)
                rng.shuffle(removal_order)
                
                for v_idx in removal_order:
                    if time.time() > deadline - 5:
                        return current_set, current_vals, current_size
                    
                    v_val = vals[v_idx]
                    temp_set = current_set - {v_idx}
                    temp_vals = current_vals - {v_val}
                    
                    new_set = set(temp_set)
                    new_vals = set(temp_vals)
                    
                    # Try to add elements
                    candidates = [i for i in range(n) if i not in new_set]
                    rng.shuffle(candidates)
                
                    for idx in candidates:
                        val = vals[idx]
                        if can_add(val, new_vals):
                            new_set.add(idx)
                            new_vals.add(val)
                    
                    if len(new_set) > current_size:
                        current_set = new_set
                        current_vals = new_vals
                        current_size = len(new_set)
                        improved = True
                        break
            
            return current_set, current_vals, current_size

        # Run local search on the best greedy solution
        best_set, best_vals, best_size = local_search(best_set, best_vals)
        
        # Iterated Local Search: Perturb and re-search
        if time.time() < deadline - 20:
            while time.time() < deadline - 5:
                # Perturbation: remove k random elements
                k = rng.randint(2, 5)
                if len(best_set) < k:
                    break
                perturb_indices = rng.sample(list(best_set), k)
                perturbed_set = best_set - set(perturb_indices)
                perturbed_vals = best_vals - {vals[i] for i in perturb_indices}
                
                # Run local search on perturbed state
                new_set, new_vals, new_size = local_search(perturbed_set, perturbed_vals)
                
                if new_size > best_size:
                    best_set = new_set
                    best_vals = new_vals
                    best_size = new_size
                else:
                    # If no improvement, we could accept a worse solution with some probability (SA)
                    # But simple ILS (keep best) is safer and often sufficient.
                    # To avoid infinite loops on same perturbation, we just continue.
                    pass
        
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
