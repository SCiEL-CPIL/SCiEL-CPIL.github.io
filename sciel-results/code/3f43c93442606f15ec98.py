"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _all_products(values):
    """Return set of all nonempty consecutive block products."""
    n = len(values)
    prods = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= values[j]
            prods.add(prod)
    return prods

def _count_collisions_fast(values):
    """Return number of product collisions using a set."""
    n = len(values)
    seen = set()
    collisions = 0
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= values[j]
            if prod in seen:
                collisions += 1
            else:
                seen.add(prod)
    return collisions

def _is_valid(values):
    """Check if values form a valid sequence (all block products distinct)."""
    return _count_collisions_fast(values) == 0

def _greedy_pass(values, products, start_from=2):
    """Continue greedy from current state, appending valid values."""
    values = list(values)
    products = set(products)
    for x in range(start_from, 129):
        if values and x <= values[-1]:
            continue
        tails = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if products.intersection(tails):
            continue
        values.append(x)
        products.update(tails)
    return values, products

def _local_search(values, deadline):
    """Apply local search: remove-and-reinsert, insert-new, replace operations."""
    values = list(values)
    best = len(values)
    best_values = list(values)
    
    for _ in range(3):  # Multiple passes
        improved = False
        
        # Remove-and-reinsert: drop one element, re-greedy the whole range.
        for i in range(len(values)):
            if time.time() > deadline:
                break
            vals_without = values[:i] + values[i+1:]
            prods_without = _all_products(vals_without)
            new_vals, new_prods = _greedy_pass(vals_without, prods_without, start_from=2)
            if len(new_vals) > best:
                best = len(new_vals)
                best_values = list(new_vals)
                values = list(new_vals)
                improved = True
        
        # Insert-new: try inserting a fresh value at each gap.
        used_set = set(values)
        for i in range(len(values) + 1):
            if time.time() > deadline:
                break
            lo = (values[i - 1] + 1) if i > 0 else 2
            hi = (values[i] - 1) if i < len(values) else 128
            for v in range(lo, hi + 1):
                if v in used_set:
                    continue
                test_vals = values[:i] + [v] + values[i:]
                if _is_valid(test_vals):
                    new_vals, new_prods = _greedy_pass(test_vals, _all_products(test_vals), start_from=128)
                    if len(new_vals) > best:
                        best = len(new_vals)
                        best_values = list(new_vals)
                        values = list(new_vals)
                        used_set = set(values)
                        improved = True
        
        # Replace: try replacing one element with a different value
        used_set = set(values)
        for i in range(len(values)):
            if time.time() > deadline:
                break
            lo = (values[i-1] + 1) if i > 0 else 2
            hi = (values[i+1] - 1) if i < len(values) - 1 else 128
            for v in range(lo, hi + 1):
                if v == values[i]:
                    continue
                test_vals = values[:i] + [v] + values[i+1:]
                if _is_valid(test_vals):
                    new_vals, new_prods = _greedy_pass(test_vals, _all_products(test_vals), start_from=128)
                    if len(new_vals) > best:
                        best = len(new_vals)
                        best_values = list(new_vals)
                        values = list(new_vals)
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_values

def construct():
    deadline = time.time() + 85.0
    
    # 1. Get initial greedy solution
    values1, products1 = [], set()
    for x in range(2, 129):
        if not values1 or x > values1[-1]:
            tails = [x]
            prod = x
            for a in reversed(values1):
                prod *= a
                tails.append(prod)
            if not products1.intersection(tails):
                values1.append(x)
                products1.update(tails)
    
    best_values = list(values1)
    best_len = len(best_values)
    
    # 2. Simulated Annealing
    current_set = set(best_values)
    current_vals = sorted(current_set)
    current_collisions = _count_collisions_fast(current_vals)
    current_score = len(current_vals) - 10 * current_collisions
    
    best_valid_vals = list(current_vals) if current_collisions == 0 else []
    best_valid_len = len(best_valid_vals)
    
    T = 1.0
    cooling_rate = 0.999
    
    all_candidates = list(range(2, 129))
    
    while time.time() < deadline - 10.0:  # Leave 10s for local search
        # Generate neighbor
        if random.random() < 0.7:
            # Swap
            if not current_set:
                continue
            u = random.choice(list(current_set))
            available = [v for v in all_candidates if v not in current_set]
            if not available:
                continue
            v = random.choice(available)
            
            new_set = set(current_set)
            new_set.remove(u)
            new_set.add(v)
        else:
            # Add or Remove
            if random.random() < 0.5:
                # Add
                available = [v for v in all_candidates if v not in current_set]
                if not available:
                    continue
                v = random.choice(available)
                new_set = set(current_set)
                new_set.add(v)
            else:
                # Remove
                if not current_set:
                    continue
                u = random.choice(list(current_set))
                new_set = set(current_set)
                new_set.remove(u)
        
        new_vals = sorted(new_set)
        new_collisions = _count_collisions_fast(new_vals)
        new_score = len(new_vals) - 10 * new_collisions
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            current_set = new_set
            current_vals = new_vals
            current_collisions = new_collisions
            current_score = new_score
            
            if current_collisions == 0:
                if len(current_vals) > best_valid_len:
                    best_valid_len = len(current_vals)
                    best_valid_vals = list(current_vals)
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
    
    # 3. Local search refinement on the best valid solution
    if best_valid_vals and time.time() < deadline:
        refined = _local_search(best_valid_vals, deadline)
        if len(refined) > best_valid_len:
            best_valid_vals = refined
            best_valid_len = len(refined)
    
    # Also try local search on the initial greedy solution if it's valid
    if _is_valid(values1) and time.time() < deadline:
        refined2 = _local_search(values1, deadline)
        if len(refined2) > best_valid_len:
            best_valid_vals = refined2
            best_valid_len = len(refined2)
    
    return best_valid_vals if best_valid_vals else best_values

# EVOLVE-BLOCK-END
