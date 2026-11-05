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

def _can_add(values, products, x):
    """Check if x can be appended to values (strictly increasing, no product collision)."""
    if values and x <= values[-1]:
        return False
    tails = [x]
    prod = x
    for a in reversed(values):
        prod *= a
        tails.append(prod)
    if products.intersection(tails):
        return False
    return True

def _greedy_pass(values, products, start_from=2):
    """Continue greedy from current state."""
    values = list(values)
    products = set(products)
    for x in range(start_from, 129):
        if _can_add(values, products, x):
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            values.append(x)
            products.update(tails)
    return values, products

def _local_search(values, products, deadline):
    """Apply local search: remove-and-reinsert, insert-new operations."""
    values = list(values)
    products = set(products)
    best = len(values)
    best_values = list(values)
    n = len(values)
    
    for _ in range(6):  # More passes
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
                products = set(new_prods)
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
                test_prods = _all_products(test_vals)
                if len(test_prods) == len(test_vals) * (len(test_vals) + 1) // 2:
                    new_vals, new_prods = _greedy_pass(test_vals, test_prods, start_from=128)
                    if len(new_vals) > best:
                        best = len(new_vals)
                        best_values = list(new_vals)
                        values = list(new_vals)
                        products = set(new_prods)
                        used_set = set(values)
                        improved = True
        
        # Swap: try swapping two adjacent elements (if still strictly increasing)
        for i in range(len(values) - 1):
            if time.time() > deadline:
                break
            # Swap values[i] and values[i+1]
            if values[i] + 1 < values[i+1]:  # ensure strict increase after swap
                swapped = values[:i] + [values[i+1], values[i]] + values[i+2:]
                if swapped[i] < swapped[i+1]:
                    test_prods = _all_products(swapped)
                    if len(test_prods) == len(swapped) * (len(swapped) + 1) // 2:
                        new_vals, new_prods = _greedy_pass(swapped, test_prods, start_from=128)
                        if len(new_vals) > best:
                            best = len(new_vals)
                            best_values = list(new_vals)
                            values = list(new_vals)
                            products = set(new_prods)
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
                test_prods = _all_products(test_vals)
                if len(test_prods) == len(test_vals) * (len(test_vals) + 1) // 2:
                    new_vals, new_prods = _greedy_pass(test_vals, test_prods, start_from=128)
                    if len(new_vals) > best:
                        best = len(new_vals)
                        best_values = list(new_vals)
                        values = list(new_vals)
                        products = set(new_prods)
                        improved = True
                        break
        
        # Shake: remove 2-3 random elements and re-greedy
        if len(values) > 4 and time.time() < deadline:
            for _shake in range(10):
                if time.time() > deadline:
                    break
                n_remove = min(3, len(values) - 2)
                remove_idx = random.sample(range(len(values)), n_remove)
                remove_set = set(remove_idx)
                vals_shake = [v for i, v in enumerate(values) if i not in remove_set]
                prods_shake = _all_products(vals_shake)
                new_vals, new_prods = _greedy_pass(vals_shake, prods_shake, start_from=2)
                if len(new_vals) > best:
                    best = len(new_vals)
                    best_values = list(new_vals)
                    values = list(new_vals)
                    products = set(new_prods)
                    improved = True
        
        if not improved:
            break
    
    return best_values

def construct():
    deadline = time.time() + 82.0  # Safety margin before 90s kill
    
    # Strategy 1: Standard greedy
    values1, products1 = [], set()
    for x in range(2, 129):
        if _can_add(values1, products1, x):
            tails = [x]
            prod = x
            for a in reversed(values1):
                prod *= a
                tails.append(prod)
            values1.append(x)
            products1.update(tails)
    best_values = list(values1)
    best_len = len(values1)
    
    # Strategy 2: Greedy with different starting points (skip some small values)
    for skip_start in range(2, 15):
        if time.time() > deadline:
            break
        values_try, products_try = [], set()
        for x in range(2, 129):
            if x < skip_start and x != 2:
                continue
            if _can_add(values_try, products_try, x):
                tails = [x]
                prod = x
                for a in reversed(values_try):
                    prod *= a
                    tails.append(prod)
                values_try.append(x)
                products_try.update(tails)
        if len(values_try) > best_len:
            best_len = len(values_try)
            best_values = list(values_try)
    
    # Strategy 3: Randomized multi-start greedy
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        values_try, products_try = [], set()
        # Random subset of initial values - try different densities
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        for x in sorted(candidates):
            if _can_add(values_try, products_try, x):
                tails = [x]
                prod = x
                for a in reversed(values_try):
                    prod *= a
                    tails.append(prod)
                values_try.append(x)
                products_try.update(tails)
        if len(values_try) > best_len:
            best_len = len(values_try)
            best_values = list(values_try)
    
    # Strategy 4: Greedy with random perturbations - occasionally skip a candidate
    for trial in range(30):
        if time.time() > deadline:
            break
        random.seed(42 + trial)
        values_try, products_try = [], set()
        skip_prob = 0.05 + 0.35 * (trial / 30.0)
        for x in range(2, 129):
            if random.random() < skip_prob:
                continue
            if _can_add(values_try, products_try, x):
                tails = [x]
                prod = x
                for a in reversed(values_try):
                    prod *= a
                    tails.append(prod)
                values_try.append(x)
                products_try.update(tails)
        if len(values_try) > best_len:
            best_len = len(values_try)
            best_values = list(values_try)
    
    # Strategy 5: Greedy with periodic skips (skip every k-th candidate)
    for k in range(2, 8):
        if time.time() > deadline:
            break
        values_try, products_try = [], set()
        count = 0
        for x in range(2, 129):
            count += 1
            if count % k == 0:
                continue
            if _can_add(values_try, products_try, x):
                tails = [x]
                prod = x
                for a in reversed(values_try):
                    prod *= a
                    tails.append(prod)
                values_try.append(x)
                products_try.update(tails)
        if len(values_try) > best_len:
            best_len = len(values_try)
            best_values = list(values_try)
    
    # Apply local search to the best found
    if time.time() < deadline:
        products_best = _all_products(best_values)
        best_values = _local_search(best_values, products_best, deadline)
    
    # Final validation
    final_prods = _all_products(best_values)
    expected = len(best_values) * (len(best_values) + 1) // 2
    if len(final_prods) != expected:
        # Fallback to a known valid greedy result
        values_fallback, products_fallback = [], set()
        for x in range(2, 129):
            if _can_add(values_fallback, products_fallback, x):
                tails = [x]
                prod = x
                for a in reversed(values_fallback):
                    prod *= a
                    tails.append(prod)
                values_fallback.append(x)
                products_fallback.update(tails)
        best_values = values_fallback
    
    return best_values
# EVOLVE-BLOCK-END
