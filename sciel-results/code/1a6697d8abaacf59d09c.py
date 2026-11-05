"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from functools import lru_cache

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

def _is_valid_sequence(values):
    """Check if a sequence is strictly increasing and all consecutive block products are distinct."""
    if not values:
        return True
    if any(values[i] >= values[i+1] for i in range(len(values)-1)):
        return False
    prods = _all_products(values)
    expected = len(values) * (len(values) + 1) // 2
    return len(prods) == expected

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
    
    for _ in range(4):  # Multiple passes
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
        
        if not improved:
            break
    
    return best_values

def _branch_and_bound(max_len, deadline, initial_solution):
    """
    Branch-and-bound to find a longer valid sequence.
    We build the sequence from left to right, choosing the next element.
    State: (last_value, current_length, current_sequence, current_products)
    We prune if:
      1. current_length + (128 - last_value) <= max_len (can't beat best)
      2. No valid next element exists
    """
    best_len = len(initial_solution)
    best_seq = list(initial_solution)
    
    # Precompute: for each possible last value, what's the max additional length?
    # Simple upper bound: 128 - last_value (since we need strictly increasing)
    
    def dfs(last_val, seq, products, depth):
        nonlocal best_len, best_seq
        
        if time.time() > deadline:
            return
        
        # Upper bound: we can add at most (128 - last_val) more elements
        if len(seq) + (128 - last_val) <= best_len:
            return
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)
        
        # Try adding each possible next value
        for x in range(last_val + 1, 129):
            if time.time() > deadline:
                return
            
            # Check if x can be added
            if _can_add(seq, products, x):
                new_seq = seq + [x]
                new_prods = set(products)
                prod = x
                new_prods.add(prod)
                for a in reversed(seq):
                    prod *= a
                    new_prods.add(prod)
                
                dfs(x, new_seq, new_prods, depth + 1)
    
    # Start from the initial solution's last value
    if initial_solution:
        last = initial_solution[-1]
        products = _all_products(initial_solution)
        dfs(last, initial_solution, products, 0)
    else:
        # Try starting from 2
        dfs(1, [], set(), 0)
    
    return best_seq

def _conflict_graph_bnb(deadline, initial_solution):
    """
    Branch-and-bound with conflict graph pruning.
    Build a conflict graph where nodes are values 2-128, and edges indicate
    that two values cannot both appear in the sequence (due to product conflicts
    when they would be adjacent or in certain positions).
    
    Actually, the conflict depends on position, so a simple conflict graph isn't enough.
    Instead, we use a smarter DFS with memoization on (last_value, frozenset of recent products).
    """
    best_len = len(initial_solution)
    best_seq = list(initial_solution)
    
    # Use a recursive search with aggressive pruning
    # Key insight: the product set grows quickly, so we can prune early
    
    search_count = [0]
    
    def dfs(last_val, seq, products):
        nonlocal best_len, best_seq
        
        if time.time() > deadline:
            return
        
        search_count[0] += 1
        
        # Upper bound
        if len(seq) + (128 - last_val) <= best_len:
            return
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)
        
        # Try each possible next value, preferring larger gaps (less likely to conflict)
        for x in range(last_val + 1, 129):
            if time.time() > deadline:
                return
            
            if _can_add(seq, products, x):
                new_seq = seq + [x]
                new_prods = set(products)
                prod = x
                new_prods.add(prod)
                for a in reversed(seq):
                    prod *= a
                    new_prods.add(prod)
                
                dfs(x, new_seq, new_prods)
    
    if initial_solution:
        last = initial_solution[-1]
        products = _all_products(initial_solution)
        dfs(last, initial_solution, products)
    else:
        dfs(1, [], set())
    
    return best_seq

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
    
    # Strategy 2: Greedy with different starting points
    for skip_start in range(2, 12):
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
    for trial in range(30):
        if time.time() > deadline:
            break
        values_try, products_try = [], set()
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
    
    # Strategy 4: Greedy with random perturbations
    for trial in range(20):
        if time.time() > deadline:
            break
        random.seed(42 + trial)
        values_try, products_try = [], set()
        skip_prob = 0.1 + 0.3 * (trial / 20.0)
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
    
    # Strategy 5: Branch-and-bound from the best greedy solution
    if time.time() < deadline - 5.0:
        bnb_values = _branch_and_bound(best_len, deadline - 2.0, best_values)
        if len(bnb_values) > best_len:
            best_len = len(bnb_values)
            best_values = list(bnb_values)
    
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
