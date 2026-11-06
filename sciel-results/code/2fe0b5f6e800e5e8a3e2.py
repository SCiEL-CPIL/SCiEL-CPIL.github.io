"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _all_products(values):
    """Compute the set of all consecutive block products for a strictly increasing list."""
    prods = set()
    n = len(values)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= values[j]
            prods.add(p)
    return prods

def _can_append(values, products, x):
    """Check if x can be appended to the end of values without creating duplicate products."""
    # New blocks are: [x], [values[-1], x], [values[-2], values[-1], x], ...
    p = x
    if p in products:
        return False
    for a in reversed(values):
        p *= a
        if p in products:
            return False
    return True

def _can_insert_at(values, products, x, pos):
    """
    Check if x can be inserted at position pos in values.
    values is strictly increasing, x is not in values.
    Returns True if no new product conflicts with existing products.
    """
    n = len(values)
    if pos < 0 or pos > n:
        return False
    
    # Compute left suffix products: prod of values[i..pos-1] for i in 0..pos-1
    # left_prods[k] = product of values[pos-1-k] * ... * values[pos-1] (k elements ending at pos-1)
    # Actually let's define:
    # left_suffix[j] = product of values[j..pos-1] for j in 0..pos-1, and left_suffix[pos] = 1
    left_suffix = [1] * (pos + 1)
    p = 1
    for i in range(pos - 1, -1, -1):
        p *= values[i]
        left_suffix[i] = p
    
    # right_prefix[j] = product of values[pos..j] for j in pos..n-1, and right_prefix[pos-1] = 1
    right_prefix = [1] * (n - pos + 1)
    p = 1
    for j in range(pos, n):
        p *= values[j]
        right_prefix[j - pos + 1] = p  # right_prefix[k] for k=1..(n-pos)
    # right_prefix[0] = 1 (empty)
    
    # New products are:
    # 1. [x] -> x
    # 2. [values[i..pos-1], x] for i in 0..pos-1 -> left_suffix[i] * x
    # 3. [x, values[pos..j]] for j in pos..n-1 -> x * right_prefix[j - pos + 1]
    # 4. [values[i..pos-1], x, values[pos..j]] for i in 0..pos-1, j in pos..n-1
    #    -> left_suffix[i] * x * right_prefix[j - pos + 1]
    
    new_prods = set()
    
    # Category 1
    new_prods.add(x)
    
    # Category 2
    for i in range(pos):
        new_prods.add(left_suffix[i] * x)
    
    # Category 3
    for j in range(pos, n):
        new_prods.add(x * right_prefix[j - pos + 1])
    
    # Category 4
    for i in range(pos):
        lp = left_suffix[i]
        for j in range(pos, n):
            new_prods.add(lp * x * right_prefix[j - pos + 1])
    
    # Check conflicts
    for p in new_prods:
        if p in products:
            return False
    
    return True

def _try_insert(values, products, x):
    """Try to insert x at its correct sorted position. Returns (success, new_values, new_products)."""
    # Find position
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    pos = lo
    
    if pos < len(values) and values[pos] == x:
        return False, values, products
    
    if _can_insert_at(values, products, x, pos):
        new_values = values[:pos] + [x] + values[pos:]
        new_products = _all_products(new_values)
        return True, new_values, new_products
    
    return False, values, products

def _try_swap(values, products, i, j):
    """
    Try swapping elements at positions i and j (i < j) such that the result is still
    strictly increasing. Returns (success, new_values, new_products).
    """
    # For the result to be strictly increasing, we need:
    # If i and j are adjacent: values[i] < values[j] already, after swap values[j] < values[i] must hold,
    #   which is false since values[i] < values[j]. So adjacent swaps don't work for increasing order.
    # For non-adjacent: after swapping values[i] and values[j], the sequence must remain strictly increasing.
    # This requires:
    #   - If i > 0: values[i-1] < values[j] (the new value at position i)
    #   - values[j] < values[i+1] if i+1 != j, else values[j] < values[i+2] (but this is complex)
    # Actually, swapping in a sorted array to keep it sorted is only possible if the values are equal,
    # which they aren't. So swaps that maintain strict increasing order are not useful.
    # Instead, let's consider removing an element and re-inserting it elsewhere, or
    # swapping two non-adjacent elements where the order is preserved.
    
    # For a strictly increasing sequence, swapping values[i] and values[j] (i<j) keeps it increasing
    # only if values[i] < values[j] and the swapped values fit in their new positions.
    # After swap: ... values[i-1], values[j], values[i+1], ..., values[j-1], values[i], values[j+1] ...
    # We need: values[i-1] < values[j] (if i>0), values[j] < values[i+1] (if i+1<=j-1, i.e., j>=i+2),
    #          values[j-1] < values[i] (if j-1>=i+1, i.e., j>=i+2), values[i] < values[j+1] (if j<n-1)
    
    n = len(values)
    a, b = values[i], values[j]
    
    # Check if swap maintains strict increasing order
    if i > 0 and values[i-1] >= b:
        return False, values, products
    if j < n-1 and a >= values[j+1]:
        return False, values, products
    if j - i >= 2:
        if b >= values[i+1]:
            return False, values, products
        if values[j-1] >= a:
            return False, values, products
    
    new_values = values[:]
    new_values[i], new_values[j] = b, a
    
    # Verify strictly increasing
    for k in range(n-1):
        if new_values[k] >= new_values[k+1]:
            return False, values, products
    
    new_products = _all_products(new_values)
    return True, new_values, new_products

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    random.seed(42)
    
    # Phase 1: Start with all primes in [2, 128]
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        f = 5
        while f * f <= n:
            if n % f == 0 or n % (f + 2) == 0:
                return False
            f += 6
        return True
    
    primes = [x for x in range(2, 129) if is_prime(x)]
    values = primes[:]
    products = _all_products(values)
    best_values = values[:]
    best_len = len(values)
    
    # Phase 2: Greedily add composites in increasing order
    composites = [x for x in range(2, 129) if not is_prime(x)]
    for x in composites:
        if time.time() >= deadline:
            break
        if _can_append(values, products, x):
            values.append(x)
            # Update products
            p = x
            products.add(p)
            for a in reversed(values[:-1]):
                p *= a
                products.add(p)
            if len(values) > best_len:
                best_len = len(values)
                best_values = values[:]
    
    # Phase 3: Try to insert excluded numbers (composites not yet added)
    excluded = [x for x in range(2, 129) if x not in set(values)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in excluded[:]:
            if time.time() >= deadline:
                break
            success, new_values, new_products = _try_insert(values, products, x)
            if success:
                values = new_values
                products = new_products
                excluded.remove(x)
                improved = True
                if len(values) > best_len:
                    best_len = len(values)
                    best_values = values[:]
    
    # Phase 4: Local search - try random insertions and swaps
    remaining = [x for x in range(2, 129) if x not in set(values)]
    
    # Try random insertions
    attempts = 0
    while time.time() < deadline and attempts < 5000:
        attempts += 1
        if not remaining:
            break
        x = random.choice(remaining)
        success, new_values, new_products = _try_insert(values, products, x)
        if success:
            values = new_values
            products = new_products
            remaining.remove(x)
            if len(values) > best_len:
                best_len = len(values)
                best_values = values[:]
    
    # Try removing some elements to allow adding others
    if time.time() < deadline:
        for _ in range(200):
            if time.time() >= deadline:
                break
            if len(values) <= 1:
                break
            # Remove a random element
            idx = random.randint(0, len(values) - 1)
            removed = values[idx]
            new_values = values[:idx] + values[idx+1:]
            new_products = _all_products(new_values)
            
            # Try to add back removed or other excluded elements
            candidates = [removed] + [x for x in range(2, 129) if x not in set(new_values)]
            random.shuffle(candidates)
            
            best_candidate = None
            for x in candidates:
                if time.time() >= deadline:
                    break
                success, cv, cp = _try_insert(new_values, new_products, x)
                if success:
                    best_candidate = (cv, cp)
                    break
            
            if best_candidate and len(best_candidate[0]) >= len(values):
                values, products = best_candidate
                if len(values) > best_len:
                    best_len = len(values)
                    best_values = values[:]
    
    # Final: try to add any remaining
    remaining = [x for x in range(2, 129) if x not in set(values)]
    for x in remaining:
        if time.time() >= deadline:
            break
        success, new_values, new_products = _try_insert(values, products, x)
        if success:
            values = new_values
            products = new_products
            if len(values) > best_len:
                best_len = len(values)
                best_values = values[:]
    
    return best_values
# EVOLVE-BLOCK-END
