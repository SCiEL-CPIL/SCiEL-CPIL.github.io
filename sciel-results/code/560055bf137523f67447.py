import time
import random

def _compute_products(numbers):
    """Compute all subset products."""
    products = {1}
    for n in numbers:
        products = products | {p * n for p in products}
    return products

def _can_add_with_products(products, candidate):
    """Check if adding candidate to set with given subset products is valid."""
    for p in products:
        if p * candidate in products:
            return False
    return True

def _greedy_build(available, random_seed=None):
    """Greedy build with random ordering, maintaining product set."""
    if random_seed is not None:
        random.seed(random_seed)
    
    pool = list(available)
    random.shuffle(pool)
    
    result = []
    products = {1}
    
    for candidate in pool:
        if _can_add_with_products(products, candidate):
            result.append(candidate)
            products = products | {p * candidate for p in products}
    
    return result, products

def _local_search(initial, initial_products, available, max_steps=1000, deadline=None):
    """Local search: try adding elements to improve."""
    start_time = time.time()
    current = list(initial)
    current_products = set(initial_products)
    current_set = set(current)
    
    best = list(current)
    best_products = set(current_products)
    best_len = len(current)
    
    steps = 0
    while steps < max_steps:
        if deadline and time.time() > deadline:
            break
        steps += 1
        
        # Try adding a random element not in current
        for _ in range(30):
            candidate = random.choice(available)
            if candidate not in current_set:
                if _can_add_with_products(current_products, candidate):
                    current.append(candidate)
                    current_set.add(candidate)
                    current_products = current_products | {p * candidate for p in current_products}
                    if len(current) > best_len:
                        best = list(current)
                        best_products = set(current_products)
                        best_len = len(current)
                    break
    
    return best, best_products

def construct():
    """
    Returns a list of distinct integers in the range 2..40 such that
    all 2^k subset products are distinct. Maximizes k.
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    available = list(range(2, 41))
    
    # Start with the known good set of 16
    base_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 16, 9, 25]
    base_products = _compute_products(base_set)
    best = list(base_set)
    best_products = set(base_products)
    best_len = len(base_set)
    
    # Try randomized greedy with different seeds
    for seed in range(500):
        if time.time() > deadline - 5:
            break
        result, products = _greedy_build(available, random_seed=seed)
        if len(result) > best_len:
            best = result
            best_products = products
            best_len = len(result)
    
    # Local search from best
    if time.time() < deadline - 5:
        improved, improved_products = _local_search(best, best_products, available, 
                                                     max_steps=5000,
                                                     deadline=deadline - 3)
        if len(improved) > best_len:
            best = improved
            best_products = improved_products
            best_len = len(improved)
    
    # Verify the result
    if not _is_valid_set(best):
        best = list(base_set)
    
    return [int(x) for x in best]

def _is_valid_set(numbers):
    """Check if all subset products of the given set are distinct."""
    products = {1}
    for n in numbers:
        new_products = {p * n for p in products}
        if new_products & products:
            return False
        products = products | new_products
    return True
