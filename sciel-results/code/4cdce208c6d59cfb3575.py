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

def _count_collisions(values):
    """Return number of product collisions."""
    n = len(values)
    prods = []
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= values[j]
            prods.append(prod)
    
    # Count duplicates
    seen = set()
    collisions = 0
    for p in prods:
        if p in seen:
            collisions += 1
        else:
            seen.add(p)
    return collisions

def _is_valid(values):
    """Check if values form a valid sequence."""
    return _count_collisions(values) == 0

def _greedy_pass(values, products, start_from=2):
    """Continue greedy from current state."""
    values = list(values)
    products = set(products)
    for x in range(start_from, 129):
        if values and x <= values[-1]:
            continue
        # Check if x can be added
        # We need to check if adding x creates any collision
        # New products are x, x*values[-1], x*values[-1]*values[-2], ...
        # We need to check if any of these are in 'products'
        # Also, we need to check if any *existing* product becomes a collision?
        # No, existing products remain the same.
        # But wait, if we insert x in the middle, existing products change?
        # The current greedy appends to the end.
        # So only new products ending at x are created.
        # Products: x, x*val[-1], ..., x*val[0]*...*val[-1]
        
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

def construct():
    deadline = time.time() + 55.0
    
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
    # State: set of values
    # We represent state as a sorted list for easy product calculation
    
    current_set = set(best_values)
    current_vals = sorted(current_set)
    current_collisions = _count_collisions(current_vals)
    current_score = len(current_vals) - 10 * current_collisions
    
    best_valid_vals = list(current_vals) if current_collisions == 0 else []
    best_valid_len = len(best_valid_vals)
    
    T = 1.0
    cooling_rate = 0.999
    
    all_candidates = list(range(2, 129))
    
    while time.time() < deadline:
        # Generate neighbor
        # 70% Swap, 30% Add/Remove
        if random.random() < 0.7:
            # Swap
            if not current_set:
                continue
            u = random.choice(list(current_set))
            # Pick v not in current_set
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
        new_collisions = _count_collisions(new_vals)
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
            T = 0.01 # Floor temperature to allow some exploration
    
    return best_valid_vals if best_valid_vals else best_values

# EVOLVE-BLOCK-END
