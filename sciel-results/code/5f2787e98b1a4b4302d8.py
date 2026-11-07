import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    candidates = list(range(2, 41))
    best_set = []
    best_size = 0
    
    # Initial: Primes
    primes = [a for a in candidates if all(a % d for d in range(2, int(a**0.5) + 1))]
    best_set = primes
    best_size = len(primes)
    
    def can_add(current_products, x):
        # Check if adding x creates duplicates
        # new_products = current_products U {p*x for p in current_products}
        # We need len(new_products) == 2 * len(current_products)
        # This implies no p1 * x == p2 for p1, p2 in current_products
        # And p1 * x != p2 * x (trivial)
        
        # Optimization: Check intersection
        # {p*x for p in current_products} intersects current_products?
        # If yes, invalid.
        # Also, are all p*x distinct? Yes, if x != 0.
        
        # So we just need to check if any p*x is in current_products.
        # But we also need to ensure that the NEW products are distinct among themselves?
        # p1*x == p2*x implies p1==p2. So yes, distinct.
        # So the only condition is: (current_products * x) is disjoint from current_products.
        
        # However, we also need to update current_products for the next step.
        # So we must compute the new set.
        
        new_products = set()
        for p in current_products:
            new_products.add(p)
            new_products.add(p * x)
        
        return len(new_products) == 2 * len(current_products), new_products

    # To improve search, we can sort candidates by some heuristic.
    # But random shuffle is robust.
    
    # We can also try to start with a known good set (primes) and try to ADD more?
    # No, primes are already "maximally independent" in a sense, but we can replace them?
    # No, we want to MAXIMIZE size.
    # Starting from scratch with random order is fine.
    
    while time.time() < deadline:
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        current_products = {1}
        
        for x in shuffled:
            if time.time() > deadline:
                break
            
            valid, new_products = can_add(current_products, x)
            if valid:
                current_set.append(x)
                current_products = new_products
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            if best_size == 18:
                break

    return best_set
