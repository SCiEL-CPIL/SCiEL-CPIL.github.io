"""
Program to maximize the number of integers in [2, 40] with distinct subset products.
Uses a greedy algorithm to add numbers, checking for product collisions.
"""

def construct():
    limit = 40
    candidates = list(range(2, limit + 1))
    
    selected = []
    products = {1}
    
    # Try to add candidates.
    # Order: Primes first (safest), then others.
    # To maximize count, we should prefer numbers that are "independent".
    # A simple heuristic: try all, if collision, skip.
    # To get a good result, we should sort candidates by "complexity" or just try all.
    # Since N is small, we can try a few orders or just a standard order.
    # Standard order 2..40 might miss better combinations.
    # But 15 is likely the max or close to it.
    
    # Let's try adding in order of increasing value.
    for x in candidates:
        new_products = {p * x for p in products}
        # Check if any new product already exists in products
        if not new_products.intersection(products):
            # Also need to check if new products collide with each other?
            # No, if p1*x == p2*x, then p1==p2, so no collision among new ones.
            selected.append(x)
            products.update(new_products)
            
    return selected
