"""Improved: Try multiple candidate orderings with backtracking to find longer
sequences with distinct consecutive-block products."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50.0  # safety margin well under the 90s kill

    def _check_block_products(values):
        """Check if all nonempty consecutive block products are distinct."""
        products = set()
        n = len(values)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= values[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    def _build_greedy(cands, values=None, products=None, best=None):
        """Greedy construction with optional backtracking."""
        if values is None:
            values = []
        if products is None:
            products = set()
        if best is None:
            best = []
        
        if len(values) > len(best):
            best = list(values)
        
        if time.time() > deadline:
            return best
        
        # Try each candidate not yet used
        used = set(values)
        for idx, x in enumerate(cands):
            if x in used:
                continue
            
            # Check if adding x creates any duplicate block product
            new_tails = []
            prod = x
            new_tails.append(prod)
            for a in reversed(values):
                prod *= a
                new_tails.append(prod)
            
            if not products.intersection(new_tails):
                values.append(x)
                products.update(new_tails)
                
                # Recurse
                best = _build_greedy(cands, values, products, best)
                
                values.pop()
                for p in new_tails:
                    products.discard(p)
        
        return best

    def _build_simple(cands):
        """Simple greedy without backtracking - fast baseline."""
        values, products = [], set()
        for x in cands:
            if time.time() > deadline:
                break
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    # Strategy 1: Odd numbers ascending (current best approach)
    best = _build_simple(list(range(3, 128, 2)))
    
    # Strategy 2: Full range ascending
    alt = _build_simple(list(range(2, 129)))
    if len(alt) > len(best):
        best = alt
    
    # Strategy 3: Try descending odd numbers
    alt2 = _build_simple(list(range(127, 1, -2)))
    if len(alt2) > len(best):
        best = alt2
    
    # Strategy 4: Try with some backtracking on a smaller candidate set
    # Use only numbers that are "rare" (prime powers or products of few primes)
    # to reduce collision probability
    if time.time() < deadline - 10:
        # Try a mixed ordering: start with primes, then composite odds
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
                  53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 
                  113, 127]
        odds_composite = [x for x in range(3, 128, 2) if x not in primes]
        mixed_cands = primes + odds_composite
        alt3 = _build_simple(mixed_cands)
        if len(alt3) > len(best):
            best = alt3
    
    # Strategy 5: If we have time, try limited backtracking on the best so far
    if time.time() < deadline - 20 and len(best) < 15:
        # Try backtracking on a reduced candidate pool
        small_cands = list(range(2, 30))
        bt_best = _build_greedy(small_cands)
        if len(bt_best) > len(best):
            best = bt_best
    
    return best
# EVOLVE-BLOCK-END
