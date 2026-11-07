import time
import random
from math import isqrt

def _is_prime(a):
    if a < 2: return False
    if a < 4: return True
    if a % 2 == 0 or a % 3 == 0: return False
    d = 5
    while d * d <= a:
        if a % d == 0 or a % (d + 2) == 0: return False
        d += 6
    return True

def construct():
    # Precompute primes and composites
    candidates = list(range(2, 41))
    primes = [a for a in candidates if _is_prime(a)]
    composites = [a for a in candidates if not _is_prime(a)]
    
    best_size = 0
    best_set = []
    deadline = time.time() + 50 # 50s margin
    
    # Helper to check if c can be added to S with current products
    def can_add(c, S, products):
        # products is a set of all subset products of S
        new_products = set()
        for p in products:
            new_products.add(p * c)
        # Check intersection
        if products.isdisjoint(new_products):
            return True, products.union(new_products)
        return False, None

    # Strategy 1: Greedy from primes
    for _ in range(1000): # Adjust based on time
        if time.time() > deadline: break
        S = primes.copy()
        products = {1}
        # Add primes to products
        for p in S:
            new_p = set()
            for x in products:
                new_p.add(x * p)
            products = products.union(new_p)
        
        # Shuffle composites
        c_list = composites.copy()
        random.shuffle(c_list)
        
        for c in c_list:
            ok, new_products = can_add(c, S, products)
            if ok:
                S.append(c)
                products = new_products
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
        
        # Local search on S
        # Try to swap or add more?
        # If we hit a local max, try swapping a composite with another composite?
        # Or remove a prime to add 2 composites?
        # Let's do a simple swap loop
        for _ in range(100):
            if time.time() > deadline: break
            # Pick random s in S, c not in S
            if not S: continue
            s_idx = random.randint(0, len(S)-1)
            s_val = S[s_idx]
            # Pick c from remaining
            remaining = [x for x in candidates if x not in S]
            if not remaining: break
            c_val = random.choice(remaining)
            
            # Test S' = S - {s_val} + {c_val}
            # We need to recompute products for S - {s_val}
            # This is expensive. 
            # Optimization: Only try swaps if we are stuck at < 18?
            # Or just rely on greedy restarts.
            pass 
            
    # Return best set found
    return best_set
