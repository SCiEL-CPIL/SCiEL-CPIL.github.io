"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

def _can_add(products, x):
    """
    Check if adding x to a set with existing subset products 'products'
    keeps all subset products distinct.
    
    New products will be {p * x for p in products}.
    For validity: none of p*x can collide with existing products.
    (p1*x == p2*x implies p1==p2, so new products are distinct among themselves.)
    """
    # Optimization: if x is 1, it's always a problem, but x >= 2.
    # Check if any p * x is in products.
    # This is equivalent to checking if x is in the ratio set {p2/p1 | p1,p2 in products}.
    # But direct multiplication is simpler.
    for p in products:
        if p * x in products:
            return False
    return True

def _add_number(products, x):
    """Return new set of products after adding x."""
    new_prods = set(products)
    for p in products:
        new_prods.add(p * x)
    return new_prods

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    """
    Find the largest set of distinct integers in 2..40 such that all
    2^k subset products (including empty product = 1) are distinct.
    """
    candidates = list(range(2, 41))
    
    # Separate primes and composites
    primes = [x for x in candidates if _is_prime(x)]
    composites = [x for x in candidates if not _is_prime(x)]
    
    # Sort composites by number of prime factors (ascending) to find valid sets faster?
    # Or maybe descending?
    # Let's try a few orderings for the greedy initialization.
    
    def num_prime_factors(x):
        count = 0
        d = 2
        n = x
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    # Greedy initialization
    # Strategy 1: Primes first, then composites sorted by num factors
    order1 = primes + sorted(composites, key=lambda x: num_prime_factors(x))
    products = {1}
    selected = []
    for x in order1:
        if _can_add(products, x):
            products = _add_number(products, x)
            selected.append(x)
    
    best_size = len(selected)
    best_set = list(selected)
    
    # Backtracking search to improve
    # We use the greedy result as a lower bound.
    # We search for sets larger than best_size.
    
    # To make backtracking efficient, we order candidates.
    # Primes are very likely to be in the optimal set.
    # Let's fix the first few candidates to primes?
    # Or just use a standard DFS with pruning.
    
    # We'll use a recursive helper.
    # To avoid deep recursion issues or excessive branching, we limit the search.
    
    # We can also try different starting points for the greedy phase.
    
    # Let's run a DFS that tries to include all primes first, then adds composites.
    # This is a heuristic to find a large set quickly.
    
    # Actually, let's just run a full backtracking on the sorted candidates.
    # Since N=40 is small, and we have a good lower bound, it should be fast.
    
    # Sort candidates to prioritize those that are "easier" to add.
    # Primes are easiest.
    search_candidates = primes + composites
    
    best = [best_set]
    
    def backtrack(candidates, products, selected, depth):
        if time.time() > _DEADLINE:
            return
        
        if len(selected) > len(best[0]):
            best[0] = list(selected)
            # Update best_size implicitly via best[0]
            
        if depth >= len(candidates):
            return
            
        # Pruning: if we can't beat the best size with remaining candidates
        if len(selected) + (len(candidates) - depth) <= len(best[0]):
            return
            
        x = candidates[depth]
        
        # Try including x
        if _can_add(products, x):
            new_prods = _add_number(products, x)
            selected.append(x)
            backtrack(candidates, new_prods, selected, depth + 1)
            selected.pop()
            
        # Try excluding x
        backtrack(candidates, products, selected, depth + 1)
        
    # Run backtracking
    backtrack(search_candidates, {1}, [], 0)
    
    # Verify the best result
    def verify(nums):
        prods = {1}
        for x in nums:
            new_prods = set()
            for p in prods:
                new_prods.add(p * x)
            if len(prods | new_prods) != len(prods) + len(new_prods):
                return False
            prods = prods | new_prods
        return True
        
    if not verify(best[0]):
        # Fallback to primes
        best[0] = primes
        
    return [int(x) for x in best[0]]

# EVOLVE-BLOCK-END
