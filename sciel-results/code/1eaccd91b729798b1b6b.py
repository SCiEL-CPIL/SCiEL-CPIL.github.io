import time
from functools import lru_cache

# Pre-compute primes
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

PRIMES = [x for x in range(2, 129) if is_prime(x)]
COMPOSITES = [x for x in range(2, 129) if not is_prime(x)]

def construct():
    start_time = time.time()
    
    # Initial greedy solution to set a high lower bound
    # Start with all primes
    current_set = set(PRIMES)
    current_list = sorted(current_set)
    
    # Greedily add composites
    # We need to maintain the set of all block products to check validity
    def get_all_products(lst):
        prods = set()
        n = len(lst)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= lst[j]
                prods.add(p)
        return prods

    initial_products = get_all_products(current_list)
    
    # Try to add composites in increasing order of "risk" (smaller first?)
    # Actually, let's just try all composites in random or sorted order
    for c in COMPOSITES:
        if c in current_set:
            continue
        # Check if c can be added
        # New products: c, last*c, last2*last*c, ...
        # We need the suffix products of current_list
        suffix = [1]
        prod = 1
        for val in reversed(current_list):
            prod *= val
            suffix.append(prod)
        
        valid = True
        new_prods = set()
        for s in suffix:
            p = c * s
            if p in initial_products:
                valid = False
                break
            new_prods.add(p)
        
        if valid:
            current_set.add(c)
            current_list.append(c)
            current_list.sort()
            initial_products.update(new_prods)

    best_len = len(current_list)
    best_set = set(current_list)
    
    # Now use Branch and Bound to improve
    # We will search for subsets of the remaining composites or rearrangements?
    # The order is fixed by value. So we are just selecting a subset.
    # The greedy above added composites to the prime base.
    # It might have missed some composites that could be added if we removed some primes?
    # Unlikely, primes are "safe".
    # But maybe we can add *more* composites by removing a composite that blocks others?
    # Or maybe the greedy order was suboptimal.
    
    # Let's run a limited backtracking search from the current state or from scratch?
    # From scratch is safer to ensure global optimality within time.
    # But 127 numbers is too many for full backtracking.
    # We can restrict the search to "improving" the current set.
    
    # Alternative: Use the MIS idea.
    # Conflict graph: u, v conflict if they cannot be in the same valid sequence.
    # As discussed, this is hard to define pairwise.
    
    # Let's stick to the Backtracking on the full set, but with strong pruning.
    # We'll iterate numbers from 2 to 128.
    # State: (index, current_list, current_products)
    
    global_best = best_len
    global_best_set = best_set.copy()
    
    def backtrack(idx, current_list, current_products, current_set):
        nonlocal global_best, global_best_set
        
        if time.time() - start_time > 55:
            return
        
        # Pruning
        if len(current_list) + (128 - idx + 1) <= global_best:
            return
            
        # Update best
        if len(current_list) > global_best:
            global_best = len(current_list)
            global_best_set = set(current_list)
            
        # Try to add numbers from idx to 128
        # Optimization: Only consider numbers that don't conflict immediately
        for x in range(idx, 129):
            # Check if x is already in current_set (it shouldn't be if we iterate idx upwards)
            if x in current_set:
                continue
                
            # Check validity of adding x
            # Suffix products of current_list
            suffix = [1]
            prod = 1
            for val in reversed(current_list):
                prod *= val
                suffix.append(prod)
                
            valid = True
            new_prods = set()
            for s in suffix:
                p = x * s
                if p in current_products:
                    valid = False
                    break
                new_prods.add(p)
                
            if not valid:
                continue
                
            # Recurse
            new_list = current_list + [x]
            new_set = current_set | {x}
            new_prods_set = current_products | new_prods
            
            backtrack(x + 1, new_list, new_prods_set, new_set)
            
            # Backtrack
            # No need to explicitly remove from sets if we pass copies or use immutable structures
            # Here we passed new_list, new_set, new_prods_set which are new objects
    
    # Start search from scratch?
    # If we start from scratch, we might find a better solution than the greedy one.
    # The greedy one is a valid lower bound.
    # We start with empty list.
    backtrack(2, [], set(), set())
    
    return sorted(global_best_set)
