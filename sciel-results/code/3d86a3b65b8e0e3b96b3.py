import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    # Candidates are integers from 2 to 40
    candidates = list(range(2, 41))
    n_candidates = len(candidates)
    
    # We want to find the largest subset S of candidates such that
    # all 2^k subset products are distinct.
    # This is equivalent to: for any two distinct subsets A, B of S,
    # product(A) != product(B).
    # This is also equivalent to: the elements of S are "multiplicatively independent"
    # in the sense that no non-trivial product relation exists.
    # Actually, a simpler necessary and sufficient condition:
    # If we write each number as a vector of prime exponents, then the subset products
    # are distinct iff the vectors are linearly independent over the reals? No, that's not quite right.
    # The subset products are distinct iff the only solution to sum_{i in A} v_i = sum_{i in B} v_i
    # with A, B subsets of indices is A = B. This is equivalent to the vectors being "sum-free"
    # in a certain sense, or more precisely, that all 2^k sums of subsets of the vectors are distinct.
    
    # A key insight: if we choose numbers that are powers of distinct primes, they will work.
    # But we're limited to numbers 2..40.
    # Let's try a greedy approach: start with an empty set, and add elements one by one
    # if they maintain the distinct subset products property.
    
    # To check if adding an element x to a set S maintains the property:
    # We need to check that for all existing subset products p of S, p != x * q for any
    # existing subset product q of S, where p and q are from different "halves" in a sense.
    # Actually, the new subset products are the old ones plus {x * p : p in old_products}.
    # So we need: old_products ∩ {x * p : p in old_products} = {1} (only the empty product 1
    # can appear in both if x=1, but x>=2 so 1 won't be in x*p unless p=1/x which isn't an integer).
    # Wait, 1 is in old_products (empty product). x*1 = x. So we need x not in old_products.
    # More generally, we need: for all p, q in old_products, p != x * q.
    # This is equivalent to: x * q not in old_products for all q in old_products.
    # Or: old_products ∩ (x * old_products) = empty set.
    
    def get_subset_products(S):
        """Returns the set of all subset products of S."""
        prods = {1}
        for x in S:
            new_prods = set()
            for p in prods:
                new_prods.add(x * p)
            prods |= new_prods
        return prods
    
    def is_valid(S):
        """Check if all subset products of S are distinct."""
        prods = get_subset_products(S)
        return len(prods) == (1 << len(S))
    
    # Greedy approach: try to build a large valid set.
    # We can try different orderings of candidates.
    
    best_set = []
    best_size = 0
    
    # First, try a deterministic greedy: sort by some heuristic and add if valid.
    # Heuristic: prefer numbers with fewer/smaller prime factors? Or just try in order?
    
    # Let's try multiple random orderings and greedy additions.
    random.seed(42)
    
    # Precompute: for quick checking, we can maintain the set of subset products.
    
    def greedy_add(candidates_order, current_set, current_prods):
        """Try to add elements from candidates_order to current_set."""
        for x in candidates_order:
            if x in current_set:
                continue
            # Check if adding x maintains validity
            # New prods = current_prods ∪ {x * p : p in current_prods}
            # Need: len(new_prods) == len(current_prods) + len(current_prods) = 2 * len(current_prods)
            # i.e., current_prods ∩ {x * p : p in current_prods} must be empty
            new_prods = set()
            for p in current_prods:
                new_prods.add(x * p)
            if len(new_prods) != len(current_prods):
                # There's overlap, so x * p == q for some p, q in current_prods
                # This means adding x would create duplicate subset products
                continue
            # Also need to check that new_prods doesn't intersect current_prods
            # Wait, new_prods is {x*p : p in current_prods}. We need this disjoint from current_prods.
            if new_prods & current_prods:
                continue
            # Valid to add
            current_set.add(x)
            current_prods |= new_prods
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
        
        return current_set, current_prods
    
    # Try several random permutations
    num_trials = 500
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        order = candidates[:]
        random.shuffle(order)
        
        current_set = set()
        current_prods = {1}
        
        # Greedily add elements
        for x in order:
            if time.time() > deadline:
                break
            new_prods = set()
            for p in current_prods:
                new_prods.add(x * p)
            
            # Check disjointness
            if len(new_prods) == len(current_prods) and not (new_prods & current_prods):
                current_set.add(x)
                current_prods |= new_prods
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
    
    # Also try a deterministic approach: sorted by value
    order = candidates[:]
    current_set = set()
    current_prods = {1}
    for x in order:
        if time.time() > deadline:
            break
        new_prods = set()
        for p in current_prods:
            new_prods.add(x * p)
        if len(new_prods) == len(current_prods) and not (new_prods & current_prods):
            current_set.add(x)
            current_prods |= new_prods
    
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = list(current_set)
    
    # Try sorted by number of prime factors (ascending)
    def num_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    order = sorted(candidates, key=num_prime_factors)
    current_set = set()
    current_prods = {1}
    for x in order:
        if time.time() > deadline:
            break
        new_prods = set()
        for p in current_prods:
            new_prods.add(x * p)
        if len(new_prods) == len(current_prods) and not (new_prods & current_prods):
            current_set.add(x)
            current_prods |= new_prods
    
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = list(current_set)
    
    # Return as a list of Python ints
    return sorted(best_set)

# EVOLVE-BLOCK-END
