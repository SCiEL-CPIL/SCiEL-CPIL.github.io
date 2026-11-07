"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

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

def _subset_products_set(lst):
    """Return the set of all subset products of lst."""
    k = len(lst)
    prods = {1}
    for a in lst:
        new_prods = set()
        for p in prods:
            np = p * a
            new_prods.add(np)
        prods |= new_prods
    return prods

def _is_valid_set(lst):
    """Check if all subset products of lst are distinct."""
    prods = _subset_products_set(lst)
    return len(prods) == (1 << len(lst))

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin for 60s limit
    
    # Pre-compute primes and composites
    primes = [n for n in range(2, 41) if _is_prime(n)]
    composites = [n for n in range(2, 41) if not _is_prime(n)]
    all_numbers = list(range(2, 41))
    
    best_size = 0
    best_set = []
    
    # Initial greedy solution
    current_set = primes[:]
    if _is_valid_set(current_set):
        best_size = len(current_set)
        best_set = current_set[:]
        
    # Try to improve by adding composites greedily
    current_prods = _subset_products_set(current_set)
    for c in composites:
        if time.time() > deadline:
            break
        # Check if c can be added
        # We need to check if any existing product * c is in current_prods
        # Or more efficiently, just generate new products and check overlap
        can_add = True
        for p in current_prods:
            if p * c in current_prods:
                can_add = False
                break
        if can_add:
            current_set.append(c)
            # Update products
            new_prods = set(current_prods)
            for p in current_prods:
                new_prods.add(p * c)
            current_prods = new_prods
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
                
    # Local Search / Randomized Search
    # We will try to find a set of size best_size + 1 or more.
    # Strategy: Randomly select a subset of size k, check validity.
    # Or, start from best_set and try swaps.
    
    # Let's try a randomized approach to find larger sets.
    # We know the max possible is 18.
    # We can try to construct sets of size 13, 14, ..., 18.
    
    # A better heuristic: 
    # 1. Start with a random set of size 'target_size'.
    # 2. Check if valid.
    # 3. If not, try to fix it by swapping elements.
    
    # Alternatively, use the fact that primes are "independent".
    # Composites are "dependent" on primes.
    # If we include a composite, we might be able to exclude one of its prime factors?
    # No, that's not quite right. The condition is on the *products*.
    
    # Let's try a simple random restart search.
    # In each restart, we pick a random subset of size 'target_size' and check.
    # This is inefficient if target_size is large and probability of validity is low.
    
    # Better: Start with best_set, and try to add elements.
    # If we can't add, try to remove one element and add two?
    
    # Let's implement a more directed search.
    # We will try to build a set of size 18.
    # We know that if we pick numbers with distinct prime factors, it works.
    # But we have only 12 primes.
    # So we must use composites.
    
    # Let's try to use a greedy algorithm with randomization.
    # Sort candidates randomly, pick if valid.
    
    for _ in range(100): # 100 restarts
        if time.time() > deadline:
            break
            
        random.shuffle(all_numbers)
        current_set = []
        current_prods = {1}
        
        for n in all_numbers:
            # Check if n can be added
            can_add = True
            # Optimization: if n is in current_set, skip (distinct integers required)
            if n in current_set:
                continue
                
            # Check for collision
            # new_prods = {p * n for p in current_prods}
            # if new_prods & current_prods: collision
            # We can check this faster
            for p in current_prods:
                if p * n in current_prods:
                    can_add = False
                    break
            
            if can_add:
                current_set.append(n)
                new_prods = set(current_prods)
                for p in current_prods:
                    new_prods.add(p * n)
                current_prods = new_prods
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we reached 18, we can stop early? 
            # The task says "1..18", so 18 is the max possible.
            if best_size == 18:
                break

    # Final verification of best_set
    if not _is_valid_set(best_set):
        # Fallback to primes if something went wrong
        best_set = primes
        
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
