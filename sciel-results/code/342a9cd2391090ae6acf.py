"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

P = {'n': 40, 'maximum_size': 18}

def _is_prime(a):
    if a < 2:
        return False
    if a < 4:
        return True
    if a % 2 == 0 or a % 3 == 0:
        return False
    d = 5
    while d * d <= a:
        if a % d == 0 or a % (d + 2) == 0:
            return False
        d += 6
    return True

def _prime_signature(a):
    """Return tuple of prime factors (with multiplicity) for a."""
    factors = []
    d = 2
    n = a
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return tuple(sorted(factors))

def _check_distinct_subset_products_fast(S):
    """
    Check if all 2^k subset products of S are distinct using meet-in-the-middle.
    S is a list of distinct integers >= 2.
    """
    k = len(S)
    if k <= 1:
        return True
    
    # Split S into two halves
    mid = k // 2
    left = S[:mid]
    right = S[mid:]
    
    # Compute all subset products for left half
    left_products = set()
    n_left = len(left)
    for mask in range(1 << n_left):
        prod = 1
        for i in range(n_left):
            if mask & (1 << i):
                prod *= left[i]
        left_products.add(prod)
    
    # Compute all subset products for right half
    right_products = set()
    n_right = len(right)
    for mask in range(1 << n_right):
        prod = 1
        for i in range(n_right):
            if mask & (1 << i):
                prod *= right[i]
        right_products.add(prod)
    
    # Check: all combined products must be distinct
    # The total set of products is {l * r for l in left_products, r in right_products}
    # We need this set to have size 2^k
    # Instead of enumerating all (could be up to 2^18 = 262144 which is fine), 
    # we can check for collisions more cleverly.
    
    # Actually, for k=18, 2^18 = 262144 products. We can just compute them all.
    # But let's be smart: if left_products has size 2^mid and right_products has size 2^(k-mid),
    # then the total number of distinct products is at most 2^mid * 2^(k-mid) = 2^k.
    # It equals 2^k iff there are no collisions, i.e., the mapping (l, r) -> l*r is injective.
    
    # A collision occurs if l1*r1 = l2*r2 with (l1,r1) != (l2,r2).
    # This is equivalent to l1/l2 = r2/r1, i.e., the ratio of two left products equals the ratio of two right products.
    
    # Simpler approach: just compute all 2^k products and check uniqueness.
    # For k=18, this is 262144 big integers. Should be fast enough.
    
    all_products = set()
    for l in left_products:
        for r in right_products:
            prod = l * r
            if prod in all_products:
                return False
            all_products.add(prod)
    
    return len(all_products) == (1 << k)

def _check_distinct_subset_products(S):
    """
    Check if all 2^k subset products of S are distinct.
    Uses meet-in-the-middle for efficiency.
    """
    k = len(S)
    if k <= 1:
        return True
    
    # For small k, direct enumeration is fine
    if k <= 10:
        products = set()
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= S[i]
            if prod in products:
                return False
            products.add(prod)
        return True
    
    # For larger k, use meet-in-the-middle
    return _check_distinct_subset_products_fast(S)

def construct():
    """
    Find a set of distinct integers in 2..40 such that all subset products are distinct.
    Maximize the number of selected integers (up to 18).
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Start with primes in 2..40, which always have distinct subset products
    # because of unique factorization
    primes = [a for a in range(2, 41) if _is_prime(a)]
    
    best_set = primes[:]
    best_size = len(best_set)
    
    # Try to add composite numbers to the set
    # A composite number can be added if it doesn't create a collision
    # Collision condition: there exist disjoint subsets A, B of current set S
    # such that product(A) = composite * product(B), i.e., product(A) / product(B) = composite
    # This means composite must not be expressible as a ratio of two distinct subset products
    
    # More precisely: adding x to S creates a collision iff there exist disjoint
    # subsets A, B of S such that prod(A) = x * prod(B), i.e., prod(A) / prod(B) = x
    # Since all elements are >= 2, prod(A) > prod(B) if A is "larger" in some sense.
    
    # Let's use a greedy approach with local search
    candidates = [a for a in range(2, 41) if a not in best_set]
    
    # Sort candidates by some heuristic - maybe by number of prime factors or by size
    # Try to add composites that are "far" from existing subset product ratios
    
    # For each candidate, check if it can be added
    # We need to check: for all pairs of disjoint subsets A, B of S, prod(A) != x * prod(B)
    # Equivalently: x is not in the set {prod(A) / prod(B) : A, B disjoint subsets of S}
    
    # This is expensive to check directly. Let's use the fact that:
    # prod(A) / prod(B) = x means prod(A) = x * prod(B)
    # So we need to check if x * prod(B) is ever a subset product for some B
    
    # Alternative: use the prime signature approach
    # Each number has a prime signature (exponent vector)
    # Subset products correspond to sums of exponent vectors
    # Distinct subset products iff all subset sums of exponent vectors are distinct
    # This is a linear independence condition over the non-negative integers
    
    # For the set to have distinct subset products, the prime exponent vectors
    # must be such that all subset sums are distinct.
    
    # Let's try a different approach: greedy with backtracking
    # Start with primes, then try to add composites one by one
    
    # Get all subset products of current set
    def get_subset_products(S):
        k = len(S)
        products = set()
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= S[i]
            products.add(prod)
        return products
    
    # Greedy: try to add each candidate if it doesn't break the property
    # Process candidates in some order
    random.seed(42)
    
    # Try multiple random orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = sorted(candidates, key=lambda x: x)
        else:
            order = candidates[:]
            random.shuffle(order)
        
        current_set = primes[:]
        current_size = len(current_set)
        
        for x in order:
            if time.time() > deadline:
                break
            
            if current_size >= P['maximum_size']:
                break
            
            # Try adding x
            new_set = current_set + [x]
            
            # Check if new_set has distinct subset products
            if _check_distinct_subset_products(new_set):
                current_set = new_set
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
        
        if best_size >= P['maximum_size']:
            break
    
    # Try local search: swap elements to potentially improve
    if best_size < P['maximum_size']:
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Try removing one element and adding a different one
            improved = False
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                
                # Remove best_set[i]
                reduced = best_set[:i] + best_set[i+1:]
                
                # Try adding each candidate not in reduced
                for x in range(2, 41):
                    if x in reduced:
                        continue
                    
                    new_set = reduced + [x]
                    if _check_distinct_subset_products(new_set):
                        # Check if this is an improvement
                        # Since we're swapping, size stays the same
                        # We want to find a configuration that allows further additions
                        # For now, just accept if it works and is different
                        if sorted(new_set) != sorted(best_set):
                            best_set = new_set
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                break
    
    # Final verification
    if not _check_distinct_subset_products(best_set):
        # Fallback to primes
        best_set = primes[:]
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
