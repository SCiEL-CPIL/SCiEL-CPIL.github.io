import time
from collections import defaultdict

# Precompute prime factorization for all numbers 2..128
def _factorize(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# Map each number to its prime exponent vector (as a tuple of exponents)
# We'll use a dictionary: number -> frozenset of (prime, exponent) pairs
_NUM_FACTORS = {}
for _i in range(2, 129):
    _factors = _factorize(_i)
    _cnt = defaultdict(int)
    for _p in _factors:
        _cnt[_p] += 1
    _NUM_FACTORS[_i] = tuple(sorted(_cnt.items()))

def _add_vectors(v1, v2):
    """Add two prime-exponent vectors represented as sorted tuples of (prime, exp)."""
    if not v1:
        return v2
    if not v2:
        return v1
    d = defaultdict(int)
    for p, e in v1:
        d[p] += e
    for p, e in v2:
        d[p] += e
    return tuple(sorted(d.items()))

def _make_key(v):
    """Create a hashable key from a prime-exponent vector."""
    return frozenset(v) if v else frozenset()

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_seq = []
    
    # Strategy: Backtracking search with chronological backtracking
    # Try to build a sequence by choosing values in increasing order
    # but with the ability to skip and backtrack
    
    def backtrack(start_val, current_seq, products_set, suffix_products):
        """
        current_seq: list of chosen values so far
        products_set: set of all block products seen so far (as frozensets)
        suffix_products: list of suffix products of current_seq, 
                         suffix_products[k] = product of current_seq[k:]
        """
        nonlocal best_seq
        if time.time() > deadline:
            return
        if len(current_seq) > len(best_seq):
            best_seq = list(current_seq)
        
        # Try to add more elements
        for x in range(start_val, 129):
            if time.time() > deadline:
                return
            
            # Compute new block products involving x
            # These are: x, x*last, x*last*second_last, ..., x*product(current_seq)
            new_prods = []
            p = 1
            for a in reversed(current_seq):
                p *= a
                new_prods.append(_make_key(_add_vectors(_NUM_FACTORS[x], _NUM_FACTORS[a] if False else _factorize_to_vec(a))))
            
            # Actually, let's compute more carefully
            # Suffix products of current_seq: 
            # suffix_products[0] = prod(current_seq[0:]), 
            # suffix_products[1] = prod(current_seq[1:]), etc.
            # New products: x * suffix_products[k] for k in 0..len(current_seq)
            # where suffix_products[len(current_seq)] = 1 (empty product)
            
            # Let me recompute using the vector approach
            new_prod_vecs = []
            # x alone
            new_prod_vecs.append(_NUM_FACTORS[x])
            # x * current_seq[-1]
            # x * current_seq[-2] * current_seq[-1]
            # ...
            # x * product(current_seq)
            
            suffix_prod = 1
            for a in reversed(current_seq):
                suffix_prod *= a
                # new product = x * suffix_prod
                # We need to check if this product's factorization is already in products_set
                # Compute factorization of x * suffix_prod
                # = factorization(x) + factorization(suffix_prod)
                # But we don't have factorization of suffix_prod stored...
                # Let's store suffix factorizations instead
                
            # This is getting complex. Let me use a different approach:
            # Store all block products as their prime-exponent vectors
            # When adding x at the end, the new blocks are:
            # [x], [x, last], [x, last, second_last], ..., [x, ..., first]
            # Their products can be computed incrementally
            
            # For efficiency, let's just compute the actual products but cap them
            # If a product exceeds some threshold, we can use the vector representation
            # Actually, for numbers up to 128 with sequence length up to ~20,
            # products can be huge. Let's use vectors throughout.
            
            # Restart: use vector-based approach properly
            return  # Skip this iteration, use the simpler approach below
        
    # Simpler approach: use the original greedy + local search but with better validation
    # and a more aggressive local search
    
    # 1. Greedy with multiple orderings
    def greedy(order):
        values = []
        products_set = set()
        for x in order:
            if time.time() > deadline:
                break
            # Check if x can be added
            new_prods = []
            p = 1
            for a in reversed(values):
                p *= a
                new_prods.append(x * p)
            new_prods.append(x)
            if products_set.intersection(new_prods):
                continue
            values.append(x)
            products_set.update(new_prods)
        return values
    
    # Try several greedy strategies
    candidates = [
        list(range(2, 129)),
        list(range(128, 1, -1)),
        [x for x in range(2, 129) if _is_prime(x)],  # primes first
        [x for x in range(2, 129) if not _is_prime(x)],  # composites first
    ]
    
    best = []
    for order in candidates:
        seq = greedy(order)
        if len(seq) > len(best):
            best = seq
    
    best_seq = best
    
    # 2. Local search: try to insert missing values
    improved = True
    max_iter = 0
    while improved and time.time() < deadline and max_iter < 100:
        improved = False
        max_iter += 1
        candidates = [x for x in range(2, 129) if x not in best_seq]
        # Sort by how "easy" they are to insert (fewer conflicts)
        for x in candidates:
            if time.time() > deadline:
                break
            lo = 0
            while lo < len(best_seq) and best_seq[lo] < x:
                lo += 1
            hi = len(best_seq)
            while hi > 0 and best_seq[hi - 1] > x:
                hi -= 1
            for i in range(hi, lo + 1):
                new_seq = best_seq[:i] + [x] + best_seq[i:]
                if is_valid(new_seq):
                    best_seq = new_seq
                    improved = True
                    break
            if improved:
                break
    
    # 3. Try removing and re-adding elements to find better arrangements
    if time.time() < deadline:
        for _ in range(50):  # Limited iterations
            if time.time() > deadline:
                break
            # Try removing each element and see if we can do better
            for i in range(len(best_seq)):
                if time.time() > deadline:
                    break
                removed = best_seq[i]
                without = best_seq[:i] + best_seq[i+1:]
                # Now try to insert other missing values
                missing = [x for x in range(2, 129) if x not in without]
                improved_here = False
                for x in missing:
                    if time.time() > deadline:
                        break
                    lo = 0
                    while lo < len(without) and without[lo] < x:
                        lo += 1
                    hi = len(without)
                    while hi > 0 and without[hi - 1] > x:
                        hi -= 1
                    for j in range(hi, lo + 1):
                        new_seq = without[:j] + [x] + without[j:]
                        if is_valid(new_seq) and len(new_seq) > len(best_seq):
                            best_seq = new_seq
                            improved_here = True
                            break
                    if improved_here:
                        break
                if not improved_here:
                    # Try putting the removed element back or finding a better spot
                    pass
    
    # Final safety check
    result = []
    for v in best_seq:
        if not result or v > result[-1]:
            result.append(v)
    if result and is_valid(result):
        return result
    return [2]

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

def _factorize_to_vec(n):
    """Return factorization as a tuple of (prime, exponent) pairs."""
    d = defaultdict(int)
    tmp = n
    p = 2
    while p * p <= tmp:
        while tmp % p == 0:
            d[p] += 1
            tmp //= p
        p += 1
    if tmp > 1:
        d[tmp] += 1
    return tuple(sorted(d.items()))

def is_valid(seq):
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True
