"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

def _prime_factorization(n):
    """Return dict of prime->exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _all_subset_products_distinct(nums):
    """Check if all 2^k subset products (including empty=1) are distinct."""
    k = len(nums)
    products = set()
    products.add(1)
    for mask in range(1, 1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def construct():
    start_time = time.time()
    deadline = 75.0  # safety margin before 90s kill
    
    # Get prime factorizations for all numbers 2..40
    candidates = list(range(2, 41))
    factorizations = {n: _prime_factorization(n) for n in candidates}
    
    # Collect all primes involved
    all_primes = set()
    for n in candidates:
        all_primes.update(factorizations[n].keys())
    all_primes = sorted(all_primes)
    prime_index = {p: i for i, p in enumerate(all_primes)}
    
    # Convert each number to its exponent vector
    vectors = {}
    for n in candidates:
        vec = [0] * len(all_primes)
        for p, e in factorizations[n].items():
            vec[prime_index[p]] = e
        vectors[n] = vec
    
    # Greedy approach: try to build largest linearly independent set
    # Use integer row reduction to check independence
    def is_independent(vecs):
        """Check if list of integer vectors are linearly independent over Q."""
        if not vecs:
            return True
        # Gaussian elimination over rationals (using fractions for exactness)
        from fractions import Fraction
        mat = [[Fraction(x) for x in v] for v in vecs]
        m = len(mat)
        n = len(mat[0]) if m > 0 else 0
        rank = 0
        for col in range(n):
            if rank >= m:
                break
            # Find pivot
            pivot_row = None
            for row in range(rank, m):
                if mat[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                continue
            mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
            # Eliminate below
            for row in range(rank + 1, m):
                if mat[row][col] != 0:
                    factor = mat[row][col] / mat[rank][col]
                    for c in range(col, n):
                        mat[row][c] -= factor * mat[rank][c]
            rank += 1
        return rank == len(vecs)
    
    # Strategy: try different orderings and greedy selections
    # First, try a simple greedy: add vectors that maintain independence
    def greedy_independent_order(order):
        selected = []
        for n in order:
            candidate_vecs = [vectors[x] for x in selected] + [vectors[n]]
            if is_independent(candidate_vecs):
                selected.append(n)
        return selected
    
    # Try multiple orderings to find best
    import random
    random.seed(42)
    
    best_set = []
    best_size = 0
    
    # First try natural order
    natural_order = list(range(2, 41))
    result = greedy_independent_order(natural_order)
    if len(result) > best_size:
        best_size = len(result)
        best_set = result
    
    # Try several random orderings
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        random_order = list(range(2, 41))
        random.shuffle(random_order)
        result = greedy_independent_order(random_order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
        elif len(result) == best_size:
            # Keep the one with smaller sum (arbitrary tie-break)
            if sum(result) < sum(best_set):
                best_set = result
    
    # Try to improve: for each element not in best_set, try swapping
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for n in range(2, 41):
                if n in best_set:
                    continue
                # Try adding n and removing one existing element
                for i in range(len(best_set)):
                    new_set = best_set[:i] + best_set[i+1:] + [n]
                    vecs = [vectors[x] for x in new_set]
                    if is_independent(vecs):
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            best_size = len(new_set)
                            improved = True
                            break
                        elif len(new_set) == len(best_set) and sum(new_set) < sum(best_set):
                            best_set = new_set
                            improved = True
                            break
                if improved:
                    break
    
    # Verify the final answer
    if best_set and not _all_subset_products_distinct(best_set):
        # Fallback to a known valid small set
        best_set = [2, 3, 5, 7, 11, 13]
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
