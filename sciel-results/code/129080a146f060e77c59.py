"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    from math import gcd
    
    # Generate all integers 2..40
    candidates = list(range(2, 41))
    
    # Factor each number into prime exponent vector
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def factorize(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    vecs = [factorize(n) for n in candidates]
    
    # We need to find the largest subset of indices such that
    # the corresponding vectors are linearly independent over Q.
    # Use Gaussian elimination / rank tracking with branch and bound.
    
    n = len(candidates)
    
    # Precompute: for quick independence check, we'll maintain a basis.
    # Use a recursive branch-and-bound with memoization on the basis state.
    
    # State: (current_index, basis) where basis is a list of vectors in row echelon form
    # To make it efficient, represent basis as a frozenset or tuple of tuples.
    
    from functools import lru_cache
    
    # Since n=39, and max independent set is 12, we can do DFS with pruning.
    
    def is_independent(vec, basis):
        """Check if vec is independent of basis (list of vectors in reduced form)."""
        v = list(vec)
        for b in basis:
            if v[b[0]] != 0:  # b[0] is the pivot index
                factor = v[b[0]] // b[b[0]] if b[b[0]] != 0 else 0
                # Actually, use exact division since we're over integers and basis is in echelon form
                # Better: use rational arithmetic or keep basis in reduced row echelon
                pass
        # This is getting complex. Let's use a different approach:
        # Maintain basis in reduced row echelon form with rational numbers.
        pass
    
    # Simpler approach: use the fact that max size is 12, and just find any independent set of size 12.
    # The primes themselves form an independent set of size 12.
    # Verify and return.
    
    # Find indices of primes in candidates
    prime_indices = [i for i, v in enumerate(vecs) if sum(v) == 1]
    
    # Verify these are linearly independent (they should be, as they are standard basis vectors)
    # Return the corresponding numbers
    result = [candidates[i] for i in prime_indices]
    
    # Verify all subset products are distinct
    from itertools import combinations
    
    def verify(nums):
        seen = set()
        seen.add(1)  # empty product
        k = len(nums)
        for r in range(1, k + 1):
            for combo in combinations(nums, r):
                prod = 1
                for x in combo:
                    prod *= x
                if prod in seen:
                    return False
                seen.add(prod)
        return True
    
    if not verify(result):
        # Fallback: greedy
        result = []
        for i in range(n):
            test = result + [candidates[i]]
            if verify(test):
                result = test
    
    return result

# EVOLVE-BLOCK-END
