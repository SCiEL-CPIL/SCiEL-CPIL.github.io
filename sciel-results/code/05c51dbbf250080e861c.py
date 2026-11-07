"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # We model each integer n in 2..40 by its prime-exponent vector over the 12 primes
    # that can appear.  The condition "all 2^k subset products are distinct" is
    # equivalent to: no non-trivial {-1,0,1} linear combination of the chosen vectors
    # equals the zero vector (this is the "1-sum-free" condition over Z^12).
    #
    # Strategy:
    #   1. Factorize 2..40 into exponent vectors.
    #   2. Find a maximal linearly independent basis via Gaussian elimination over Q.
    #      This gives 12 numbers that are linearly independent (hence their subset
    #      products are trivially distinct, since any collision would imply a
    #      {-1,0,1} relation, which is a Z-linear relation, contradicting
    #      linear independence over Q).
    #   3. Greedily add the remaining 28 numbers, checking the full 2^k subset-product
    #      distinctness each time.  Try multiple orderings (ascending, descending,
    #      by "rarity") and keep the best result.
    #   4. Do local improvement: try swapping one element for another to gain size.

    from math import isqrt
    from itertools import combinations, product

    # Primes that appear in factorizations of numbers 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    pmap = {p: i for i, p in enumerate(primes)}
    dim = len(primes)

    def factor(n):
        f = [0] * dim
        d = 2
        while d * d <= n:
            while n % d == 0:
                if d in pmap:
                    f[pmap[d]] += 1
                n //= d
            d += 1
        if n > 1 and n in pmap:
            f[pmap[n]] += 1
        return f

    # Build exponent vectors for all candidates
    vecs = {}
    for n in range(2, 41):
        vecs[n] = tuple(factor(n))

    # Gaussian elimination over rationals to find a basis
    # We represent vectors as lists of fractions or just use integer arithmetic
    # with a simple rank computation. Since vectors are small, we can use
    # sympy-free rational elimination via cross-multiplication or just use
    # the fact that we only need to check independence of small sets.
    #
    # Simpler: use the fact that a set of integer vectors is linearly independent
    # over Q iff the corresponding rational matrix has full row rank.
    # For small dimensions, we can do this with integer row reduction.

    def rank_of_vectors(vlist):
        """Compute rank of a list of integer vectors using Gaussian elimination with fractions."""
        if not vlist:
            return 0
        # Work with a copy as list of lists
        mat = [list(v) for v in vlist]
        rows = len(mat)
        cols = len(mat[0])
        rank = 0
        for col in range(cols):
            # Find pivot
            pivot = -1
            for r in range(rank, rows):
                if mat[r][col] != 0:
                    pivot = r
                    break
            if pivot == -1:
                continue
            # Swap
            mat[rank], mat[pivot] = mat[pivot], mat[rank]
            # Eliminate below
            for r in range(rank + 1, rows):
                if mat[r][col] != 0:
                    factor = mat[r][col]
                    pivot_factor = mat[rank][col]
                    for c in range(col, cols):
                        mat[r][c] = mat[r][c] * pivot_factor - mat[rank][c] * factor
                    # We don't normalize the pivot row to avoid fractions
                    # But we need to track that we've done elimination
                    # Actually, for rank computation with integers, we can just
                    # count non-zero rows after elimination. Let's use a cleaner approach.
            rank += 1
        # The above has issues with integer overflow and correctness. Let's use a
        # simpler approach: try all subsets of size up to dim and check determinant.
        # Actually, dim=12, so let's just use a proper rational-free method.
        return rank

    # Better: use the fact that for our purposes, we just need a maximal independent set.
    # Greedily build a basis: add a vector if it's independent of the current basis.
    # Independence check: the new vector is not in the span of the basis.
    # For small dim, we can check by solving a linear system.

    def is_independent(basis, v):
        """Check if v is linearly independent of the basis vectors (over Q)."""
        if not basis:
            return True
        # Build augmented matrix [basis | v] and check if v is in the span.
        # Equivalent: rank(basis) < rank(basis + [v])
        # For small sizes, we can use the determinant trick for square submatrices,
        # but basis size varies. Let's use a simple approach:
        # Try to express v as a linear combination of basis vectors.
        # This is a system of dim equations with len(basis) unknowns.
        # Since dim=12 and basis size <= 12, we can do Gaussian elimination on the
        # augmented system.
        
        # Actually, for our use case, we can just check: if we add v to basis,
        # does the rank increase? We can compute rank of basis and rank of basis+[v].
        # But computing rank is the hard part. Let's use a different approach:
        # Since all vectors are in Z^12, v is dependent on basis iff there exist
        # rational coefficients c_i such that sum c_i * basis_i = v.
        # This is equivalent to: the system A * c = v has a solution, where A is
        # the matrix with rows = basis vectors.
        
        # For simplicity and correctness, let's just use the subset-product check
        # directly for the final validation, and for the basis construction, use
        # a simpler heuristic: greedily add vectors that have a "new" prime
        # (i.e., a prime where all current basis vectors have exponent 0).
        
        # Even simpler: just use the known fact that the primes themselves
        # (2,3,5,7,11,13,17,19,23,29,31,37) form a basis of size 12.
        # So we can just use those as our initial basis!
        return True  # Not used in final approach

    # The primes 2,3,5,7,11,13,17,19,23,29,31,37 are linearly independent
    # (each has a unique prime factor with exponent 1 and all others 0).
    # So they form a valid basis of size 12.
    basis_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def has_distinct_subset_products(selected):
        """Check that all 2^len(selected) subset products are distinct."""
        k = len(selected)
        if k > 22:
            return False  # safety
        products = set()
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask >> i & 1:
                    prod *= selected[i]
            if prod in products:
                return False
            products.add(prod)
        return True

    # Start with the prime basis
    best_set = list(basis_primes)
    
    # Verify
    if not has_distinct_subset_products(best_set):
        # Shouldn't happen, but fallback
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Greedily add remaining numbers in multiple orders
    remaining = [n for n in range(2, 41) if n not in best_set]
    
    # Try several orderings
    orderings = [
        sorted(remaining),
        sorted(remaining, reverse=True),
        sorted(remaining, key=lambda n: sum(vecs[n])),  # by total exponent
        sorted(remaining, key=lambda n: max(vecs[n])),  # by max exponent
        sorted(remaining, key=lambda n: -sum(vecs[n])),  # by total exponent desc
    ]
    
    for order in orderings:
        trial = list(best_set)
        for n in order:
            if n in trial:
                continue
            if has_distinct_subset_products(trial + [n]):
                trial = trial + [n]
        if len(trial) > len(best_set):
            best_set = trial

    # Local improvement: try swapping elements
    improved = True
    while improved:
        improved = False
        current_size = len(best_set)
        for i in range(current_size):
            for n in range(2, 41):
                if n in best_set:
                    continue
                trial = best_set[:i] + best_set[i+1:] + [n]
                if has_distinct_subset_products(trial):
                    # Check if this actually helps (it should since same size,
                    # but maybe enables further additions)
                    # For now, only accept if we can then add more
                    extended = list(trial)
                    for m in range(2, 41):
                        if m in extended:
                            continue
                        if has_distinct_subset_products(extended + [m]):
                            extended = extended + [m]
                    if len(extended) > current_size:
                        best_set = extended
                        improved = True
                        break
            if improved:
                break

    return list(best_set)

# EVOLVE-BLOCK-END
