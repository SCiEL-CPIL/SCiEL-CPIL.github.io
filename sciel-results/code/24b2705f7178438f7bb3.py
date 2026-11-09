import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 57.0  # 3s margin from 60s limit

    # Fallback: known valid set of 12 elements
    fallback_set = [
        2, 4, 16,   # 2^1, 2^2, 2^4
        3, 9,       # 3^1, 3^2
        5, 25,      # 5^1, 5^2
        7,          # 7^1
        11,         # 11^1
        13,         # 13^1
        17,         # 17^1
        19,         # 19^1
        23,         # 23^1
        29,         # 29^1
        31,         # 31^1
        37          # 37^1
    ]

    # Try to find a larger set via backtracking search
    # Candidate elements: all integers 2..40
    candidates = list(range(2, 41))

    best_set = list(fallback_set)
    best_size = len(fallback_set)

    # For efficient subset product checking, we track the set of all subset products
    # A new element x is compatible with current set S iff
    # for all p in subset_products(S), p*x is not in subset_products(S)
    # and x is not in subset_products(S)

    def check_compatibility(current_products, x):
        """Check if x can be added given the set of subset products."""
        # x must not be in current products (except 1, which is the empty product)
        if x in current_products:
            return False
        # For each existing product p, p*x must not be in current_products
        for p in current_products:
            if p * x in current_products:
                return False
        return True

    def add_element(current_products, x):
        """Add x to the set and update the subset products."""
        new_products = set(current_products)
        for p in list(current_products):
            new_products.add(p * x)
        return new_products

    # Backtracking search: try to build a set of size > best_size
    # We use a heuristic: prefer smaller numbers and numbers with few prime factors
    # to leave room for more elements

    # Pre-compute prime factorizations for ordering
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    # Order candidates: prefer those with fewer prime factors (prime powers are good)
    # and smaller values
    ordered_candidates = sorted(candidates, key=lambda x: (num_prime_factors(x), x))

    # State for backtracking
    def backtrack(selected, current_products, remaining_candidates, depth):
        nonlocal best_set, best_size, deadline

        if time.time() > deadline:
            return

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)

        # If we can't beat best_size even by adding all remaining, prune
        if len(selected) + len(remaining_candidates) <= best_size:
            return

        for i, x in enumerate(remaining_candidates):
            if time.time() > deadline:
                return

            if check_compatibility(current_products, x):
                new_products = add_element(current_products, x)
                selected.append(x)
                backtrack(selected, new_products, remaining_candidates[i+1:], depth + 1)
                selected.pop()

    # Start with empty set
    initial_products = {1}
    backtrack([], initial_products, ordered_candidates, 0)

    if time.time() > deadline:
        return fallback_set

    return best_set
