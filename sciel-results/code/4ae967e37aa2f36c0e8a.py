"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from functools import lru_cache

def _get_prime_factors(n):
    """Return a dictionary of prime factors and their exponents for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _check_distinct_subset_products(numbers):
    """
    Check if all 2^k subset products of 'numbers' are distinct.
    We do this by checking if all subset sums of the prime exponent vectors are distinct.
    """
    if not numbers:
        return True
    
    # Get prime factors for each number
    vecs = []
    for n in numbers:
        vf = _get_prime_factors(n)
        # Convert to a tuple of (prime, exp) pairs, sorted for consistent hashing
        # Actually, we can just sum the dictionaries for subsets.
        # To be efficient, we can represent each vector as a tuple of exponents for a fixed set of primes.
        # But the set of primes changes.
        # Simpler: Use a set of frozensets of (prime, exponent) pairs? No, summing is needed.
        # Let's just store the factor dict.
        vecs.append(vf)
    
    k = len(numbers)
    seen_products = set()
    
    # Enumerate all subsets
    # For k up to ~18, 2^18 = 262,144. This is fast enough.
    for mask in range(1 << k):
        # Calculate product for this subset
        # We can represent the product as a tuple of (prime, exponent) sorted.
        # Or just the integer product itself. Since numbers <= 40, product of 18 numbers <= 40^18 which is huge but Python handles big ints.
        # However, comparing big ints is slower than comparing small tuples.
        # Let's use the prime exponent vector approach.
        
        # Identify all primes involved
        # Actually, just accumulating the factor dict is easy.
        factor_sum = {}
        for i in range(k):
            if mask & (1 << i):
                for p, e in vecs[i].items():
                    factor_sum[p] = factor_sum.get(p, 0) + e
        
        # Create a canonical representation of this product
        # Tuple of (prime, exponent) sorted by prime
        prod_repr = tuple(sorted(factor_sum.items()))
        
        if prod_repr in seen_products:
            return False
        seen_products.add(prod_repr)
        
    return True

def _generate_candidates():
    """
    Generate candidate sets of integers.
    We focus on prime powers for small primes and singletons for large primes.
    """
    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # For each prime, determine available powers in 2..40
    # p=2: 2, 4, 8, 16, 32 (exponents 1, 2, 3, 4, 5)
    # p=3: 3, 9, 27 (exponents 1, 2, 3)
    # p=5: 5, 25 (exponents 1, 2)
    # p=7: 7 (exponent 1)
    # ...
    
    # We want to find the maximum size set.
    # Let's iterate over possible subsets of exponents for p=2, p=3, p=5.
    # For p >= 7, we can include the prime itself.
    # Wait, can we include composites?
    # The current best is 16. 
    # My hypothesis: Maybe we can get 17 or 18 by mixing?
    # E.g., if we drop a prime power, can we add a composite?
    # Composites are "expensive" in terms of dimension because they use multiple prime bases.
    # A prime power uses 1 prime base. A composite uses >= 2.
    # To maximize cardinality, we should prefer numbers that use "new" prime bases or minimal exponents.
    # Actually, the constraint is that subset products must be distinct.
    # This is equivalent to the vectors being "subset-sum unique".
    
    # Let's just search all combinations of prime powers.
    # This covers the "safe" space.
    # If the max is in this space, we find it.
    # If the max requires composites, this search will miss it, but we can add a heuristic check later.
    
    powers_by_prime = {
        2: [2, 4, 8, 16, 32],
        3: [3, 9, 27],
        5: [5, 25],
        7: [7],
        11: [11],
        13: [13],
        17: [17],
        19: [19],
        23: [23],
        29: [29],
        31: [31],
        37: [37]
    }
    
    best_set = []
    best_size = 0
    
    # Iterate over subsets of powers for 2, 3, 5
    # For 7, 11, ..., 37, we will always include the prime itself in the final candidate set
    # because they don't conflict with each other (distinct primes) and don't conflict with 2,3,5 powers
    # UNLESS we add composites involving them.
    # So, let's fix the set of "large primes" L = {7, 11, 13, 17, 19, 23, 29, 31, 37}.
    # We include all of them. Size = 9.
    # Now we need to find the best subset of {2,4,8,16,32, 3,9,27, 5,25} to add.
    
    base_set = [7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    candidates_2 = [2, 4, 8, 16, 32]
    candidates_3 = [3, 9, 27]
    candidates_5 = [5, 25]
    
    # Iterate over all subsets of candidates_2, candidates_3, candidates_5
    # 2^5 * 2^3 * 2^2 = 32 * 8 * 4 = 1024 combinations.
    # For each combination, form the full set and check validity.
    
    for mask2 in range(1 << len(candidates_2)):
        subset2 = [candidates_2[i] for i in range(len(candidates_2)) if mask2 & (1 << i)]
        for mask3 in range(1 << len(candidates_3)):
            subset3 = [candidates_3[i] for i in range(len(candidates_3)) if mask3 & (1 << i)]
            for mask5 in range(1 << len(candidates_5)):
                subset5 = [candidates_5[i] for i in range(len(candidates_5)) if mask5 & (1 << i)]
                
                current_set = base_set + subset2 + subset3 + subset5
                size = len(current_set)
                
                if size <= best_size:
                    continue
                
                if _check_distinct_subset_products(current_set):
                    best_size = size
                    best_set = current_set
    
    # Now, can we do better by swapping some base primes for composites?
    # E.g., instead of 7, use 14? No, 14 = 2*7. If we have 2, we can't have 14.
    # If we don't have 2, maybe we can have 14?
    # This requires a more general search.
    # Let's try a randomized local search starting from the best prime-power set found.
    
    start_set = best_set
    current_best = start_set
    current_size = len(start_set)
    
    # Define all possible numbers 2..40
    all_nums = list(range(2, 41))
    
    # Local search: try to add a number not in the set, or swap a number in the set with one not in the set.
    # Since we want to maximize size, we first try to ADD.
    # If we can't add, we try to SWAP (keep size same, but maybe enables a later add? No, size is fixed).
    # Actually, if we can't add any number, the size is local max.
    # Can we increase size by removing one and adding two?
    # That's a complex move.
    
    # Let's try a simple greedy add first.
    improved = True
    while improved:
        improved = False
        for n in all_nums:
            if n in current_best:
                continue
            new_set = current_best + [n]
            if _check_distinct_subset_products(new_set):
                current_best = new_set
                current_size += 1
                improved = True
                # Restart loop to try adding more
                break
    
    # Now try swaps to see if we can reach a different local maximum with same size?
    # Or maybe a swap allows a subsequent add?
    # Let's try 1-for-1 swaps.
    for i in range(len(current_best)):
        for n in all_nums:
            if n in current_best:
                continue
            # Swap current_best[i] with n
            test_set = current_best[:i] + [n] + current_best[i+1:]
            if _check_distinct_subset_products(test_set):
                # Check if this new set allows adding another number
                # This is expensive. Just accept the swap if it leads to a valid set of same size.
                # But we want to INCREASE size.
                # So, if we swap, we should immediately check if we can add.
                # Let's just collect all valid swaps of same size, and then try to add from each.
                # This might be too slow if done exhaustively.
                pass

    # Given the time budget, the exhaustive search over prime powers likely found the global max
    # because composites are "inefficient" in terms of the number of distinct subset products they generate
    # relative to the "dimension" they consume.
    # Prime powers are the most "efficient" generators of distinct products.
    
    # Let's double check if 17 or 18 is possible.
    # Max size from prime powers was 16.
    # Is it possible to have 17?
    # We have 12 primes.
    # If we use 12 primes (one power each), size is 12.
    # We can add more powers of 2, 3, 5.
    # Max additional powers for 2: 2 more (total 3).
    # Max additional powers for 3: 1 more (total 2).
    # Max additional powers for 5: 1 more (total 2).
    # Total 12 + 2 + 1 + 1 = 16.
    
    # What if we drop a prime, say 37, and add two composites involving 37?
    # E.g., 37 and 2*37=74 (out of range).
    # So we can't add composites of large primes.
    # What about composites of small primes?
    # We already have many powers of 2, 3, 5.
    # Adding a composite like 6 (2*3) creates a dependency.
    # If we have 2 and 3, we cannot have 6.
    # If we remove 2 and 3, and add 6, we lose 2 elements and gain 1. Net -1.
    # If we remove 2, keep 3, add 6?
    # 3 and 6. 3*6=18. 6 is 2*3.
    # Subset products: 1, 3, 6, 18. Distinct.
    # But we lost 2. Did we gain anything?
    # We have 3, 6.
    # Can we add 9? 3, 6, 9.
    # 3*6=18, 6*9=54, 3*9=27, 3*6*9=162.
    # Vectors:
    # 3: (0,1)
    # 6: (1,1)
    # 9: (0,2)
    # Sums:
    # 0
    # (0,1)
    # (1,1)
    # (0,2)
    # (0,1)+(1,1) = (1,2) -> 18
    # (0,1)+(0,2) = (0,3) -> 27
    # (1,1)+(0,2) = (1,3) -> 54
    # (0,1)+(1,1)+(0,2) = (1,4) -> 162
    # All distinct.
    # So {3, 6, 9} is valid. Size 3.
    # Compare to prime powers {2, 4, 16, 3, 9, 5, 25...}
    # For prime 2, we had 3 numbers.
    # For prime 3, we had 2 numbers.
    # Total 5 numbers from primes 2 and 3.
    # In {3, 6, 9}, we have 3 numbers from primes 2 and 3.
    # We lost 2 numbers.
    # So it seems prime powers are indeed optimal for maximizing count.
    
    # Therefore, the result from the exhaustive search over prime powers is likely the global maximum.
    
    return [int(x) for x in current_best]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    result = _generate_candidates()
    
    # Verify result
    if not _check_distinct_subset_products(result):
        # Fallback to a known good set
        result = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    return result
# EVOLVE-BLOCK-END
