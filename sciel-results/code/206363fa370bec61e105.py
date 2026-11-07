"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Factorize each number 2..40 into exponent vector
    def factorize(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    candidates = list(range(2, 41))
    vecs = {n: factorize(n) for n in candidates}
    
    # Greedy: try to build a maximal set with distinct subset products
    # Check: adding a new vector v to set S is valid iff
    # for all existing subset sums s, s+v is not already a subset sum
    # and s+v != 0 (but products are positive so this is fine)
    
    # Use a set of subset sums (as tuples of exponents)
    # Start with {zero vector}
    
    best_set = []
    
    # Try different orderings to find a good set
    # First, try adding primes (they're most "independent")
    # Then try prime powers and composites
    
    # Strategy: greedy with multiple restarts using different orderings
    # Order 1: primes first, then prime powers, then composites
    # Order 2: by "newness" - numbers that contribute new prime dimensions
    
    def is_valid_addition(current_sums, new_vec):
        """Check if adding new_vec keeps all subset sums distinct."""
        # New subset sums = old_sums union (old_sums + new_vec)
        # Need: no collision within old_sums + new_vec, and no overlap with old_sums
        new_sums = set()
        for s in current_sums:
            new_s = tuple(s[i] + new_vec[i] for i in range(len(primes)))
            if new_s in current_sums or new_s in new_sums:
                return False
            new_sums.add(new_s)
        return True
    
    def greedy_add(candidates_order, current_set, current_sums):
        """Try to add candidates in given order."""
        added = list(current_set)
        sums = set(current_sums)
        for n in candidates_order:
            if n in added:
                continue
            vec = vecs[n]
            if is_valid_addition(sums, vec):
                added.append(n)
                # Update sums
                new_sums = set()
                for s in sums:
                    new_s = tuple(s[i] + vec[i] for i in range(len(primes)))
                    new_sums.add(new_s)
                sums = sums | new_sums
        return added, sums
    
    # Multiple strategies
    strategies = []
    
    # Strategy 1: primes first (in order), then remaining sorted
    primes_list = [p for p in primes if p <= 40]
    remaining1 = [n for n in candidates if n not in primes_list]
    strategies.append(primes_list + remaining1)
    
    # Strategy 2: sort by number of prime factors (fewer first)
    def num_prime_factors(n):
        return sum(vecs[n])
    strategies.append(sorted(candidates, key=lambda n: (num_prime_factors(n), n)))
    
    # Strategy 3: primes, then prime powers of small primes, then rest
    prime_powers = [4, 9, 16, 25, 27, 32]  # prime powers in range
    pp_in_range = [pp for pp in prime_powers if 2 <= pp <= 40]
    rest3 = [n for n in candidates if n not in primes_list and n not in pp_in_range]
    strategies.append(primes_list + pp_in_range + rest3)
    
    # Strategy 4: by "rank" - prioritize numbers with unique prime factors
    # Primes with large values first (less likely to combine)
    strategies.append(sorted(primes_list, reverse=True) + 
                      sorted([n for n in candidates if n not in primes_list], 
                           key=lambda n: (num_prime_factors(n), -n)))
    
    # Strategy 5: all numbers sorted descending
    strategies.append(sorted(candidates, reverse=True))
    
    # Strategy 6: primes ascending, then composites by descending size
    strategies.append(primes_list + sorted([n for n in candidates if n not in primes_list], reverse=True))
    
    best_result = []
    best_len = 0
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        result, _ = greedy_add(strategy, [], {tuple(0 for _ in primes)})
        if len(result) > best_len:
            best_len = len(result)
            best_result = result
    
    # Try to improve: for the best set, try swapping out elements for better ones
    if best_result and time.time() < deadline:
        # Get the current sums
        current_sums = {tuple(0 for _ in primes)}
        for n in best_result:
            vec = vecs[n]
            new_sums = set()
            for s in current_sums:
                new_s = tuple(s[i] + vec[i] for i in range(len(primes)))
                new_sums.add(new_s)
            current_sums = current_sums | new_sums
        
        # Try adding each non-selected number
        selected = set(best_result)
        for n in sorted(candidates, key=lambda x: -num_prime_factors(x)):
            if n in selected:
                continue
            if is_valid_addition(current_sums, vecs[n]):
                best_result.append(n)
                # Update sums
                vec = vecs[n]
                new_sums = set()
                for s in current_sums:
                    new_s = tuple(s[i] + vec[i] for i in range(len(primes)))
                    new_sums.add(new_s)
                current_sums = current_sums | new_sums
                selected.add(n)
                if time.time() > deadline:
                    break
    
    # Verify the result
    def verify(result):
        products = {1}
        for n in result:
            new_products = set()
            for p in products:
                np = p * n
                if np in products or np in new_products:
                    return False
                new_products.add(np)
            products = products | new_products
        return True
    
    if not verify(best_result):
        # Fallback to primes only
        best_result = [p for p in primes if p <= 40]
    
    # Ensure Python ints
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
