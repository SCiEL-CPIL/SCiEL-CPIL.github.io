import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i in range(2, n + 1) if sieve[i]]

    def prime_factors_set(x):
        """Return set of distinct prime factors of x."""
        factors = set()
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return factors

    def is_valid(seq):
        """Check if all consecutive block products are distinct."""
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

    def get_all_products(seq):
        """Return set of all consecutive block products."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    def is_valid_insert(seq, x, pos):
        """
        Fast check: is inserting x at position pos valid?
        Only need to check products that include x (new products).
        Products not including x are unchanged and already valid.
        """
        n = len(seq)
        # Pre-compute products of existing sequence for reference
        # Actually, we need to check that no new product (containing x) 
        # equals any existing product or another new product.
        
        # Get all existing products
        existing_products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                existing_products.add(prod)
        
        # Check all new products that include x
        # A product including x spans from some i <= pos to some j >= pos
        # In the new sequence, x is at index pos
        # Products including x: seq[i..pos-1] * x * seq[pos..j] for i in [0,pos], j in [pos, n-1]
        # (using 0-indexed in new sequence where x is at pos)
        
        new_products = set()
        for i in range(pos + 1):  # start index in new sequence (0 to pos)
            left_prod = 1
            for k in range(i, pos):
                left_prod *= seq[k]
            for j in range(pos, n):  # end index in new sequence (pos to n-1)
                right_prod = 1
                for k in range(pos, j + 1):
                    right_prod *= seq[k]
                p = left_prod * x * right_prod
                if p in existing_products or p in new_products:
                    return False
                new_products.add(p)
        return True

    def is_valid_insert_fast(seq, x, pos, existing_products):
        """
        Fast check using pre-computed existing products.
        """
        n = len(seq)
        new_products = set()
        for i in range(pos + 1):
            left_prod = 1
            for k in range(i, pos):
                left_prod *= seq[k]
            for j in range(pos, n):
                right_prod = 1
                for k in range(pos, j + 1):
                    right_prod *= seq[k]
                p = left_prod * x * right_prod
                if p in existing_products or p in new_products:
                    return False
                new_products.add(p)
        return True

    # Pre-compute prime factor sets for all numbers
    pf_sets = {x: prime_factors_set(x) for x in range(2, 129)}
    primes = get_primes(128)
    all_nums = list(range(2, 129))

    # --- Greedy construction: start with primes, add composites by conflict score ---
    current_set = set(primes)
    current_seq = sorted(current_set)
    existing_products = get_all_products(current_seq)

    composites = [x for x in all_nums if x not in current_set]

    # Score composites: fewer shared prime factors with current set = less conflict
    def conflict_score(x):
        shared = 0
        for y in current_set:
            shared += len(pf_sets[x] & pf_sets[y])
        return shared

    composites.sort(key=lambda x: (conflict_score(x), x))

    for x in composites:
        if time.time() > deadline - 10:
            break
        # Find correct insertion position
        pos = 0
        while pos < len(current_seq) and current_seq[pos] < x:
            pos += 1
        if is_valid_insert_fast(current_seq, x, pos, existing_products):
            # Insert
            new_seq = current_seq[:pos] + [x] + current_seq[pos:]
            current_set.add(x)
            # Update existing products
            existing_products = get_all_products(new_seq)
            current_seq = new_seq

    best_seq = current_seq[:]
    best_len = len(best_seq)
    current_set = set(best_seq)
    existing_products = get_all_products(best_seq)

    # --- Simulated Annealing ---
    T = 50.0
    cooling = 0.9995
    min_T = 0.5

    while time.time() < deadline - 5:
        # Generate a move
        move_type = random.choices(['add', 'remove', 'swap'], weights=[0.35, 0.1, 0.55])[0]
        new_set = set(current_set)

        if move_type == 'add':
            candidates = [x for x in all_nums if x not in current_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_set.add(x)
        elif move_type == 'remove':
            if not current_set:
                continue
            x = random.choice(list(current_set))
            new_set.discard(x)
        else:  # swap
            candidates = [x for x in all_nums if x not in current_set]
            if not candidates or not current_set:
                continue
            x_out = random.choice(list(current_set))
            x_in = random.choice(candidates)
            new_set.discard(x_out)
            new_set.add(x_in)

        new_seq = sorted(new_set)
        new_len = len(new_seq)

        # Compute score: length if valid, heavily penalized if not
        if new_len == len(best_seq) or new_len > len(best_seq):
            # Check validity
            if is_valid(new_seq):
                new_score = new_len
            else:
                new_score = -10000 + new_len
        else:
            # Shorter sequence - only accept if we're stuck (but prefer to avoid)
            if is_valid(new_seq):
                new_score = new_len - 100  # penalty for shorter
            else:
                new_score = -10000 + new_len

        # Current score
        if is_valid(current_seq):
            current_score = len(current_seq)
        else:
            current_score = -10000 + len(current_seq)

        delta = new_score - current_score

        if delta > 0 or (T > min_T and random.random() < math.exp(delta / T)):
            current_set = new_set
            current_seq = new_seq
            existing_products = get_all_products(new_seq)

            if len(new_seq) > best_len:
                best_len = len(new_seq)
                best_seq = new_seq[:]

        T *= cooling
        if T < min_T:
            T = min_T

    # --- Final exhaustive polish: try all single additions repeatedly ---
    current_set = set(best_seq)
    current_seq = sorted(current_set)
    existing_products = get_all_products(current_seq)

    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for x in all_nums:
            if x in current_set:
                continue
            pos = 0
            while pos < len(current_seq) and current_seq[pos] < x:
                pos += 1
            if is_valid_insert_fast(current_seq, x, pos, existing_products):
                new_seq = current_seq[:pos] + [x] + current_seq[pos:]
                current_set.add(x)
                current_seq = new_seq
                existing_products = get_all_products(new_seq)
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq[:]
                improved = True
                break

    # One more pass with full validity check to be safe
    if not is_valid(best_seq):
        # Fallback: use a simple valid sequence
        best_seq = sorted(current_set)
        if not is_valid(best_seq):
            best_seq = list(range(2, 129))  # Should be valid as fallback? No, but at least returns something
            # Actually let's just return the best we have that's valid
            # Find longest valid prefix
            for end in range(len(best_seq), 0, -1):
                if is_valid(best_seq[:end]):
                    best_seq = best_seq[:end]
                    break

    return best_seq
