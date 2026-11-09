import time
import random

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    deadline = time.time() + 85.0
    best_seq = []
    best_len = 0
    
    # Strategy 1: Ascending Greedy (Seed)
    values, products = [], set()
    for x in range(2, 129):
        # Check if x can be appended
        # New products: x, values[-1]*x, values[-2]*values[-1]*x, etc.
        valid = True
        prod = x
        if prod in products:
            valid = False
        else:
            products.add(prod) # Tentatively add? No, better to collect and check all first.
            # Actually, if we append, the new products are:
            # x
            # values[-1] * x
            # values[-2] * values[-1] * x
            # ...
            # values[0] * ... * values[-1] * x
            
            # Let's just compute them and check.
            pass
            
        # Re-implementing append check properly
        new_prods = []
        prod = x
        new_prods.append(prod)
        for a in reversed(values):
            prod *= a
            new_prods.append(prod)
            
        if not products.intersection(new_prods):
            values.append(x)
            products.update(new_prods)
            
    if len(values) > best_len:
        best_len = len(values)
        best_seq = values

    # Strategy 2: Prime-based
    primes = [x for x in range(2, 129) if is_prime(x)]
    # Primes are always valid?
    # Product of primes is unique prime factorization.
    # Any subsequence product is a product of a subset of consecutive primes.
    # Can two different consecutive blocks have the same product?
    # No, because prime factorization is unique.
    # So primes are always a valid sequence.
    if len(primes) > best_len:
        best_len = len(primes)
        best_seq = primes

    # Strategy 3: Optimized Random Search / Completion
    def try_insert(seq, x, products_set):
        """
        Try to insert x into sorted seq.
        Returns (new_seq, new_products_set) if valid, else None.
        """
        n = len(seq)
        # Find insertion index k such that seq[k-1] < x < seq[k]
        # Since seq is sorted, we can use bisect
        import bisect
        k = bisect.bisect_left(seq, x)
        
        # If x is already in seq, invalid (but we check missing so x not in seq)
        
        # Compute L: products of suffixes ending at k-1
        # L[i] = seq[k-1-i] * ... * seq[k-1]
        # We want products: seq[i]...seq[k-1] for i from k-1 down to 0
        L = []
        prod = 1
        for i in range(k-1, -1, -1):
            prod *= seq[i]
            L.append(prod)
            
        # Compute R: products of prefixes starting at k
        # R[j] = seq[k] * ... * seq[k+j]
        # We want products: seq[k]...seq[j] for j from k to n-1
        R = []
        prod = 1
        for j in range(k, n):
            prod *= seq[j]
            R.append(prod)
            
        new_prods = set()
        
        # 1. x itself
        if x in products_set:
            return None
        new_prods.add(x)
        
        # 2. l * x for l in L
        for l in L:
            p = l * x
            if p in products_set:
                return None
            new_prods.add(p)
            
        # 3. x * r for r in R
        for r in R:
            p = x * r
            if p in products_set:
                return None
            new_prods.add(p)
            
        # 4. l * x * r for l in L, r in R
        for l in L:
            for r in R:
                p = l * x * r
                if p in products_set:
                    return None
                new_prods.add(p)
                
        # If we reach here, it's valid
        new_seq = seq[:k] + [x] + seq[k:]
        new_products_set = products_set.union(new_prods)
        
        return new_seq, new_products_set

    def extend_sequence(base_seq, iterations):
        current_best = list(base_seq)
        current_best_len = len(current_best)
        missing = [x for x in range(2, 129) if x not in current_best]
        
        # Pre-compute products for base_seq
        base_products = set()
        n = len(current_best)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= current_best[j]
                base_products.add(prod)
                
        current_products = base_products

        for _ in range(iterations):
            if time.time() > deadline:
                break
            random.shuffle(missing)
            temp_seq = list(current_best)
            temp_products = set(current_products)
            # Try to add each missing number in the shuffled order
            for x in missing:
                if time.time() > deadline:
                    break
                res = try_insert(temp_seq, x, temp_products)
                if res:
                    temp_seq, temp_products = res
            if len(temp_seq) > current_best_len:
                current_best = temp_seq
                current_best_len = len(current_best)
                current_products = temp_products
                missing = [x for x in range(2, 129) if x not in current_best]
                if not missing:
                    break
        return current_best

    # Extend Seed
    seed_extended = extend_sequence(best_seq, 2000) # More iterations possible now
    if len(seed_extended) > best_len:
        best_len = len(seed_extended)
        best_seq = seed_extended
        
    # Extend Primes
    if len(primes) > 0:
        prime_extended = extend_sequence(primes, 2000)
        if len(prime_extended) > best_len:
            best_len = len(prime_extended)
            best_seq = prime_extended

    return best_seq
