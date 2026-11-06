import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

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

    all_numbers = list(range(2, 129))
    primes = [p for p in all_numbers if is_prime(p)]
    composites = [c for c in all_numbers if not is_prime(c)]

    def num_distinct_prime_factors(n):
        count = 0
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count += 1
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            count += 1
        return count

    # Precompute factor counts for all composites
    factor_counts = {c: num_distinct_prime_factors(c) for c in composites}

    def compute_all_products(seq):
        """Compute all consecutive block products as a set."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    def can_insert_at(seq, all_prods, pos, x):
        """Check if inserting x at position pos keeps validity. Returns (valid, new_all_prods)"""
        n = len(seq)
        prefix = [1] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] * seq[i]
        suffix = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] * seq[i]
        
        new_prods = set()
        for i in range(pos + 1):
            left_prod = 1
            if i < pos:
                left_prod = prefix[pos] // prefix[i]
            for j in range(pos, n + 1):
                right_prod = 1
                if j > pos:
                    right_prod = suffix[pos] // suffix[j]
                p = left_prod * x * right_prod
                if p in all_prods or p in new_prods:
                    return False, None
                new_prods.add(p)
        return True, all_prods | new_prods

    def greedy_build_multi(base_seq, composite_order, max_passes=3):
        """Build sequence by greedily inserting composites, with multiple passes."""
        seq = list(base_seq)
        all_prods = compute_all_products(seq)
        
        for _pass in range(max_passes):
            if time.time() > deadline:
                break
            progress = False
            for c in composite_order:
                if time.time() > deadline:
                    break
                if c in seq:
                    continue
                
                # Find valid insertion positions (must maintain strict increase)
                lo, hi = 0, len(seq)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if seq[mid] < c:
                        lo = mid + 1
                    else:
                        hi = mid
                insert_pos = lo
                
                # Try insert at natural position first, then nearby positions
                tried = set()
                for pos in [insert_pos, insert_pos - 1, insert_pos + 1, insert_pos - 2, insert_pos + 2]:
                    if pos in tried or pos < 0 or pos > len(seq):
                        continue
                    tried.add(pos)
                    
                    # Check strict increase
                    if pos > 0 and seq[pos - 1] >= c:
                        continue
                    if pos < len(seq) and seq[pos] <= c:
                        continue
                    
                    valid, new_prods = can_insert_at(seq, all_prods, pos, c)
                    if valid:
                        seq = seq[:pos] + [c] + seq[pos:]
                        all_prods = new_prods
                        progress = True
                        break
            if not progress:
                break
        
        return seq

    # Start with all primes (known valid, length 30)
    best_seq = sorted(primes)
    best_len = len(best_seq)

    # Multiple ordering strategies
    orders = []
    # Fewer distinct prime factors = less likely to cause collisions
    orders.append(sorted(composites, key=lambda x: (factor_counts[x], x)))
    orders.append(sorted(composites, key=lambda x: (factor_counts[x], -x)))
    orders.append(sorted(composites, key=lambda x: x))
    orders.append(sorted(composites, key=lambda x: -x))
    # Composites with exactly 2 prime factors (semiprimes) are good candidates
    orders.append(sorted([c for c in composites if factor_counts[c] == 2], key=lambda x: x) + 
                  sorted([c for c in composites if factor_counts[c] == 3], key=lambda x: x) +
                  sorted([c for c in composites if factor_counts[c] > 3], key=lambda x: x))
    # Alternating low/high
    orders.append(sorted(composites, key=lambda x: (x % 2, factor_counts[x], x)))

    for order in orders:
        if time.time() > deadline - 5.0:
            break
        seq = greedy_build_multi(primes, order, max_passes=4)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq

    # Try removing one prime and rebuilding
    if time.time() < deadline - 10.0:
        for i in range(len(primes)):
            if time.time() > deadline - 5.0:
                break
            seq_without = primes[:i] + primes[i+1:]
            for order in orders[:3]:
                if time.time() > deadline - 5.0:
                    break
                seq = greedy_build_multi(seq_without, order, max_passes=3)
                if len(seq) > best_len:
                    best_len = len(seq)
                    best_seq = seq

    # Local search: try to add more composites to current best
    if time.time() < deadline - 3.0:
        current = list(best_seq)
        # Sort composites not in current by factor count
        missing = [c for c in composites if c not in current]
        missing.sort(key=lambda x: (factor_counts[x], x))
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            current_prods = compute_all_products(current)
            for c in missing:
                if time.time() > deadline - 3.0:
                    break
                lo, hi = 0, len(current)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if current[mid] < c:
                        lo = mid + 1
                    else:
                        hi = mid
                insert_pos = lo
                
                for pos in range(max(0, insert_pos - 3), min(len(current) + 1, insert_pos + 4)):
                    if pos > 0 and current[pos - 1] >= c:
                        continue
                    if pos < len(current) and current[pos] <= c:
                        continue
                    
                    valid, new_prods = can_insert_at(current, current_prods, pos, c)
                    if valid:
                        current = current[:pos] + [c] + current[pos:]
                        current_prods = new_prods
                        missing.remove(c)
                        improved = True
                        if len(current) > best_len:
                            best_len = len(current)
                            best_seq = current
                        break

    return sorted(best_seq)
