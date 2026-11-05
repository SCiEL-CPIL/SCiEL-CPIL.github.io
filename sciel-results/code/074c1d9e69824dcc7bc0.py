"""Find longest strictly increasing sequence in [2,128] with distinct consecutive block products."""
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    def _prime_factors(n):
        """Return set of distinct prime factors of n."""
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    def _total_prime_factors(n):
        """Return total number of prime factors with multiplicity."""
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
    
    def get_all_products(seq):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products
    
    def is_valid(seq):
        """Check if strictly increasing and all consecutive block products are distinct."""
        if not seq:
            return True
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        products = get_all_products(seq)
        return len(products) == len(seq) * (len(seq) + 1) // 2
    
    def greedy_with_order(order):
        """Greedy: process numbers in given order, append if no product collision."""
        values = []
        products = set()
        for x in order:
            if x < 2 or x > 128:
                continue
            new_products = {x}
            prod = x
            for a in reversed(values):
                prod *= a
                new_products.add(prod)
            if not products.intersection(new_products):
                values.append(x)
                products.update(new_products)
        values.sort()
        return values
    
    candidates = list(range(2, 129))
    
    # Precompute prime factor info
    pf_info = {x: (_total_prime_factors(x), len(_prime_factors(x))) for x in candidates}
    
    # Try multiple orderings
    orderings = [
        sorted(candidates),  # natural order
        sorted(candidates, key=lambda x: (pf_info[x][0], pf_info[x][1], x)),  # fewest total PF
        sorted(candidates, key=lambda x: (pf_info[x][1], pf_info[x][0], x)),  # fewest distinct PF
        sorted(candidates, key=lambda x: (pf_info[x][0] == 1 and 0 or 1, x)),  # primes first
    ]
    
    best_seq = []
    best_len = 0
    
    for order in orderings:
        if time.time() - start_time > DEADLINE:
            break
        seq = greedy_with_order(order)
        if is_valid(seq) and len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Iterative insertion: try to add one element at a time
    def try_insert(seq, elem, pos):
        """Try inserting elem at position pos. Return new seq if valid, else None."""
        if pos < 0 or pos > len(seq):
            return None
        new_seq = seq[:pos] + [elem] + seq[pos:]
        # Check strictly increasing
        for i in range(len(new_seq) - 1):
            if new_seq[i] >= new_seq[i + 1]:
                return None
        if not is_valid(new_seq):
            return None
        return new_seq
    
    used = set(best_seq)
    remaining = [x for x in range(2, 129) if x not in used]
    
    for _ in range(100):
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        for elem in remaining[:]:
            if time.time() - start_time > DEADLINE:
                break
            for pos in range(len(best_seq) + 1):
                result = try_insert(best_seq, elem, pos)
                if result is not None:
                    best_seq = result
                    best_len = len(result)
                    used.add(elem)
                    remaining.remove(elem)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # Final validation
    if not is_valid(best_seq):
        best_seq = greedy_with_order(sorted(candidates))
    
    return best_seq
