import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def is_valid(seq):
        if not seq:
            return True
        prods = set()
        for i in range(len(seq)):
            p = 1
            for j in range(i, len(seq)):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    # 1. Start with all primes in [2, 128]
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [i for i in range(2, 129) if is_prime(i)]
    composites = [i for i in range(2, 129) if not is_prime(i)]
    
    best_seq = primes[:]
    
    # 2. Greedily insert composites
    # Sort composites by number of prime factors? Or just value?
    # Let's try value ascending first.
    # We can also try multiple orders or random restarts if time permits.
    
    def try_insert(seq, x):
        # Try inserting x at all positions
        # seq is sorted
        for i in range(len(seq) + 1):
            # Check if x fits in sorted order
            if i > 0 and seq[i-1] >= x:
                continue
            if i < len(seq) and seq[i] <= x:
                continue
            
            new_seq = seq[:i] + [x] + seq[i:]
            if is_valid(new_seq):
                return new_seq
        return None

    # Try inserting composites
    # To maximize length, we should prioritize composites that are "easy" to insert.
    # Maybe those with larger gaps?
    
    # We will iterate through composites.
    # If a composite fails, it might succeed later if other composites are added?
    # Unlikely, adding more numbers makes it harder to insert new ones (more products).
    # So we should try to add as many as possible.
    
    # Let's try a few different orders for composites to see if one yields more.
    # Order 1: Value ascending
    # Order 2: Value descending
    # Order 3: By number of distinct prime factors (fewer first?)
    
    def num_distinct_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    orders = [
        sorted(composites),
        sorted(composites, reverse=True),
        sorted(composites, key=num_distinct_prime_factors),
        sorted(composites, key=lambda x: -num_distinct_prime_factors(x))
    ]
    
    best_len = len(best_seq)
    
    for order in orders:
        if time.time() > deadline - 10.0:
            break
            
        current_seq = primes[:]
        
        for x in order:
            if time.time() > deadline - 5.0:
                break
                
            # Optimization: if x is already in current_seq, skip (not possible as we start with primes)
            # Check if x can be inserted
            new_seq = try_insert(current_seq, x)
            if new_seq is not None:
                current_seq = new_seq
        
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq

    # If we have time, try to improve best_seq with random swaps or removals/insertions?
    # The greedy approach is deterministic.
    # Let's add a simple local search: try removing one element and adding two?
    # Or just return best_seq.
    
    return best_seq
