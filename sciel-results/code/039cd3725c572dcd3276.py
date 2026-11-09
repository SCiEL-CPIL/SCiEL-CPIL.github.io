import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    # Precompute primes up to 128
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    composites = [x for x in all_numbers if not is_prime(x)]
    
    # Precompute prime exponent vectors for all numbers 2..128
    # primes up to 128: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113
    prime_list = primes  # list of all primes <= 128
    num_primes = len(prime_list)
    prime_to_idx = {p: i for i, p in enumerate(prime_list)}
    
    # Exponent vector for each number
    exp_vecs = {}
    for n in all_numbers:
        vec = [0] * num_primes
        temp = n
        for pi, p in enumerate(prime_list):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[pi] += 1
                temp //= p
        if temp > 1:
            vec[prime_to_idx[temp]] += 1
        exp_vecs[n] = tuple(vec)
    
    # Fast product-check using prime exponent vectors (addition instead of multiplication)
    def get_product_set(seq):
        """Return set of all consecutive block products as exponent vectors."""
        prods = set()
        n = len(seq)
        for i in range(n):
            vec = [0] * num_primes
            for j in range(i, n):
                ej = exp_vecs[seq[j]]
                for k in range(num_primes):
                    vec[k] += ej[k]
                prods.add(tuple(vec))
        return prods
    
    def try_insertion(seq, c):
        """Try to insert c into seq maintaining strict increase and unique block products."""
        n = len(seq)
        
        # Find valid insertion positions (where seq[i-1] < c < seq[i])
        valid_positions = []
        for i in range(n + 1):
            if i > 0 and seq[i-1] >= c:
                continue
            if i < n and c >= seq[i]:
                continue
            valid_positions.append(i)
        
        if not valid_positions:
            return None, False
        
        # Sort positions by "centrality" - prefer middle positions
        # Middle positions create fewer new block products
        mid = n / 2.0
        valid_positions.sort(key=lambda i: abs(i - mid))
        
        for i in valid_positions:
            # Quick check: build new seq and verify products using exponent vectors
            new_seq = seq[:i] + [c] + seq[i:]
            m = len(new_seq)
            expected_count = m * (m + 1) // 2
            prods = set()
            valid = True
            for a in range(m):
                vec = [0] * num_primes
                for b in range(a, m):
                    eb = exp_vecs[new_seq[b]]
                    for k in range(num_primes):
                        vec[k] += eb[k]
                    tv = tuple(vec)
                    if tv in prods:
                        valid = False
                        break
                    prods.add(tv)
                if not valid:
                    break
            if valid and len(prods) == expected_count:
                return new_seq, True
        
        return None, False
    
    # Strategy: multiple randomized greedy builds, then try to improve
    def build_seq(seed_order):
        """Build a sequence starting with primes, then greedily add composites."""
        seq = primes[:]
        used = set(seq)
        remaining = [x for x in composites if x not in used]
        
        # Apply seed_order shuffle
        random.Random(seed_order).shuffle(remaining)
        
        progress = True
        while progress and time.time() < deadline:
            progress = False
            for c in remaining[:]:
                new_seq, success = try_insertion(seq, c)
                if success:
                    seq = new_seq
                    remaining.remove(c)
                    progress = True
                    break
        return seq
    
    best_seq = primes[:]
    best_len = len(best_seq)
    
    # Try many different random seeds/orders
    seeds_used = 0
    while time.time() < deadline:
        # Generate a new random seed
        seed_val = random.randint(0, 2**31 - 1)
        seeds_used += 1
        seq = build_seq(seed_val)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
            # Try to further improve best_seq with local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                used = set(best_seq)
                remaining2 = [x for x in composites if x not in used]
                random.shuffle(remaining2)
                for c in remaining2:
                    new_seq, success = try_insertion(best_seq, c)
                    if success:
                        best_seq = new_seq
                        best_len = len(best_seq)
                        improved = True
                        break
    
    return best_seq
