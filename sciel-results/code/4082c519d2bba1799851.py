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
    prime_list = primes
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
    
    def check_sequence_valid(seq):
        """Check if a sequence has all distinct consecutive block products."""
        n = len(seq)
        if n == 0:
            return True
        prods = set()
        for i in range(n):
            vec = [0] * num_primes
            for j in range(i, n):
                ej = exp_vecs[seq[j]]
                for k in range(num_primes):
                    vec[k] += ej[k]
                tv = tuple(vec)
                if tv in prods:
                    return False
                prods.add(tv)
        return True
    
    def try_insert(seq, c):
        """Try to insert c into seq at best position. Returns new_seq or None."""
        n = len(seq)
        
        # Find valid insertion positions
        valid_positions = []
        for i in range(n + 1):
            if i > 0 and seq[i-1] >= c:
                continue
            if i < n and c >= seq[i]:
                continue
            valid_positions.append(i)
        
        if not valid_positions:
            return None
        
        # Sort by centrality
        mid = n / 2.0
        valid_positions.sort(key=lambda i: abs(i - mid))
        
        for i in valid_positions:
            new_seq = seq[:i] + [c] + seq[i:]
            if check_sequence_valid(new_seq):
                return new_seq
        
        return None
    
    def try_remove_and_reinsert(seq, c_to_remove, c_to_add):
        """Remove c_to_remove, then try to insert c_to_add."""
        if c_to_remove not in seq:
            return None
        new_seq = [x for x in seq if x != c_to_remove]
        result = try_insert(new_seq, c_to_add)
        if result is not None:
            return result
        # Also try just removing
        return new_seq if len(new_seq) < len(seq) else None
    
    def local_search(seq, max_iter=500):
        """Improve sequence through local search."""
        current = seq[:]
        current_len = len(current)
        used = set(current)
        remaining = [x for x in all_numbers if x not in used]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            
            # Try insertions
            random.shuffle(remaining)
            for c in remaining:
                new_seq = try_insert(current, c)
                if new_seq is not None:
                    current = new_seq
                    current_len = len(current)
                    used = set(current)
                    remaining = [x for x in all_numbers if x not in used]
                    improved = True
                    break
            
            # Try swaps: remove one, add another
            if not improved:
                for i in range(len(current)):
                    for c in remaining:
                        new_seq = try_remove_and_reinsert(current, current[i], c)
                        if new_seq is not None and len(new_seq) >= current_len:
                            current = new_seq
                            current_len = len(current)
                            used = set(current)
                            remaining = [x for x in all_numbers if x not in used]
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current
    
    # Build initial sequences using different strategies
    def build_initial(seed_val):
        """Build an initial sequence using randomized greedy."""
        rng = random.Random(seed_val)
        
        # Strategy: start with subset of primes, then add composites
        # Try different subsets of primes
        prime_subset_size = rng.randint(10, len(primes))
        prime_subset = rng.sample(primes, prime_subset_size)
        seq = sorted(prime_subset)
        
        if not check_sequence_valid(seq):
            seq = primes[:]
        
        used = set(seq)
        remaining = [x for x in all_numbers if x not in used]
        rng.shuffle(remaining)
        
        progress = True
        while progress and time.time() < deadline:
            progress = False
            for c in remaining[:]:
                new_seq = try_insert(seq, c)
                if new_seq is not None:
                    seq = new_seq
                    remaining.remove(c)
                    progress = True
                    break
        
        return seq
    
    best_seq = primes[:]
    best_len = len(best_seq)
    
    # Try multiple diverse starting points
    seeds_tried = set()
    while time.time() < deadline:
        seed_val = random.randint(0, 2**31 - 1)
        if seed_val in seeds_tried:
            continue
        seeds_tried.add(seed_val)
        
        seq = build_initial(seed_val)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
        
        # Local search on the best sequence found so far
        if len(seq) >= best_len:
            improved = local_search(seq)
            if len(improved) > best_len:
                best_len = len(improved)
                best_seq = improved[:]
    
    # Final local search on best sequence
    if time.time() < deadline:
        final = local_search(best_seq, max_iter=1000)
        if len(final) > best_len:
            best_seq = final
    
    return best_seq
