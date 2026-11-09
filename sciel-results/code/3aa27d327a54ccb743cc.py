import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    all_numbers = list(range(2, 129))

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

    primes = [x for x in all_numbers if is_prime(x)]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Exponent vector for each number
    exp_vecs = {}
    for n in all_numbers:
        vec = [0] * num_primes
        temp = n
        for pi, p in enumerate(primes):
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

    def get_block_products(seq):
        """Return set of all consecutive block product vectors for a sequence."""
        n = len(seq)
        prods = set()
        for i in range(n):
            vec = [0] * num_primes
            for j in range(i, n):
                ej = exp_vecs[seq[j]]
                for k in range(num_primes):
                    vec[k] += ej[k]
                prods.add(tuple(vec))
        return prods

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
            new_prods = get_block_products(new_seq)
            if len(new_prods) == len(new_seq) * (len(new_seq) + 1) // 2:
                return new_seq
        
        return None

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
                        # Remove current[i]
                        new_seq = current[:i] + current[i+1:]
                        # Try to insert c
                        result = try_insert(new_seq, c)
                        if result is not None and len(result) >= current_len:
                            current = result
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

    def build_initial(seed_val):
        """Build an initial sequence using randomized greedy."""
        rng = random.Random(seed_val)
        
        # Strategy: start with subset of primes, then add composites
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

    # Build initial sequences using different strategies
    def build_initial_v2(seed_val):
        """Alternative: start with all primes, then greedily add."""
        rng = random.Random(seed_val)
        seq = primes[:]
        
        if not check_sequence_valid(seq):
            # Try different subsets of primes
            for subset_size in range(len(primes), 5, -1):
                for _ in range(10):
                    subset = rng.sample(primes, subset_size)
                    s = sorted(subset)
                    if check_sequence_valid(s):
                        seq = s
                        break
                if check_sequence_valid(seq):
                    break
        
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

    def build_initial_v3(seed_val):
        """Alternative: start with smallest numbers, grow."""
        rng = random.Random(seed_val)
        seq = []
        for n in all_numbers:
            if time.time() > deadline:
                break
            new_seq = try_insert(seq, n)
            if new_seq is not None:
                seq = new_seq
        
        return seq

    # ========== MAIN SEARCH ==========
    best_seq = []
    best_len = 0
    
    # Try multiple diverse starting points
    for seed in range(200):
        if time.time() > deadline:
            break
        
        if seed % 3 == 0:
            seq = build_initial(seed)
        elif seed % 3 == 1:
            seq = build_initial_v2(seed)
        else:
            seq = build_initial_v3(seed)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
        
        # Local search on the sequence found
        improved = local_search(seq, max_iter=30)
        if len(improved) > best_len:
            best_len = len(improved)
            best_seq = improved[:]
    
    # Final local search on best sequence
    if time.time() < deadline:
        final = local_search(best_seq, max_iter=200)
        if len(final) > best_len:
            best_seq = final
    
    return best_seq
