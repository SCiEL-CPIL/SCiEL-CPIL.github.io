import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    rng = random.Random(12345)
    
    # Precompute primes up to 128 and their exponent vectors
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
    
    # Map each number to its prime factorization as a tuple of (prime, exponent)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    def factorize(n):
        """Return exponent vector as tuple of length num_primes."""
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    # Precompute factorization for all numbers
    factorization = {n: factorize(n) for n in all_numbers}
    
    def add_vecs(a, b):
        """Add two exponent vectors."""
        return tuple(a[i] + b[i] for i in range(num_primes))
    
    def get_block_products(seq):
        """Return set of all consecutive block products as exponent vectors."""
        prods = set()
        n = len(seq)
        for i in range(n):
            vec = [0] * num_primes
            for j in range(i, n):
                fv = factorization[seq[j]]
                for k in range(num_primes):
                    vec[k] += fv[k]
                t = tuple(vec)
                prods.add(t)
        return prods
    
    def try_insertion_fast(seq, c):
        """Try to insert c into seq maintaining strict increase and unique block products.
        Uses incremental product computation for speed."""
        n = len(seq)
        
        # Find valid insertion positions
        for i in range(n + 1):
            if i > 0 and seq[i-1] >= c:
                continue
            if i < n and c >= seq[i]:
                continue
            
            # Quick check: does inserting c at position i create any duplicate?
            # We need to check all blocks that include c
            # Blocks ending at i-1 (before c) and starting at i (after c)
            # New blocks: [i..j] for j >= i, [k..i-1] for k <= i-1, and combined [k..j]
            
            # Compute suffix products from position i-1 backwards
            suffix_prods = {}  # (start, end) -> product vector
            # Actually, let's just build the new sequence and check all products
            # But do it incrementally
            
            new_seq = seq[:i] + [c] + seq[i:]
            m = len(new_seq)
            expected_count = m * (m + 1) // 2
            prods = set()
            valid = True
            
            for a in range(m):
                vec = [0] * num_primes
                for b in range(a, m):
                    fv = factorization[new_seq[b]]
                    for k in range(num_primes):
                        vec[k] += fv[k]
                    t = tuple(vec)
                    if t in prods:
                        valid = False
                        break
                    prods.add(t)
                if not valid:
                    break
            
            if valid:
                return new_seq
        
        return None
    
    def build_greedy(seed_order):
        """Build a sequence starting with primes, then greedily add composites."""
        local_rng = random.Random(seed_order)
        seq = primes[:]
        used = set(seq)
        remaining = [x for x in composites if x not in used]
        
        # Sort remaining by "ease" - prefer numbers with fewer prime factors first
        def complexity(n):
            return sum(1 for v in factorization[n] if v > 0)
        
        remaining.sort(key=lambda x: (complexity(x), x))
        local_rng.shuffle(remaining)
        
        progress = True
        while progress and time.time() < deadline:
            progress = False
            for c in remaining[:]:
                new_seq = try_insertion_fast(seq, c)
                if new_seq is not None:
                    seq = new_seq
                    remaining.remove(c)
                    progress = True
                    break
        return seq
    
    def local_search(seq, time_limit=5.0):
        """Try to add more elements to seq via local search with removal-based escape."""
        improved = True
        limit_time = time.time() + time_limit
        no_improve_count = 0
        while improved and time.time() < limit_time and time.time() < deadline:
            improved = False
            used = set(seq)
            remaining = [x for x in composites if x not in used]
            rng.shuffle(remaining)
            
            # Try adding each remaining element
            for c in remaining:
                new_seq = try_insertion_fast(seq, c)
                if new_seq is not None:
                    seq = new_seq
                    improved = True
                    no_improve_count = 0
                    break
            
            if not improved:
                no_improve_count += 1
                # If stuck, try removing a random element to explore different subsets
                if no_improve_count >= 3 and len(seq) > len(primes) + 2:
                    remove_idx = rng.randint(0, len(seq) - 1)
                    # Prefer removing composites over primes
                    composites_in_seq = [i for i, x in enumerate(seq) if x not in primes]
                    if composites_in_seq:
                        remove_idx = rng.choice(composites_in_seq)
                    seq = seq[:remove_idx] + seq[remove_idx+1:]
                    no_improve_count = 0
        return seq
    
    def backtrack_csp(seq, available, depth, max_depth, time_limit):
        """CSP backtracking: try to extend seq by adding elements from available."""
        if time.time() > time_limit or depth >= max_depth:
            return seq
        
        best = seq[:]
        for idx, c in enumerate(available):
            if time.time() > time_limit:
                break
            new_seq = try_insertion_fast(seq, c)
            if new_seq is not None:
                # Recurse
                remaining = available[:idx] + available[idx+1:]
                result = backtrack_csp(new_seq, remaining, depth + 1, max_depth, time_limit)
                if len(result) > len(best):
                    best = result
        return best
    
    # Phase 1: Multiple greedy builds with local search
    best_seq = build_greedy(42)
    best_len = len(best_seq)
    
    # Try multiple random seeds with varied strategies
    seeds_tried = 0
    while time.time() < deadline - 15:
        seed_val = rng.randint(0, 2**31 - 1)
        seeds_tried += 1
        seq = build_greedy(seed_val)
        
        # Local search on this sequence
        seq = local_search(seq, time_limit=1.5)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
    
    # Phase 2: Enhanced local search on best sequence
    if time.time() < deadline - 8:
        best_seq = local_search(best_seq, time_limit=4.0)
        best_len = len(best_seq)
    
    # Phase 3: CSP backtracking from the best sequence
    if time.time() < deadline - 5:
        used = set(best_seq)
        available = [x for x in composites if x not in used]
        rng.shuffle(available)
        
        # Limit available to keep search tractable
        available = available[:20]
        
        result = backtrack_csp(best_seq, available, 0, best_len + 3, deadline - 3)
        if len(result) > best_len:
            best_len = len(result)
            best_seq = result[:]
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 1:
        best_seq = local_search(best_seq, time_limit=2.0)
        best_len = len(best_seq)
    
    return best_seq
