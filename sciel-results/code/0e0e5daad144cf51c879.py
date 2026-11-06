"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

# Precompute prime factors for all possible sums (1 to 20000)
_MAX_SUM = 20000
_prime_factors_cache = [None] * (_MAX_SUM + 1)

def _init_factors():
    """Precompute prime factors for all numbers up to _MAX_SUM."""
    # Sieve of Eratosthenes variant to find smallest prime factor, then derive distinct factors
    spf = list(range(_MAX_SUM + 1))
    for i in range(2, int(_MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, _MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    for n in range(2, _MAX_SUM + 1):
        factors = set()
        x = n
        while x > 1:
            p = spf[x]
            factors.add(p)
            while x % p == 0:
                x //= p
        _prime_factors_cache[n] = factors
    
    # Handle 1 (no prime factors)
    _prime_factors_cache[1] = set()

_init_factors()

def _get_factors(n):
    return _prime_factors_cache[n]

class PrimeCounter:
    def __init__(self, arr):
        self.arr = arr[:]
        self.prime_counts = defaultdict(int)
        self._rebuild()
        
    def _rebuild(self):
        self.prime_counts.clear()
        n = len(self.arr)
        for i in range(n):
            for j in range(i+1, n):
                s = self.arr[i] + self.arr[j]
                for p in _get_factors(s):
                    self.prime_counts[p] += 1
    
    def get_score(self):
        return len(self.prime_counts)
        
    def try_swap(self, idx, new_val):
        """
        Attempts to swap element at idx with new_val.
        Returns (success: bool, old_val: int).
        If success is True, the state is updated.
        If success is False, the state is unchanged.
        """
        old_val = self.arr[idx]
        if old_val == new_val:
            return False, old_val
        
        # Check distinctness: new_val must not be in arr (except at idx)
        if new_val in self.arr:
            return False, old_val
        
        # Remove contributions of old_val
        for i in range(len(self.arr)):
            if i == idx: continue
            s = old_val + self.arr[i]
            for p in _get_factors(s):
                self.prime_counts[p] -= 1
                if self.prime_counts[p] == 0:
                    del self.prime_counts[p]
        
        # Update arr
        self.arr[idx] = new_val
        
        # Add contributions of new_val
        for i in range(len(self.arr)):
            if i == idx: continue
            s = new_val + self.arr[i]
            for p in _get_factors(s):
                self.prime_counts[p] += 1
                
        return True, old_val

    def revert_swap(self, idx, old_val):
        """Reverts a successful swap at idx back to old_val."""
        new_val = self.arr[idx]
        
        # Remove contributions of new_val
        for i in range(len(self.arr)):
            if i == idx: continue
            s = new_val + self.arr[i]
            for p in _get_factors(s):
                self.prime_counts[p] -= 1
                if self.prime_counts[p] == 0:
                    del self.prime_counts[p]
        
        # Update arr
        self.arr[idx] = old_val
        
        # Add contributions of old_val
        for i in range(len(self.arr)):
            if i == idx: continue
            s = old_val + self.arr[i]
            for p in _get_factors(s):
                self.prime_counts[p] += 1

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 12
    
    # Generate a pool of "good" candidate numbers
    # Numbers with small prime factors are better for sums?
    # Actually, if all numbers are multiples of K, sums are multiples of K.
    # Let's focus on numbers that are multiples of small primes like 2, 3, 5, 6, 10, 12...
    # But we want to minimize the UNION of primes in sums.
    # If we pick numbers all divisible by 2, sums are even (prime 2).
    # If we pick numbers all divisible by 6, sums are divisible by 6 (primes 2, 3).
    # The "cost" of a sum is its prime factors.
    
    # Strategy:
    # 1. Start with a structured solution: e.g., multiples of 6.
    # 2. Local search: try replacing one element with another from a pool of "compatible" numbers.
    
    # Pool of candidates: Multiples of 2, 3, 6 up to 10000?
    # To make search faster, we can restrict the pool to numbers that share factors with the current best.
    # But initially, let's use a broad pool of "smooth" numbers or multiples of small primes.
    
    # Let's generate a pool of numbers that are multiples of 2, 3, 5, 6, 10, 15, 30.
    # This ensures sums are likely to have small prime factors.
    pool = set()
    for m in [2, 3, 5, 6, 10, 15, 30]:
        for k in range(1, 10000 // m + 1):
            pool.add(m * k)
    pool = list(pool)
    
    # Initial Candidates
    initial_cands = []
    # 1. Multiples of 6
    initial_cands.append([6 * i for i in range(1, 13)])
    # 2. Multiples of 12
    initial_cands.append([12 * i for i in range(1, 13)])
    # 3. Random from pool
    initial_cands.append(random.sample(pool, 12))
    # 4. Powers of 2? 2, 4, 8, 16... sums are even.
    initial_cands.append([2**i for i in range(1, 13)]) # 2..4096
    
    best_score = float('inf')
    best_counter = None
    
    for cand in initial_cands:
        if len(set(cand)) != n: continue
        if max(cand) > 10000: continue
        counter = PrimeCounter(cand)
        score = counter.get_score()
        if score < best_score:
            best_score = score
            best_counter = counter
    
    if best_counter is None:
        # Fallback
        best_arr = random.sample(range(1, 10001), 12)
        best_counter = PrimeCounter(best_arr)
        best_score = best_counter.get_score()
    
    current_counter = best_counter
    current_score = best_score
    
    # Local Search
    # We will try to replace elements with other elements from the pool.
    # To be efficient, we can pre-group pool elements by their value modulo small numbers?
    # Or just random sampling from pool.
    
    while time.time() < deadline:
        improved = False
        # Try a batch of random moves
        for _ in range(500):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, n - 1)
            # Pick a random candidate from pool
            new_val = random.choice(pool)
            
            # Attempt swap
            success, old_val = current_counter.try_swap(idx, new_val)
            if not success:
                continue
                
            new_score = current_counter.get_score()
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_counter = PrimeCounter(current_counter.arr)
            else:
                # Revert
                current_counter.revert_swap(idx, old_val)
        
        if not improved:
            # Random restart with a different strategy?
            # Try a new initial candidate from a different pattern
            if random.random() < 0.05:
                # Try multiples of a random small number
                m = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
                cand = [m * i for i in range(1, 13)]
                if max(cand) <= 10000 and len(set(cand)) == n:
                    counter = PrimeCounter(cand)
                    score = counter.get_score()
                    if score < current_score:
                        current_counter = counter
                        current_score = score
                        if score < best_score:
                            best_score = score
                            best_counter = PrimeCounter(cand)

    return best_counter.arr
# EVOLVE-BLOCK-END
