"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import isqrt

# EVOLVE-BLOCK-START
def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    if n % 2 == 0:
        factors.add(2)
        while n % 2 == 0:
            n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            factors.add(f)
            while n % f == 0:
                n //= f
        f += 2
    if n > 1:
        factors.add(n)
    return factors

def _objective(arr):
    """Return set of distinct prime factors of all pairwise sums."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes |= _prime_factors(s)
    return primes

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    n = 12
    max_val = 10000
    rng = random.Random(42)
    
    # Initial candidate: try to find a good starting set
    # Strategy: use local search with random restarts
    
    best_set = None
    best_primes = None
    best_count = float('inf')
    
    # Try several initial configurations
    # Configuration 1: small consecutive even numbers
    # Configuration 2: random
    # Configuration 3: multiples of a small number
    
    def evaluate(arr):
        return _objective(arr)
    
    # Generate initial candidates
    candidates = []
    
    # Candidate: consecutive even numbers starting from 2
    candidates.append(list(range(2, 2 + 2*n, 2)))
    
    # Candidate: random
    for _ in range(5):
        candidates.append(rng.sample(range(1, max_val + 1), n))
    
    # Candidate: multiples of 2 in a structured way
    candidates.append([2 * (i + 1) for i in range(n)])
    
    # Candidate: all numbers ≡ 1 mod small prime
    for mod in [3, 5, 7]:
        vals = [x for x in range(1, max_val + 1) if x % mod == 1]
        if len(vals) >= n:
            candidates.append(vals[:n])
    
    # Evaluate all candidates
    for cand in candidates:
        primes = evaluate(cand)
        if len(primes) < best_count:
            best_count = len(primes)
            best_primes = primes
            best_set = list(cand)
    
    # Local search: swap two elements
    def local_search(arr, primes, count):
        nonlocal best_set, best_primes, best_count
        current = list(arr)
        current_primes = set(primes)
        current_count = count
        no_improve = 0
        
        while time.time() < deadline:
            improved = False
            # Try all swaps
            for i in range(n):
                for j in range(i + 1, n):
                    old_i = current[i]
                    old_j = current[j]
                    # Remove old contributions
                    # This is expensive; let's just recompute for simplicity in early phases
                    # For speed, do targeted removal
                    new_current = list(current)
                    new_current[i] = rng.randint(1, max_val)
                    # Ensure distinctness
                    if len(set(new_current)) < n:
                        continue
                    
                    new_primes = set()
                    for a in range(n):
                        for b in range(a + 1, n):
                            new_primes |= _prime_factors(new_current[a] + new_current[b])
                    
                    if len(new_primes) < current_count:
                        current = new_current
                        current_primes = new_primes
                        current_count = len(new_primes)
                        improved = True
                        no_improve = 0
                        if current_count < best_count:
                            best_count = current_count
                            best_primes = current_primes
                            best_set = list(current)
                        break
                    elif len(new_primes) == current_count and rng.random() < 0.1:
                        # Accept equal to escape local minima
                        current = new_current
                        current_primes = new_primes
                        no_improve += 1
                        break
                if improved or no_improve > 50:
                    break
            if not improved:
                no_improve += 1
            if no_improve > 200:
                break
        
        return current, current_primes, current_count
    
    # Run local search from best candidate
    current, current_primes, current_count = local_search(
        best_set, best_primes, best_count
    )
    
    # If time remains, try more restarts
    if time.time() < deadline - 5:
        for restart in range(20):
            if time.time() >= deadline - 5:
                break
            # Random restart
            cand = rng.sample(range(1, max_val + 1), n)
            primes = evaluate(cand)
            if len(primes) < best_count:
                best_count = len(primes)
                best_primes = primes
                best_set = list(cand)
            else:
                # Local search from this candidate
                cand2, p2, c2 = local_search(cand, primes, len(primes))
                if c2 < best_count:
                    best_count = c2
                    best_primes = p2
                    best_set = cand2
    
    return best_set
# EVOLVE-BLOCK-END
