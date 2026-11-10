import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s
    
    n = 12
    max_val = 10000
    
    # Precompute smallest prime factor (SPF) table up to 2*max_val = 20000
    max_sum = 2 * max_val
    spf = list(range(max_sum + 1))
    for i in range(2, int(max_sum**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_sum + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def prime_factors(x):
        """Return set of prime factors of x using SPF table."""
        factors = set()
        while x > 1:
            p = spf[x]
            factors.add(p)
            while x % p == 0:
                x //= p
        return factors
    
    def score_set(arr):
        """Count distinct prime factors across all pairwise sums."""
        all_primes = set()
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                # Factor s using SPF
                x = s
                while x > 1:
                    p = spf[x]
                    all_primes.add(p)
                    while x % p == 0:
                        x //= p
        return len(all_primes)
    
    # Initializations
    initializations = []
    
    # Multiples of 6
    init1 = [6 * i for i in range(1, n+1)]
    initializations.append(init1)
    
    # Multiples of 30
    init2 = [30 * i for i in range(1, n+1)]
    initializations.append(init2)
    
    # Even numbers
    init3 = [2 * i for i in range(1, n+1)]
    initializations.append(init3)
    
    # Multiples of 12
    init4 = [12 * i for i in range(1, n+1)]
    initializations.append(init4)
    
    # Random even numbers
    random.seed(42)
    for _ in range(20):
        candidates = random.sample(range(2, max_val+1, 2), n)
        initializations.append(sorted(candidates))
    
    # Random multiples of 6
    for _ in range(20):
        candidates = random.sample(range(6, max_val+1, 6), n)
        initializations.append(sorted(candidates))
    
    best_score = float('inf')
    best_arr = None
    
    for init in initializations:
        if time.time() > deadline:
            break
        s = score_set(init)
        if s < best_score:
            best_score = s
            best_arr = list(init)
    
    if best_arr is None:
        return list(range(1, n+1))
    
    # Local search: swap-based
    arr = list(best_arr)
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Try random swaps
        improved = False
        for _ in range(500):
            if time.time() > deadline:
                break
            i, j = random.sample(range(n), 2)
            # Try swapping arr[i] with a random candidate
            new_val = random.randint(1, max_val)
            if new_val in arr and new_val != arr[i]:
                continue
            
            old_val = arr[i]
            arr[i] = new_val
            
            new_score = score_set(arr)
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                improved = True
            else:
                arr[i] = old_val
                no_improve_count += 1
        
        if not improved and no_improve_count > max_no_improve:
            break
    
    # Also try adding/replacing with values that minimize new primes
    # Final local search pass with more targeted moves
    for _ in range(3):
        if time.time() > deadline:
            break
        for i in range(n):
            if time.time() > deadline:
                break
            # Try replacing arr[i] with various values
            for trial in range(100):
                if time.time() > deadline:
                    break
                new_val = random.randint(1, max_val)
                if new_val in arr:
                    continue
                old_val = arr[i]
                arr[i] = new_val
                new_score = score_set(arr)
                if new_score < current_score:
                    current_score = new_score
                else:
                    arr[i] = old_val
    
    return sorted(arr)
