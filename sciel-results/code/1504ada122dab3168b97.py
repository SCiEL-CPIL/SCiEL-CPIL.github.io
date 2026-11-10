import random
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    MAX_VAL = 10000
    N = 12
    MAX_SUM = 2 * MAX_VAL
    
    # Precompute smallest prime factor (SPF) for all numbers up to MAX_SUM
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    # State: list of N distinct integers
    # Data structure for fast evaluation:
    # prime_counts: dict mapping prime -> count of pairs having this prime factor
    # current_score: number of distinct primes with count > 0
    
    def calculate_initial_state(A):
        counts = defaultdict(int)
        for i in range(N):
            for j in range(i+1, N):
                s = A[i] + A[j]
                pf = get_prime_factors(s)
                for p in pf:
                    counts[p] += 1
        score = sum(1 for c in counts.values() if c > 0)
        return counts, score

    def try_move(A, counts, score, i, new_val):
        # i is the index to replace
        # new_val is the new value for A[i]
        # Returns (new_counts, new_score) if valid, else None
        # We must ensure new_val is distinct from other elements
        # We assume A[i] is removed, so we check against A[0..i-1, i+1..N-1]
        for j in range(N):
            if j != i and A[j] == new_val:
                return None
        
        # Update counts for pairs involving i
        # Old pairs: (A[i], A[j]) for j != i
        # New pairs: (new_val, A[j]) for j != i
        
        # We need to decrement counts for old primes
        # and increment for new primes
        # To avoid copying the dict, we can do it in place and backtrack if rejected,
        # or just compute the delta.
        
        old_primes = set()
        new_primes = set()
        
        for j in range(N):
            if j == i:
                continue
            old_s = A[i] + A[j]
            old_pf = get_prime_factors(old_s)
            new_s = new_val + A[j]
            new_pf = get_prime_factors(new_s)
            
            for p in old_pf:
                old_primes.add(p)
            for p in new_pf:
                new_primes.add(p)
        
        # We need to be careful: a prime p might appear in multiple pairs involving i.
        # So we should count how many times it appears in the "old" set of pairs
        # and how many times in the "new" set.
        
        old_counts_delta = defaultdict(int)
        new_counts_delta = defaultdict(int)
        
        for j in range(N):
            if j == i:
                continue
            old_s = A[i] + A[j]
            for p in get_prime_factors(old_s):
                old_counts_delta[p] += 1
            new_s = new_val + A[j]
            for p in get_prime_factors(new_s):
                new_counts_delta[p] += 1
        
        # Apply deltas to a temporary copy of counts?
        # Or just calculate the change in score.
        # Score = number of primes with count > 0.
        
        # Let's simulate the change
        # For each prime p involved in the change:
        # old_count = counts[p]
        # new_count = old_count - old_counts_delta.get(p, 0) + new_counts_delta.get(p, 0)
        
        # We need to know which primes are affected.
        affected_primes = set(old_counts_delta.keys()) | set(new_counts_delta.keys())
        
        score_change = 0
        for p in affected_primes:
            old_c = counts.get(p, 0)
            new_c = old_c - old_counts_delta.get(p, 0) + new_counts_delta.get(p, 0)
            
            if old_c > 0 and new_c == 0:
                score_change -= 1
            elif old_c == 0 and new_c > 0:
                score_change += 1
            # if both > 0 or both == 0, no change in score
            
        new_score = score + score_change
        return new_score, old_counts_delta, new_counts_delta

    best_A = None
    best_score = float('inf')
    best_counts = None
    
    # Seeds
    seeds = []
    # 1. Current best
    seeds.append([6 * i for i in range(1, 13)])
    # 2. Multiples of 30
    seeds.append([30 * i for i in range(1, 13)])
    # 3. Powers of 2 (scaled?)
    p2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    seeds.append(p2)
    # 4. Random
    for _ in range(5):
        seeds.append(random.sample(range(1, MAX_VAL + 1), N))
    
    # Add some "structured" random seeds
    # e.g. multiples of random small numbers
    for m in [12, 24, 18, 36]:
        seeds.append([m * i for i in range(1, 13)])
        
    for seed in seeds:
        if time.time() > deadline:
            break
            
        # Ensure validity
        if len(set(seed)) != N or any(x < 1 or x > MAX_VAL for x in seed):
            continue
            
        counts, score = calculate_initial_state(seed)
        
        if score < best_score:
            best_score = score
            best_A = seed[:]
            best_counts = counts.copy()
            
        # Local Search
        A = seed[:]
        current_counts = counts.copy()
        current_score = score
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.995
        
        while time.time() < deadline - 2:
            if T < T_min:
                break
                
            # Pick a random index
            i = random.randint(0, N - 1)
            
            # Pick a new value
            # Bias towards values that might be good?
            # Just random for now, or local random?
            # Let's try random in [1, MAX_VAL]
            new_val = random.randint(1, MAX_VAL)
            
            res = try_move(A, current_counts, current_score, i, new_val)
            if res is None:
                continue
                
            new_score, old_delta, new_delta = res
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                # Update A
                A[i] = new_val
                # Update counts
                for p, c in old_delta.items():
                    current_counts[p] -= c
                    if current_counts[p] == 0:
                        del current_counts[p]
                for p, c in new_delta.items():
                    current_counts[p] = current_counts.get(p, 0) + c
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_A = A[:]
                    best_counts = current_counts.copy()
            
            T *= cooling_rate
            
    # Final check
    if best_A is None:
        return [6 * i for i in range(1, 13)]
        
    return best_A
