import time

N = 511

def get_prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

# Precomputation
factors = [None] * (N + 1)
for i in range(1, N + 1):
    factors[i] = get_prime_factors(i)

values = list(range(N, 0, -1))
weights = [1.0 / (512 - v) for v in values]

suffix_sums = [0.0] * (N + 1)
for i in range(N - 1, -1, -1):
    suffix_sums[i] = suffix_sums[i + 1] + weights[i]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_set = []
    
    # Greedy initialization
    greedy_set = []
    greedy_score = 0.0
    used_primes_greedy = set()
    for v in values:
        if factors[v].isdisjoint(used_primes_greedy):
            greedy_set.append(v)
            greedy_score += 1.0 / (512 - v)
            used_primes_greedy |= factors[v]
    best_score = greedy_score
    best_set = greedy_set[:]
    
    def search(k, current_score, chosen, used_primes):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
            
        # Loose bound check first (O(1))
        if current_score + suffix_sums[k] <= best_score:
            return
            
        if k == N:
            if current_score > best_score:
                best_score = current_score
                best_set = chosen[:]
            return
        
        # Tighter bound check (O(N))
        # Only compute if loose bound is significantly better than best_score?
        # Or just always compute to prune early.
        tight_bound = current_score
        for j in range(k, N):
            if factors[values[j]].isdisjoint(used_primes):
                tight_bound += weights[j]
                # Early exit if tight_bound is already <= best_score?
                # No, we are summing up, so it increases.
                # We want to know if the MAX possible is <= best_score.
                # So we must sum all valid ones.
        
        if tight_bound <= best_score:
            return
            
        v = values[k]
        w = weights[k]
        
        # Option 1: Include v
        if factors[v].isdisjoint(used_primes):
            search(k + 1, current_score + w, chosen + [v], used_primes | factors[v])
            
        # Option 2: Exclude v
        search(k + 1, current_score, chosen, used_primes)

    search(0, 0.0, [], set())
    
    return best_set
