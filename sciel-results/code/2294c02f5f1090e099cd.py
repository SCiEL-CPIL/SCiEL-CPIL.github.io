import random
import time

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes up to N
ALL_PRIMES = _primes(N + 1)

def _construct_greedy(randomize=False, seed=None):
    if seed is not None:
        random.seed(seed)
    S = list(range(N + 1))
    changed = True
    while changed:
        changed = False
        k = len(S)
        # Filter primes <= k
        primes_to_check = [p for p in ALL_PRIMES if p <= k]
        for p in primes_to_check:
            classes = [0] * p
            for v in S:
                classes[v % p] += 1
            if min(classes) > 0:
                min_count = min(classes)
                if randomize:
                    candidates = [i for i, c in enumerate(classes) if c == min_count]
                    if len(candidates) > 1:
                        target_residue = random.choice(candidates)
                    else:
                        target_residue = candidates[0]
                else:
                    target_residue = classes.index(min_count)
                S = [v for v in S if v % p != target_residue]
                changed = True
                break
    return S

def construct():
    best_S = _construct_greedy(randomize=False)
    best_size = len(best_S)
    
    start_time = time.time()
    max_time = 20.0 # 20 seconds for randomization
    
    # Run multiple random instances
    for i in range(50):
        if time.time() - start_time > max_time:
            break
        S = _construct_greedy(randomize=True, seed=i)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            
    return best_S
