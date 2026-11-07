import time

# Pre-compute prime exponent vectors for 2..40
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
prime_to_idx = {p: i for i, p in enumerate(primes)}
num_primes = len(primes)

def get_vector(n):
    v = [0] * num_primes
    temp = n
    for i, p in enumerate(primes):
        while temp % p == 0:
            v[i] += 1
            temp //= p
    return tuple(v)

vectors = {n: get_vector(n) for n in range(2, 41)}

# Order: Primes first, then composites
all_nums = list(range(2, 41))
primes_in_range = [p for p in all_nums if p in prime_to_idx]
composites_in_range = [c for c in all_nums if c not in prime_to_idx]
candidates = primes_in_range + composites_in_range

best_set = []
start_time = time.time()

def search(idx, current_set, current_sums):
    global best_set
    if time.time() - start_time > 85:
        return
    
    # Pruning
    if len(current_set) + (len(candidates) - idx) <= len(best_set):
        return
    
    if idx == len(candidates):
        if len(current_set) > len(best_set):
            best_set = current_set[:]
        return
    
    # Option 1: Include candidates[idx] (try to build larger set first)
    c = candidates[idx]
    v = vectors[c]
    
    new_sums = set()
    collision = False
    for s in current_sums:
        ns = tuple(a + b for a, b in zip(s, v))
        if ns in current_sums:
            collision = True
            break
        new_sums.add(ns)
        
    if not collision:
        search(idx + 1, current_set + [c], current_sums | new_sums)
        
    # Option 2: Skip candidates[idx]
    search(idx + 1, current_set, current_sums)

zero_vec = tuple(0 for _ in range(num_primes))
search(0, [], {zero_vec})

def construct():
    return [int(x) for x in best_set]
