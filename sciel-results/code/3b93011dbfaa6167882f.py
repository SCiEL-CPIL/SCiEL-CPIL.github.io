import time
import random
import math

def compute_metrics(A):
    """
    Computes the maximum of the size of the sumset A+A and product set A*A.
    A is a list of distinct positive integers.
    """
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(n):
            aj = A[j]
            sum_set.add(ai + aj)
            prod_set.add(ai * aj)
    return max(len(sum_set), len(prod_set))

def is_smooth(n, primes):
    """Check if n is composed only of the given primes."""
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def generate_smooth_candidates(max_val=10000, primes=[2, 3, 5, 7, 11]):
    """Generate a list of smooth numbers up to max_val."""
    smooths = set()
    def gen(idx, current_val):
        if idx == len(primes):
            if current_val <= max_val and current_val > 0:
                smooths.add(current_val)
            return
        p = primes[idx]
        # Try multiplying by p^k
        val = current_val
        while val <= max_val:
            gen(idx + 1, val)
            val *= p
    gen(0, 1)
    return list(smooths)

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Pre-generate a pool of smooth numbers
    # Using primes 2, 3, 5, 7 is usually good for small product sets
    smooth_pool = generate_smooth_candidates(10000, [2, 3, 5, 7])
    smooth_pool.sort()
    
    # Initial seeds
    seeds = []
    
    # Seed 1: Arithmetic Progression (minimizes sumset)
    seeds.append(list(range(1, 21)))
    
    # Seed 2: Smooth numbers (minimizes product set)
    # Pick 20 distinct smooth numbers
    if len(smooth_pool) >= 20:
        # Pick a mix of small and larger smooth numbers to balance sumset
        # Just taking the first 20 might make sumset small, but product set might be huge if they are all powers of 2
        # Let's pick a random subset of smooth numbers
        random.seed(42) # For reproducibility in seed generation
        smooth_seed = random.sample(smooth_pool, 20)
        smooth_seed.sort()
        seeds.append(smooth_seed)
        
    # Seed 3: Mixed - some AP, some smooth
    mixed = list(range(1, 11)) # 1-10
    # Add 10 more smooth numbers not in 1-10
    candidates = [x for x in smooth_pool if x > 10]
    if len(candidates) >= 10:
        mixed += random.sample(candidates, 10)
        mixed.sort()
        seeds.append(mixed)

    best_A = None
    best_score = float('inf')
    
    # Evaluate seeds
    for seed in seeds:
        if len(set(seed)) != 20 or any(x < 1 or x > 10000 for x in seed):
            continue
        score = compute_metrics(seed)
        if score < best_score:
            best_score = score
            best_A = seed[:]
            
    # Simulated Annealing
    A = best_A[:]
    current_score = best_score
    
    # Parameters
    T = 100.0
    T_min = 0.1
    alpha = 0.999 # Cooling rate
    
    A_set = set(A)
    
    while time.time() < deadline and T > T_min:
        # Choose a random index to mutate
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Generate a candidate
        # Strategy: 
        # 1. 50% chance: Pick from smooth pool (structure-aware)
        # 2. 30% chance: Neighbor of old_val
        # 3. 20% chance: Random 1-10000
        
        r = random.random()
        if r < 0.5:
            # Pick a smooth number
            # Bias towards numbers similar in magnitude to old_val?
            # Just random from pool is fine for now
            cand = random.choice(smooth_pool)
        elif r < 0.8:
            # Neighbor
            delta = random.choice([-3, -2, -1, 1, 2, 3])
            cand = old_val + delta
            if cand < 1: cand = 1
            if cand > 10000: cand = 10000
        else:
            # Random
            cand = random.randint(1, 10000)
            
        if cand == old_val or cand in A_set:
            continue
            
        # Apply move
        A[idx] = cand
        A_set.discard(old_val)
        A_set.add(cand)
        
        new_score = compute_metrics(A)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = A[:]
        else:
            # Revert
            A[idx] = old_val
            A_set.discard(cand)
            A_set.add(old_val)
            
        T *= alpha
        
    return best_A
