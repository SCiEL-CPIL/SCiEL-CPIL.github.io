import numpy as np
import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def greedy_sieve(seed):
    points = np.arange(N + 1)
    k = len(points)
    rng = np.random.default_rng(seed)
    
    for p in PRIMES:
        if p > k:
            break
        
        residues = points % p
        counts = np.bincount(residues, minlength=p)
        
        if counts.min() > 0:
            min_count = counts.min()
            candidates = np.where(counts == min_count)[0]
            chosen_res = rng.choice(candidates)
            
            # Keep points where residue is NOT chosen_res
            mask = residues != chosen_res
            points = points[mask]
            k = len(points)
            
    return points.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 300  # Safety margin before 330s kill
    
    best_points = []
    best_size = 0
    
    trial = 0
    while time.time() < deadline:
        seed = trial
        points = greedy_sieve(seed)
        size = len(points)
        if size > best_size:
            best_size = size
            best_points = points
        trial += 1
        
    return best_points
