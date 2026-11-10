import time
import random
import math
import numpy as np

N = 5000
MAX_TIME = 290.0  # Safety margin before 330s kill

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

PRIMES = _primes(N)
# Only primes up to N are relevant for the constraints of a set in [0, N]
# Actually, if the set size is K, we only need primes <= K.
# But K is unknown until we construct the set.
# However, if we enforce constraints for ALL primes <= N, the resulting set is admissible for ANY size <= N.
# So we can just optimize for all primes <= N.

def _calculate_size(residues, primes, x_vals):
    """
    residues: dict p -> r
    primes: list of primes
    x_vals: np array of x in [0, N]
    Returns size of set S = {x | x % p != residues[p] for all p}
    """
    # Start with all True
    valid = np.ones(len(x_vals), dtype=bool)
    for p in primes:
        r = residues[p]
        # x % p != r
        valid &= (x_vals % p != r)
    return np.sum(valid)

def _get_set(residues, primes, x_vals):
    valid = np.ones(len(x_vals), dtype=bool)
    for p in primes:
        r = residues[p]
        valid &= (x_vals % p != r)
    return x_vals[valid].tolist()

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    x_vals = np.arange(N + 1)
    
    # Initialize residues randomly
    residues = {}
    for p in PRIMES:
        residues[p] = rng.randint(0, p - 1)
    
    current_size = _calculate_size(residues, PRIMES, x_vals)
    best_residues = residues.copy()
    best_size = current_size
    best_set = _get_set(residues, PRIMES, x_vals)
    
    # Simulated Annealing on residues
    T0 = 100.0
    Tend = 1.0
    # Estimate steps: we want to run for MAX_TIME
    # Each step involves O(N * P) ops. 
    # With NumPy, N=5000, P=669. 
    # 5000 * 669 = 3.3M ops. In numpy, this is very fast, maybe 1-5ms.
    # So we can do ~200-1000 steps per second.
    # Let's aim for 10000 steps total or until time runs out.
    
    steps = 0
    total_planned = 50000 # Will be cut by time
    
    while time.time() - start_time < MAX_TIME:
        steps += 1
        frac = min(1.0, steps / total_planned)
        T = T0 * (Tend / T0) ** frac
        
        # Pick a random prime to change
        p = PRIMES[rng.randint(0, len(PRIMES) - 1)]
        old_r = residues[p]
        new_r = rng.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        residues[p] = new_r
        new_size = _calculate_size(residues, PRIMES, x_vals)
        delta = new_size - current_size
        
        if delta > 0 or (T > 1e-10 and rng.random() < math.exp(delta / T)):
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_residues = residues.copy()
                # Don't compute set yet, just track residues
        else:
            residues[p] = old_r
            
        # Check time periodically
        if steps % 100 == 0:
            if time.time() - start_time > MAX_TIME - 5:
                break

    # Finalize: get the set for the best residues
    # Use the best_residues found
    final_set = _get_set(best_residues, PRIMES, x_vals)
    
    # Verify admissibility just in case
    # The construction guarantees admissibility for primes <= N, so it's admissible for any size <= N.
    # But let's double check if the size is correct.
    
    return final_set

# Precompute primes at module level for speed
PRIMES = _primes(N)
