"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def _primes_up_to(n):
    """Generate all primes up to n using sieve of Eratosthenes."""
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.array([i for i, is_p in enumerate(sieve) if is_p], dtype=np.int64)

def _compute_L(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
    n = len(primes)
    # For each prime p_i with residue r_i, the covered integers in [1, max_check] are:
    # r_i, r_i + p_i, r_i + 2*p_i, ... (if r_i == 0, it's p_i, 2*p_i, ...)
    # We count how many congruences each integer satisfies.
    
    # Use an efficient approach: for each prime, mark its multiples
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    for i in range(n):
        p = int(primes[i])
        r = int(residues[i]) % p
        if r == 0:
            r = p  # x = 0 mod p means x = p, 2p, ...
        # Start from the first positive integer >= 1 that is congruent to r mod p
        start = r
        if start < 1:
            start += p * ((1 - start + p - 1) // p)
        # Mark all numbers start, start+p, start+2p, ... up to max_check
        for x in range(start, max_check + 1, p):
            counts[x] += 1
    
    # Find the largest L such that counts[1..L] >= 2
    # We need the first position where counts < 2
    if counts[0] < 0:  # shouldn't happen
        return 0
    
    # Check from 1 onwards
    for x in range(1, max_check + 1):
        if counts[x] < 2:
            return x - 1
    return max_check

def _compute_L_fast(residues, primes, max_check=100000):
    """Faster computation using vectorized operations."""
    n = len(primes)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    for i in range(n):
        p = int(primes[i])
        r = int(residues[i]) % p
        if r == 0:
            r = p
        # Use numpy to mark positions
        indices = np.arange(r, max_check + 1, p, dtype=np.int64)
        indices = indices[indices >= 1]
        if len(indices) > 0:
            counts[indices] += 1
    
    # Find first position >= 1 where count < 2
    # We need all of 1..L to have count >= 2
    valid = counts[1:max_check+1] >= 2
    if not np.any(valid):
        return 0
    # Find the first False in the valid array
    false_indices = np.where(~valid)[0]
    if len(false_indices) == 0:
        return max_check
    first_false = false_indices[0]
    return int(first_false)

def _surrogate_objective(x, primes, window_size):
    """
    Differentiable surrogate: penalize the maximum deficit in coverage.
    x is a real-valued vector of length len(primes).
    We project to integer residues modulo p at evaluation.
    
    For a smoother surrogate, we can use a soft min over the coverage counts.
    """
    n = len(primes)
    residues = np.mod(np.round(x).astype(np.int64), primes).astype(np.int64)
    
    # Compute coverage counts for positions 1..window_size
    counts = np.zeros(window_size + 1, dtype=np.float64)
    for i in range(n):
        p = int(primes[i])
        r = int(residues[i])
        if r == 0:
            r = p
        indices = np.arange(r, window_size + 1, p, dtype=np.int64)
        indices = indices[indices >= 1]
        if len(indices) > 0:
            counts[indices] += 1.0
    
    # The objective is to maximize the minimum coverage in [1, window_size]
    # We want min(counts[1:window_size+1]) to be >= 2
    # Surrogate: -min(counts[1:window_size+1]) + penalty for positions with count < 2
    min_count = np.min(counts[1:window_size+1])
    
    # Smooth approximation of the deficit: sum of max(0, 2 - count_i)
    deficit = np.sum(np.maximum(0.0, 2.0 - counts[1:window_size+1]))
    
    # Combine: we want to minimize deficit, and also reward high min_count
    return deficit - 0.1 * min_count

def _greedy_improve(residues, primes, window_size, max_iter=5000, time_limit=10.0):
    """
    Greedy local search: try changing one residue at a time to improve coverage.
    """
    n = len(primes)
    residues = residues.copy().astype(np.int64)
    best_L = _compute_L_fast(residues, primes, min(window_size, 100000))
    
    start_time = time.time()
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try each prime in random order
        order = np.random.permutation(n)
        for i in order:
            p = int(primes[i])
            old_r = residues[i]
            
            # Try a few candidate residues
            candidates = [
                (old_r + 1) % p,
                (old_r - 1) % p,
                (old_r + p // 2) % p,
                np.random.randint(0, p),
            ]
            
            best_new_r = old_r
            best_new_L = best_L
            
            for new_r in candidates:
                residues[i] = new_r
                new_L = _compute_L_fast(residues, primes, min(window_size, 100000))
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_new_r = new_r
                residues[i] = old_r
            
            if best_new_L > best_L:
                residues[i] = best_new_r
                best_L = best_new_L
                improved = True
        
        if not improved:
            # Try a few random restarts from current solution
            for _ in range(5):
                trial = residues.copy()
                # Perturb a random subset of residues
                k = np.random.randint(1, min(10, n))
                idx = np.random.choice(n, k, replace=False)
                for j in idx:
                    trial[j] = np.random.randint(0, int(primes[j]))
                trial_L = _compute_L_fast(trial, primes, min(window_size, 100000))
                if trial_L > best_L:
                    residues = trial
                    best_L = trial_L
                    improved = True
                    break
            
            if not improved:
                break
    
    return residues, best_L

def _simulated_annealing(residues, primes, window_size, time_limit=30.0):
    """
    Simulated annealing to escape local minima.
    """
    n = len(primes)
    residues = residues.copy().astype(np.int64)
    current_L = _compute_L_fast(residues, primes, min(window_size, 100000))
    best_residues = residues.copy()
    best_L = current_L
    
    start_time = time.time()
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random prime to change
        i = np.random.randint(n)
        p = int(primes[i])
        old_r = residues[i]
        new_r = np.random.randint(0, p)
        residues[i] = new_r
        
        new_L = _compute_L_fast(residues, primes, min(window_size, 100000))
        delta = new_L - current_L
        
        if delta >= 0 or np.random.random() < np.exp(delta / max(T, 1e-6)):
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            residues[i] = old_r
        
        T *= cooling
    
    return best_residues, best_L

def construct():
    """
    Construct 168 residues, one for each prime p <= 1000 in increasing order,
    with 0 <= r_p < p, maximizing the doubly-covered prefix length.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    primes = _primes_up_to(1000)
    n = len(primes)
    
    # Initial solution: use the seed strategy as starting point
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    current_L = _compute_L_fast(residues, primes, 100000)
    best_residues = residues.copy()
    best_L = current_L
    
    # Phase 1: Greedy improvement
    if time.time() - start_time < deadline - 40:
        residues, current_L = _greedy_improve(residues, primes, 10000, max_iter=3000, time_limit=20.0)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
    
    # Phase 2: Simulated annealing to escape local minima
    if time.time() - start_time < deadline - 10:
        sa_time = min(30.0, deadline - (time.time() - start_time) - 5)
        residues, current_L = _simulated_annealing(residues, primes, 10000, time_limit=sa_time)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
    
    # Phase 3: Final greedy polish
    if time.time() - start_time < deadline - 5:
        polish_time = min(15.0, deadline - (time.time() - start_time) - 3)
        residues, current_L = _greedy_improve(residues, primes, 10000, max_iter=2000, time_limit=polish_time)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
    
    # Verify the final result
    final_L = _compute_L_fast(best_residues, primes, 100000)
    
    # Return as a list of ints
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
