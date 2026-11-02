"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score(U):
    """Compute the score for a set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u <= 0:
        return 0.0
    # Compute U+U and U-U
    # For efficiency, use numpy
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return score


def _is_prime(n):
    """Check if n is prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _generate_primes(limit):
    """Generate primes up to limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]


def _prime_based_seed(n_primes, max_prime=10000):
    """Generate a set based on prime-step gaps.
    U = {0, p1, p1+p2, p1+p2+p3, ...}
    This exploits additive independence of primes.
    """
    primes = _generate_primes(max_prime)
    if len(primes) < n_primes:
        primes = primes + [2 * i + 1 for i in range(n_primes - len(primes))]
        # Filter to keep only primes
        primes = [p for p in primes if _is_prime(p)]
        if len(primes) < n_primes:
            # Pad with large numbers if needed
            last = primes[-1] if primes else 100
            primes += [last + 100 * (i + 1) for i in range(n_primes - len(primes))]
    
    selected = primes[:n_primes]
    U = [0]
    current = 0
    for p in selected:
        current += p
        U.append(current)
    return U


def _greedy_prime_optimization(n_primes, max_prime=50000, time_limit=30.0):
    """Greedy optimization of prime selection to maximize score."""
    start_time = time.time()
    primes = _generate_primes(max_prime)
    
    if len(primes) < n_primes:
        return _prime_based_seed(n_primes, max_prime)
    
    # Start with first n_primes primes
    selected = list(primes[:n_primes])
    U = [0]
    current = 0
    for p in selected:
        current += p
        U.append(current)
    best_score = _compute_score(U)
    best_selected = list(selected)
    
    # Try swapping primes to improve score
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
        
        # Try a random swap
        i = random.randint(0, n_primes - 1)
        j = random.randint(0, len(primes) - 1)
        if i == j:
            continue
        
        old_prime = selected[i]
        new_prime = primes[j]
        if new_prime == old_prime:
            continue
        
        selected[i] = new_prime
        U = [0]
        current = 0
        for p in selected:
            current += p
            U.append(current)
        score = _compute_score(U)
        
        if score > best_score:
            best_score = score
            best_selected = list(selected)
        else:
            selected[i] = old_prime
    
    U = [0]
    current = 0
    for p in best_selected:
        current += p
        U.append(current)
    return U


def _local_search(U, max_iterations=5000, time_limit=10.0):
    """Local search to improve the set U."""
    U = sorted(set(U))
    n = len(U)
    max_u = U[-1]
    best_score = _compute_score(U)
    best_U = list(U)
    start_time = time.time()
    
    # Try adding new elements
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try adding a new element
        # Choose a random position between 0 and max_u + some margin
        margin = max(10, max_u // 5)
        new_val = random.randint(0, max_u + margin)
        if new_val in U:
            continue
        
        new_U = sorted(U + [new_val])
        new_score = _compute_score(new_U)
        
        if new_score > best_score:
            best_score = new_score
            best_U = new_U
            U = new_U
            max_u = U[-1]
    
    return best_U, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 second limit for safety
    
    # Try prime-based constructions of different sizes
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Generate prime-based seeds of increasing size
    for n_primes in range(3, 25):
        if time.time() > deadline - 60:  # Leave 60 seconds for local search
            break
        
        try:
            U = _prime_based_seed(n_primes, max_prime=10000)
        except Exception:
            continue
        
        if not U or 0 not in U:
            continue
        
        U = sorted(U)
        score = _compute_score(U)
        
        if score > best_score:
            best_score = score
            best_U = list(U)
    
    # Try greedy prime optimization for a few sizes
    if time.time() < deadline - 40:
        for n_primes in [8, 12, 16, 20]:
            if time.time() > deadline - 30:
                break
            try:
                U = _greedy_prime_optimization(n_primes, max_prime=50000, time_limit=15)
            except Exception:
                continue
            
            if not U or 0 not in U:
                continue
            
            U = sorted(U)
            score = _compute_score(U)
            
            if score > best_score:
                best_score = score
                best_U = list(U)
    
    # Local search from the best seed
    if time.time() < deadline - 20:  # Leave 20 seconds margin
        U, score = _local_search(best_U, max_iterations=15000, time_limit=40)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Try a few more local search rounds with different random seeds
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            # Perturb the best solution
            U = list(best_U)
            # Remove a random element (not 0)
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                U.pop(idx)
            # Add a random element
            max_u = max(U) if U else 0
            new_val = random.randint(0, max_u + max(10, max_u // 3))
            if new_val not in U:
                U.append(new_val)
            
            U, score = _local_search(U, max_iterations=3000, time_limit=10)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.append(0)
    
    return sorted(best_U)

# EVOLVE-BLOCK-END
