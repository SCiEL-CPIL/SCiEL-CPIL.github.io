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
    
    # Use set operations for better performance with moderate n
    sums = set()
    diffs = set()
    for i in range(n):
        ui = U[i]
        for j in range(n):
            uj = U[j]
            sums.add(ui + uj)
            diffs.add(ui - uj)
    
    n_sums = len(sums)
    n_diffs = len(diffs)
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return score


def _golomb_ruler(n):
    """Generate a Golomb ruler of length n."""
    if n <= 1:
        return [0]
    if n == 2:
        return [0, 1]
    if n == 3:
        return [0, 1, 3]
    if n == 4:
        return [0, 1, 4, 6]
    if n == 5:
        return [0, 1, 4, 9, 11]
    if n == 6:
        return [0, 1, 4, 10, 12, 17]
    if n == 7:
        return [0, 1, 4, 10, 18, 23, 29]
    if n == 8:
        return [0, 1, 4, 10, 18, 23, 29, 36]
    if n == 9:
        return [0, 1, 4, 10, 18, 23, 29, 36, 44]
    if n == 10:
        return [0, 1, 4, 10, 18, 23, 29, 36, 44, 52]
    if n == 11:
        return [0, 1, 4, 10, 18, 23, 29, 36, 44, 52, 60]
    if n == 12:
        return [0, 1, 4, 10, 18, 23, 29, 36, 44, 52, 60, 68]
    # Fallback: use a simple construction
    U = [0]
    current = 0
    for i in range(1, n):
        current += i * 2 + 1
        U.append(current)
    return U


def _prime_based_seed(n_primes, max_prime=10000):
    """Generate a set based on prime-step gaps.
    U = {0, p1, p1+p2, p1+p2+p3, ...}
    This exploits additive independence of primes.
    """
    # Simple prime generation
    def _is_prime(m):
        if m < 2:
            return False
        if m < 4:
            return True
        if m % 2 == 0 or m % 3 == 0:
            return False
        i = 5
        while i * i <= m:
            if m % i == 0 or m % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def _generate_primes(limit):
        if limit < 2:
            return []
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = _generate_primes(max_prime)
    if len(primes) < n_primes:
        # Generate more primes
        extra = []
        candidate = max_prime + 1
        while len(extra) < n_primes - len(primes):
            if _is_prime(candidate):
                extra.append(candidate)
            candidate += 2
        primes = primes + extra
    
    selected = primes[:n_primes]
    U = [0]
    current = 0
    for p in selected:
        current += p
        U.append(current)
    return U


def _get_heuristic_seeds():
    """Get a variety of heuristic seeds for initialization."""
    seeds = []
    
    # Golomb rulers of various sizes
    for n in range(3, 15):
        try:
            seeds.append(_golomb_ruler(n))
        except Exception:
            pass
    
    # Power-of-2 minus 1 pattern
    for k in range(3, 12):
        seeds.append([2**i - 1 for i in range(k)])
    
    # Prime-based seeds
    for n_primes in range(3, 20):
        try:
            seeds.append(_prime_based_seed(n_primes, max_prime=10000))
        except Exception:
            pass
    
    return seeds


def _local_search(U, max_iterations=5000, time_limit=10.0):
    """Improved local search with multiple perturbation strategies."""
    U = sorted(set(U))
    best_score = _compute_score(U)
    best_U = list(U)
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try different perturbation strategies
        strategy = iteration % 4
        
        if strategy == 0:
            # Add a random element
            max_u = best_U[-1] if best_U else 0
            margin = max(10, max_u // 3)
            new_val = random.randint(0, max_u + margin)
            if new_val not in best_U:
                candidate = sorted(best_U + [new_val])
            else:
                continue
        elif strategy == 1:
            # Remove a random element (not 0)
            if len(best_U) > 1:
                idx = random.randint(1, len(best_U) - 1)
                candidate = [x for i, x in enumerate(best_U) if i != idx]
            else:
                continue
        elif strategy == 2:
            # Swap: remove one, add another
            if len(best_U) > 1:
                idx = random.randint(1, len(best_U) - 1)
                max_u = best_U[-1]
                margin = max(10, max_u // 3)
                new_val = random.randint(0, max_u + margin)
                candidate = [x for i, x in enumerate(best_U) if i != idx]
                if new_val not in candidate:
                    candidate.append(new_val)
                candidate = sorted(candidate)
            else:
                continue
        else:
            # Perturb an element
            if len(best_U) > 1:
                idx = random.randint(1, len(best_U) - 1)
                delta = random.randint(-5, 5)
                new_val = max(0, best_U[idx] + delta)
                if new_val not in best_U:
                    candidate = list(best_U)
                    candidate[idx] = new_val
                    candidate = sorted(candidate)
                else:
                    continue
            else:
                continue
        
        if not candidate or 0 not in candidate:
            continue
        
        score = _compute_score(candidate)
        if score > best_score:
            best_score = score
            best_U = candidate
    
    return best_U, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 second limit for safety
    
    # Get heuristic seeds
    seeds = _get_heuristic_seeds()
    
    # Find the best seed
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    for seed in seeds:
        if time.time() > deadline - 60:
            break
        if not seed or 0 not in seed:
            continue
        seed = sorted(seed)
        score = _compute_score(seed)
        if score > best_score:
            best_score = score
            best_U = list(seed)
    
    # Local search from the best seed
    if time.time() < deadline - 20:
        U, score = _local_search(best_U, max_iterations=30000, time_limit=40)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Try a few more local search rounds with different perturbations
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            U = list(best_U)
            # Perturb
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                delta = random.randint(-10, 10)
                new_val = max(0, U[idx] + delta)
                if new_val not in U:
                    U[idx] = new_val
                    U = sorted(U)
            
            U, score = _local_search(U, max_iterations=10000, time_limit=8)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.append(0)
    
    return sorted(best_U)

# EVOLVE-BLOCK-END
