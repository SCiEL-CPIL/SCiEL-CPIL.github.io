import time
import random
import math
import numpy as np

N = 5000
MAX_TIME = 295.0  # Safety margin before 330s kill

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
# Primes that can be violated for a tuple of size k: p <= k
# Since k <= N, we check all primes <= N.
PRIMES_FOR_CHECK = [p for p in PRIMES if p <= N]

def _get_violating_prime(S, primes):
    """Return the smallest violating prime, or None if admissible."""
    len_S = len(S)
    for p in primes:
        if p > len_S:
            break
        residues = set(x % p for x in S)
        if len(residues) == p:
            return p
    return None

def _is_admissible(S):
    """Check if S is admissible."""
    return _get_violating_prime(S, PRIMES_FOR_CHECK) is None

def _greedy_construct():
    """
    Greedy construction: start with [0, N], remove minority residue class
    for each violating prime until admissible.
    """
    S = list(range(N + 1))
    while True:
        viol_p = _get_violating_prime(S, PRIMES_FOR_CHECK)
        if viol_p is None:
            break
        p = viol_p
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
        min_count = min(counts)
        # Remove the residue class with minimum count (tie-break: lowest residue)
        r_to_remove = min(r for r in range(p) if counts[r] == min_count)
        S = [x for x in S if x % p != r_to_remove]
    return S

def _local_improve(S, max_time=1.0):
    """
    Local improvement: try to add back elements from [0, N] \ S
    that don't violate admissibility.
    Uses randomized ordering for diversity.
    """
    start_time = time.time()
    S_set = set(S)
    # Candidates to try adding
    candidates = [x for x in range(N + 1) if x not in S_set]
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() - start_time > max_time:
            break
        len_S = len(S)
        # Check primes p <= len_S
        violated = False
        for p in PRIMES_FOR_CHECK:
            if p > len_S:
                break
            # Count residues in S
            residues = [0] * p
            for y in S:
                residues[y % p] += 1
            # Find missing residues
            missing = [r for r in range(p) if residues[r] == 0]
            if len(missing) == 1:
                if x % p == missing[0]:
                    violated = True
                    break
        if not violated:
            S_set.add(x)
            S.append(x)
    
    return S

def _randomized_greedy(seed):
    """
    Randomized greedy: like greedy but with random tie-breaking.
    """
    random.seed(seed)
    S = list(range(N + 1))
    while True:
        viol_p = _get_violating_prime(S, PRIMES_FOR_CHECK)
        if viol_p is None:
            break
        p = viol_p
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
        min_count = min(counts)
        candidates = [r for r in range(p) if counts[r] == min_count]
        # Sometimes pick from min_count+1 for diversity
        if random.random() < 0.3:
            candidates = [r for r in range(p) if counts[r] <= min_count + 1]
        r_to_remove = random.choice(candidates)
        S = [x for x in S if x % p != r_to_remove]
    return S

def _branch_and_bound_greedy(seed, depth_limit=13):
    """
    Branch and bound: explore different residue class removals for small primes.
    Use numpy for vectorized operations.
    """
    random.seed(seed)
    S = np.arange(N + 1)
    S_set = set(S.tolist())
    
    # Process primes in order of size (smallest first)
    small_primes = [p for p in PRIMES_FOR_CHECK if p <= 17]
    
    def _process_prime(S, p, depth):
        """Try different residue classes to remove for prime p."""
        if depth >= depth_limit:
            return S
        
        # Check if p is violated
        if p > len(S):
            return S
        
        residues = np.array([x % p for x in S])
        counts = np.bincount(residues, minlength=p)
        
        if counts.min() > 0:
            # All residues present - need to remove one
            # Try each residue class as a candidate
            best_S = S
            best_size = len(S)
            
            # Sort residues by count (ascending) - remove smallest first
            order = np.argsort(counts)
            
            # Try a few candidates
            for r in order[:min(3, p)]:
                new_S = S[residues != r]
                if len(new_S) < best_size:
                    continue
                    
                # Recursively process next prime
                next_S = _process_prime(new_S, next_prime_idx, depth + 1) if next_prime_idx < len(small_primes) else new_S
                if len(next_S) > best_size:
                    best_size = len(next_S)
                    best_S = next_S
            
            return best_S
        else:
            # Already admissible for this prime, move on
            return S
    
    # Process primes sequentially
    current_S = S
    for idx, p in enumerate(small_primes):
        if time.time() - start_time > max_time:
            break
        if p > len(current_S):
            continue
        
        residues = np.array([x % p for x in current_S])
        counts = np.bincount(residues, minlength=p)
        
        if counts.min() > 0:
            # All residues present - need to remove one
            # Try different residue classes
            order = np.argsort(counts)
            
            # Try top 3 candidates
            best_S = current_S
            for r in order[:min(3, p)]:
                new_S = current_S[residues != r]
                if len(new_S) >= len(best_S):
                    # Check if this leads to a better final result
                    # For now, just take the one with largest size
                    if len(new_S) > len(best_S):
                        best_S = new_S
            
            current_S = best_S
    
    return current_S

def construct():
    start_time = time.time()
    best_size = 0
    best_tuple = []
    
    # Run deterministic greedy first
    S = _greedy_construct()
    if len(S) > best_size:
        best_size = len(S)
        best_tuple = list(S)
    
    # Run randomized greedy + local improvement with multiple seeds
    seed = 42
    while time.time() - start_time < MAX_TIME:
        elapsed = time.time() - start_time
        remaining = MAX_TIME - elapsed
        
        if remaining < 5.0:
            break
        
        S = _randomized_greedy(seed)
        # Local improvement
        time_left = MAX_TIME - (time.time() - start_time)
        if time_left > 2.0:
            S = _local_improve(S, max_time=min(2.0, time_left - 1.0))
        if len(S) > best_size:
            best_size = len(S)
            best_tuple = list(S)
        seed += 1
    
    return best_tuple
