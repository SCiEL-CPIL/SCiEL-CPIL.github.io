import time
import random
import math

N = 5000
MAX_TIME = 295.0  # Safety margin

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
# Primes that can be violated (p <= len(tuple) <= N)
# But actually, for a tuple of size k, only p <= k can be violated.
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
        # Check if adding x violates admissibility
        # x violates if for some prime p <= len(S)+1, adding x makes all residues present
        # Actually, we need to check: after adding x, is the tuple still admissible?
        # The tuple S' = S ∪ {x} is admissible iff for every prime p <= len(S'), some residue is missing.
        # Since len(S') = len(S) + 1, we need to check primes p <= len(S) + 1.
        # For primes p <= len(S): S was admissible, so some residue was missing. Adding x can only
        # fill that missing residue if x % p == that missing residue. If it does, we need to check
        # if it fills ALL missing residues (which can't happen since only one element is added).
        # Wait, that's not right. S is admissible means for each p <= len(S), at least one residue is missing.
        # Adding x might fill the last missing residue for some p, making all residues present -> violation.
        # For primes p where len(S) < p <= len(S)+1: these are new primes to check.
        # Actually, if p > len(S), then len(S) < p, so it's impossible to have all p residues present
        # with only len(S) elements. So no violation from those.
        # Wait, that's true: if |S| < p, you can't have all p residues. So only need to check p <= len(S).
        # But after adding x, len(S') = len(S) + 1. So we need p <= len(S) + 1.
        # For p <= len(S): S was admissible, so for each such p, at least one residue was missing in S.
        # Adding x can fill at most one residue class. So if there was exactly one missing residue,
        # and x % p equals that residue, then S' would have all residues present -> violation.
        # If there were 2+ missing residues, adding x can't fill all of them.
        
        len_S = len(S)
        # Check primes p <= len_S (since for p > len_S, S already has < p elements, so can't be violated)
        # Actually, for p in range where len_S < p <= len_S + 1, i.e., p = len_S + 1 if prime:
        # S has len_S elements, so at most len_S residues present. Since len_S < p, at least one residue missing.
        # So no violation from p = len_S + 1.
        # So only need to check p <= len_S.
        
        # For each prime p <= len_S, check if x would fill the last missing residue
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
                # Only one missing residue
                if x % p == missing[0]:
                    # Adding x would fill it, making all residues present
                    violated = True
                    break
            # If len(missing) >= 2, adding x can't fill all
        if not violated:
            S_set.add(x)
            S.append(x)
            # We could continue, but to be safe and simple, let's restart
            # Actually, let's just continue; it's a greedy local search
            # But restarting ensures we always check against the current S
            # For efficiency, let's just continue and recheck later
    
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
