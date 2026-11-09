import time
import random
import math

N = 5000
MAX_TIME = 280.0  # Safety margin (300s limit, keep 20s buffer)

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

def _greedy_construct(strategy=0, seed=None):
    """
    strategy 0: Deterministic (remove min count, tie-break by lowest residue)
    strategy 1: Random tie-break among min count
    strategy 2: Random choice among min count + 1
    strategy 3: Score-based - prefer residues that help other primes
    """
    if seed is not None:
        random.seed(seed)
    
    S = set(range(N + 1))
    
    while True:
        len_S = len(S)
        if len_S == 0:
            break
            
        # Find violating prime
        viol_p = None
        for p in PRIMES:
            if p > len_S:
                break
            residues = set(x % p for x in S)
            if len(residues) == p:
                viol_p = p
                break
        
        if viol_p is None:
            break
        
        p = viol_p
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
        
        min_count = min(counts)
        candidates = [r for r in range(p) if counts[r] == min_count]
        
        if strategy == 0:
            r_to_remove = candidates[0]
        elif strategy == 1:
            r_to_remove = random.choice(candidates)
        elif strategy == 2:
            candidates2 = [r for r in range(p) if counts[r] <= min_count + 1]
            r_to_remove = random.choice(candidates2)
        elif strategy == 3:
            # Score-based: for each candidate, estimate impact on other primes
            best_r = None
            best_score = float('inf')
            for r in candidates:
                # Score = number of elements removed (lower is better)
                # Plus penalty if this removal doesn't help other primes
                score = counts[r]
                # Bonus if removing this residue helps satisfy other prime constraints
                # (i.e., elements in this residue class have diverse residues mod other primes)
                # Simplified: just use count as score
                if score < best_score or (score == best_score and (best_r is None or r < best_r)):
                    best_score = score
                    best_r = r
            r_to_remove = best_r
        else:
            r_to_remove = candidates[0]
            
        # Remove elements
        S = {x for x in S if x % p != r_to_remove}
        
    return list(S)

def _local_improve(S, primes, max_iters=50):
    """
    Try to add back removed elements to increase size while maintaining admissibility.
    Returns improved set.
    """
    S = set(S)
    full_set = set(range(N + 1))
    removed = full_set - S
    
    for _ in range(max_iters):
        improved = False
        # Try adding back one element at a time
        for x in list(removed):
            S_new = S | {x}
            # Check if S_new is admissible
            if _get_violating_prime(S_new, primes) is None:
                S = S_new
                removed.discard(x)
                improved = True
                break
        if not improved:
            break
    
    return list(S)

def construct():
    start_time = time.time()
    best_size = 0
    best_tuple = []
    
    # 1. Deterministic
    t0 = time.time()
    res = _greedy_construct(strategy=0)
    if len(res) > best_size:
        best_size = len(res)
        best_tuple = res
    
    # 2. Try local improvement on the best solution
    if time.time() - start_time < MAX_TIME:
        improved = _local_improve(best_tuple, PRIMES)
        if len(improved) > best_size:
            best_size = len(improved)
            best_tuple = improved
    
    # 3. Randomized variants
    for i in range(2000):
        if time.time() - start_time > MAX_TIME - 10:  # Keep 10s buffer
            break
        
        # Vary strategy and seed
        strat = random.randint(0, 3)
        res = _greedy_construct(strategy=strat, seed=i)
        if len(res) > best_size:
            best_size = len(res)
            best_tuple = res
            
            # Try local improvement on new best
            if time.time() - start_time < MAX_TIME - 5:
                improved = _local_improve(best_tuple, PRIMES)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_tuple = improved
            
    return best_tuple
