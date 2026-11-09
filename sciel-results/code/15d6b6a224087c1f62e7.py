import time
import random
import math
import numpy as np

N = 5000
MAX_TIME = 285.0  # Safety margin (330s limit, keep 45s buffer)

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
                score = counts[r]
                if score < best_score or (score == best_score and (best_r is None or r < best_r)):
                    best_score = score
                    best_r = r
            r_to_remove = best_r
        else:
            r_to_remove = candidates[0]
            
        # Remove elements
        S = {x for x in S if x % p != r_to_remove}
        
    return list(S)

def _local_improve(S, primes, max_iters=100):
    """
    Try to add back removed elements to increase size while maintaining admissibility.
    Returns improved set.
    """
    S = set(S)
    full_set = set(range(N + 1))
    removed = full_set - S
    
    # Sort removed by some heuristic - try smaller numbers first, then larger
    removed_list = list(removed)
    random.shuffle(removed_list)
    
    for _ in range(max_iters):
        improved = False
        for x in removed_list:
            if x not in removed:
                continue
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

def _greedy_construct_np(strategy=0, seed=None):
    """
    Vectorized greedy construction using numpy for speed.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Start with all elements
    S = np.ones(N + 1, dtype=bool)
    
    while True:
        len_S = S.sum()
        if len_S == 0:
            break
        
        # Find violating prime
        viol_p = None
        for p in PRIMES:
            if p > len_S:
                break
            # Get residues of elements in S
            indices = np.where(S)[0]
            residues = indices % p
            if len(np.unique(residues)) == p:
                viol_p = p
                break
        
        if viol_p is None:
            break
        
        p = viol_p
        indices = np.where(S)[0]
        residues = indices % p
        
        # Count residues
        counts = np.bincount(residues, minlength=p)
        
        min_count = counts.min()
        candidates = np.where(counts == min_count)[0]
        
        if strategy == 0:
            r_to_remove = candidates[0]
        elif strategy == 1:
            r_to_remove = candidates[rng.randint(len(candidates))]
        elif strategy == 2:
            candidates2 = np.where(counts <= min_count + 1)[0]
            r_to_remove = candidates2[rng.randint(len(candidates2))]
        elif strategy == 3:
            # Score-based
            best_r = candidates[0]
            best_score = counts[best_r]
            for r in candidates[1:]:
                score = counts[r]
                if score < best_score:
                    best_score = score
                    best_r = r
            r_to_remove = best_r
        else:
            r_to_remove = candidates[0]
        
        # Remove elements with this residue
        S[indices[residues == r_to_remove]] = False
        
    return np.where(S)[0].tolist()

def _local_improve_np(S_list, primes, max_iters=100):
    """
    Try to add back removed elements using numpy for speed.
    """
    S = np.ones(N + 1, dtype=bool)
    for x in S_list:
        S[x] = True
    
    full_set = np.ones(N + 1, dtype=bool)
    removed = full_set & ~S
    
    removed_indices = np.where(removed)[0]
    np.random.shuffle(removed_indices)
    
    for _ in range(max_iters):
        improved = False
        for x in removed_indices:
            if S[x]:
                continue
            # Try adding x
            S[x] = True
            # Check admissibility
            len_S = S.sum()
            viol = None
            for p in primes:
                if p > len_S:
                    break
                indices = np.where(S)[0]
                residues = indices % p
                if len(np.unique(residues)) == p:
                    viol = p
                    break
            if viol is None:
                improved = True
                break
            else:
                S[x] = False
        if not improved:
            break
    
    return np.where(S)[0].tolist()

def construct():
    start_time = time.time()
    best_size = 0
    best_tuple = []
    
    # 1. Deterministic numpy version
    t0 = time.time()
    res = _greedy_construct_np(strategy=0, seed=42)
    if len(res) > best_size:
        best_size = len(res)
        best_tuple = res
    
    # 2. Try local improvement on the best solution
    if time.time() - start_time < MAX_TIME - 5:
        improved = _local_improve_np(best_tuple, PRIMES)
        if len(improved) > best_size:
            best_size = len(improved)
            best_tuple = improved
    
    # 3. Randomized variants with numpy (much faster per iteration)
    seed_counter = 0
    while time.time() - start_time < MAX_TIME - 15:
        strat = random.randint(0, 3)
        seed_counter += 1
        res = _greedy_construct_np(strategy=strat, seed=seed_counter)
        if len(res) > best_size:
            best_size = len(res)
            best_tuple = res
            
            # Try local improvement on new best
            if time.time() - start_time < MAX_TIME - 10:
                improved = _local_improve_np(best_tuple, PRIMES)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_tuple = improved
    
    return best_tuple
