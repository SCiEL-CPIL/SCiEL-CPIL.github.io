import time
import random
import math

N = 5000
MAX_TIME = 290.0  # Safety margin

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

def _is_admissible(S, primes):
    """
    Check if S is admissible.
    Returns the first violating prime if any, else None.
    """
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
            counts = [0] * p
            for x in S:
                counts[x % p] += 1
            if all(c > 0 for c in counts):
                viol_p = p
                break
        
        if viol_p is None:
            break
        
        p = viol_p
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
        
        min_count = min(counts)
        
        # Determine candidates for removal
        candidates = [r for r in range(p) if counts[r] == min_count]
        
        if strategy == 0:
            r_to_remove = candidates[0]
        elif strategy == 1:
            r_to_remove = random.choice(candidates)
        elif strategy == 2:
            # Include classes with count <= min_count + 1
            candidates2 = [r for r in range(p) if counts[r] <= min_count + 1]
            r_to_remove = random.choice(candidates2)
        else:
            r_to_remove = candidates[0]
            
        # Remove elements
        S = {x for x in S if x % p != r_to_remove}
        
    return list(S)

def _local_improve(S, max_iters=500, seed=None):
    """
    Try to improve S by local moves:
    1. Try to add back an element that was removed (if we track removed elements)
       - This is hard without tracking, so we'll try a different approach.
    2. Actually, since we only remove, local improvement is tricky.
    
    Alternative: Random restarts with different initial conditions or strategies.
    Or: Simulated annealing-like approach where we allow temporary violations?
    No, that's complex.
    
    Let's try a different local improvement:
    - Take the current S.
    - For each prime p <= len(S), we know we removed one residue class.
    - But we don't know which one.
    
    Better idea: Use a different constructive approach.
    Instead of removing from the full set, try to BUILD the set by adding elements.
    Start with an empty set, and add elements one by one, ensuring admissibility.
    But that's hard to ensure.
    
    Alternative: Use a bitset representation for faster manipulation.
    """
    pass

def _greedy_bitset():
    """
    Use a bitset (or boolean array) for faster set operations.
    """
    S = [True] * (N + 1)
    count = N + 1
    
    while True:
        # Find violating prime
        viol_p = None
        for p in PRIMES:
            if p > count:
                break
            # Count residues
            counts = [0] * p
            for x in range(N + 1):
                if S[x]:
                    counts[x % p] += 1
            if all(c > 0 for c in counts):
                viol_p = p
                break
        
        if viol_p is None:
            break
        
        p = viol_p
        counts = [0] * p
        for x in range(N + 1):
            if S[x]:
                counts[x % p] += 1
        
        min_count = min(counts)
        candidates = [r for r in range(p) if counts[r] == min_count]
        r_to_remove = random.choice(candidates)
        
        # Remove elements
        removed = 0
        for x in range(N + 1):
            if S[x] and x % p == r_to_remove:
                S[x] = False
                removed += 1
        count -= removed
    
    return [x for x in range(N + 1) if S[x]]

def construct():
    start_time = time.time()
    best_size = 0
    best_tuple = []
    
    # 1. Deterministic greedy
    t0 = time.time()
    res = _greedy_construct(strategy=0)
    if len(res) > best_size:
        best_size = len(res)
        best_tuple = res
    
    # 2. Randomized variants
    for i in range(1000):
        if time.time() - start_time > MAX_TIME:
            break
        
        strat = random.randint(1, 2)
        res = _greedy_construct(strategy=strat, seed=i)
        if len(res) > best_size:
            best_size = len(res)
            best_tuple = res
    
    return best_tuple
