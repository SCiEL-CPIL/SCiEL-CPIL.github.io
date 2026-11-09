import time
import random
import math

N = 5000
MAX_TIME = 295.0  # Safety margin before 300s limit

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

def _local_improve(S, max_iters=2000, seed=None):
    """
    Local improvement: try to add back elements that were removed.
    We track which elements are missing and try to add them back
    if doing so doesn't create a new violation.
    """
    if seed is not None:
        random.seed(seed)
    
    S_set = set(S)
    all_elements = set(range(N + 1))
    removed = all_elements - S_set
    
    if not removed:
        return list(S_set)
    
    # Shuffle removed elements
    removed_list = list(removed)
    random.shuffle(removed_list)
    
    improved = False
    for x in removed_list:
        # Try adding x back
        S_set.add(x)
        
        # Check if still admissible
        len_S = len(S_set)
        admissible = True
        for p in PRIMES:
            if p > len_S:
                break
            residues = set(xi % p for xi in S_set)
            if len(residues) == p:
                admissible = False
                break
        
        if not admissible:
            S_set.discard(x)
        else:
            improved = True
    
    # If we improved, try another pass
    if improved:
        removed = all_elements - S_set
        if removed:
            return _local_improve(list(S_set), max_iters=max_iters//2, seed=None)
    
    return list(S_set)

def _construct_with_backtrack():
    """
    A different approach: build the set incrementally.
    Start with a random subset and try to grow it.
    """
    # Start with a moderately sized random set
    size = random.randint(2000, 4000)
    S = set(random.sample(range(N + 1), size))
    
    # Try to add elements from the complement
    complement = set(range(N + 1)) - S
    comp_list = list(complement)
    random.shuffle(comp_list)
    
    for x in comp_list:
        S.add(x)
        len_S = len(S)
        admissible = True
        for p in PRIMES:
            if p > len_S:
                break
            residues = set(xi % p for xi in S)
            if len(residues) == p:
                admissible = False
                break
        
        if not admissible:
            S.discard(x)
    
    return list(S)

def _simulated_annealing(start_set, max_iters=5000, T_start=100, T_end=1, seed=None):
    """
    Simulated annealing: allow temporary violations, penalize by size loss.
    """
    if seed is not None:
        random.seed(seed)
    
    S = set(start_set)
    best_S = set(S)
    best_size = len(S)
    
    complement = set(range(N + 1)) - S
    
    for it in range(max_iters):
        T = T_start * (T_end / T_start) ** (it / max_iters)
        
        # Random move: add or remove a random element
        if random.random() < 0.5 and complement:
            x = random.choice(list(complement))
            S.add(x)
            complement.discard(x)
        else:
            x = random.choice(list(S))
            S.discard(x)
            complement.add(x)
        
        # Check admissibility
        len_S = len(S)
        viol = None
        for p in PRIMES:
            if p > len_S:
                break
            residues = set(xi % p for xi in S)
            if len(residues) == p:
                viol = p
                break
        
        if viol is None:
            # Admissible
            if len_S > best_size:
                best_size = len_S
                best_S = set(S)
        else:
            # Violation: reject unless at high temperature
            if random.random() < math.exp(-1.0 / max(T, 0.001)):
                pass  # Keep the move
            else:
                # Revert
                if x in complement:
                    S.add(x)
                    complement.discard(x)
                else:
                    S.discard(x)
                    complement.add(x)
    
    return list(best_S)

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
    
    # 2. Local improvement on the greedy result
    improved = _local_improve(best_tuple, seed=42)
    if len(improved) > best_size:
        best_size = len(improved)
        best_tuple = improved
    
    # 3. Randomized greedy variants
    for i in range(200):
        if time.time() - start_time > MAX_TIME:
            break
        
        strat = random.randint(1, 2)
        res = _greedy_construct(strategy=strat, seed=i)
        
        # Local improvement
        improved = _local_improve(res, max_iters=500, seed=i)
        if len(improved) > best_size:
            best_size = len(improved)
            best_tuple = improved
    
    # 4. Constructive approach
    if time.time() - start_time < MAX_TIME:
        for i in range(50):
            if time.time() - start_time > MAX_TIME:
                break
            res = _construct_with_backtrack()
            improved = _local_improve(res, max_iters=300, seed=i + 1000)
            if len(improved) > best_size:
                best_size = len(improved)
                best_tuple = improved
    
    # 5. Final local improvement pass
    if time.time() - start_time < MAX_TIME:
        improved = _local_improve(best_tuple, max_iters=1000, seed=9999)
        if len(improved) > best_size:
            best_tuple = improved
    
    return best_tuple
