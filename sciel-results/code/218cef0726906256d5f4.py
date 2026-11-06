import math
import random
import time
from fractions import Fraction

# ---- primes up to 1000 (fixed) ----
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = [False] * len(sieve[start:n + 1:step])
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_upto(1000)
NUM_PRIMES = len(_PRIMES)
_PRIME_TO_IDX = {p: i for i, p in enumerate(_PRIMES)}
_RECIPROCAL = [Fraction(1, p) for p in _PRIMES]


def _coverage_length(assign, max_check=10000):
    """assign: list of residues (0..p-1) per prime; 0 means 'not used'.
    Return largest L such that every 1..L is covered by some class."""
    covered = bytearray(max_check + 1)
    for idx, r in enumerate(assign):
        if r == 0:
            continue
        p = _PRIMES[idx]
        for n in range(r, max_check + 1, p):
            covered[n] = 1
    L = 0
    for n in range(1, max_check + 1):
        if covered[n]:
            L = n
        else:
            break
    return L


def _is_valid(assign):
    c = Fraction(0)
    for idx, r in enumerate(assign):
        if r != 0:
            c += _RECIPROCAL[idx]
    return c <= Fraction(1, 1)


def _initial_solution():
    """Seed: i-th prime from 13 upward covers integer i+1."""
    primes_from_13 = [p for p in _PRIMES if p >= 13]
    assign = [0] * NUM_PRIMES
    for i, p in enumerate(primes_from_13):
        idx = _PRIME_TO_IDX[p]
        r = (i + 1) % p
        if r == 0:
            r = 1
        assign[idx] = r
    return assign


def _simulated_annealing(assign, time_limit, target_L, seed=42):
    """
    Simulated annealing with incremental coverage updates.
    Escapes local minima by accepting temporary coverage drops.
    Returns (best_assign, best_L).
    """
    start = time.time()
    rng = random.Random(seed)
    
    # Compute initial state
    covered = bytearray(target_L + 1)
    for idx, r in enumerate(assign):
        if r != 0:
            p = _PRIMES[idx]
            for n in range(r, target_L + 1, p):
                covered[n] = 1
    
    current_L = 0
    for n in range(1, target_L + 1):
        if covered[n]:
            current_L = n
        else:
            break
    
    best_assign = assign[:]
    best_L = current_L
    current_cost = Fraction(0)
    for idx, r in enumerate(assign):
        if r != 0:
            current_cost += _RECIPROCAL[idx]
    
    # Temperature schedule
    T = 1.0
    T_min = 0.001
    cooling = 0.995
    
    # Track used primes for quick lookup
    used_mask = [r != 0 for r in assign]
    
    step = 0
    while time.time() - start < time_limit:
        step += 1
        
        if T < T_min:
            T = T_min
        
        # Choose a random prime to modify
        # Bias toward primes that are currently used or small
        candidates = []
        for i in range(NUM_PRIMES):
            if _PRIMES[i] > 200:  # Limit to smaller primes for speed
                break
            candidates.append(i)
        
        idx = rng.choice(candidates)
        p = _PRIMES[idx]
        old_r = assign[idx]
        
        # Choose new residue
        if old_r == 0:
            # Add a new prime
            if current_cost + _RECIPROCAL[idx] > Fraction(1, 1):
                T *= cooling
                continue
            new_r = rng.randint(1, p)
        else:
            # Modify existing or remove
            choice = rng.random()
            if choice < 0.1:
                new_r = 0  # Remove
            else:
                new_r = rng.randint(1, p)
        
        if new_r == old_r:
            T *= cooling
            continue
        
        # Compute delta: how does coverage change?
        # Remove old coverage, add new coverage
        delta_covered = bytearray(target_L + 1)
        delta = 0
        
        if old_r != 0:
            for n in range(old_r, target_L + 1, p):
                if covered[n] and (not any(assign[j] != 0 and n % _PRIMES[j] == assign[j] 
                                            for j in range(NUM_PRIMES) if j != idx and assign[j] != 0)):
                    delta -= 1
                delta_covered[n] = -1
        
        if new_r != 0:
            for n in range(new_r, target_L + 1, p):
                if not covered[n]:
                    delta += 1
                delta_covered[n] = 1
        
        # Simplified: just compute new L directly for accuracy
        # This is slower but correct
        if old_r != 0:
            for n in range(old_r, target_L + 1, p):
                # Check if still covered by another prime
                still_covered = False
                for j in range(NUM_PRIMES):
                    if j == idx or assign[j] == 0:
                        continue
                    if n % _PRIMES[j] == assign[j]:
                        still_covered = True
                        break
                if not still_covered:
                    covered[n] = 0
        
        if new_r != 0:
            for n in range(new_r, target_L + 1, p):
                covered[n] = 1
        
        # Compute new L
        new_L = 0
        for n in range(1, target_L + 1):
            if covered[n]:
                new_L = n
            else:
                break
        
        # Cost change
        cost_delta = Fraction(0)
        if old_r != 0 and new_r == 0:
            cost_delta = -_RECIPROCAL[idx]
        elif old_r == 0 and new_r != 0:
            cost_delta = _RECIPROCAL[idx]
        
        if current_cost + cost_delta > Fraction(1, 1):
            # Invalid, revert
            if new_r != 0:
                for n in range(new_r, target_L + 1, p):
                    if not any(assign[j] != 0 and n % _PRIMES[j] == assign[j] 
                               for j in range(NUM_PRIMES) if j != idx and assign[j] != 0):
                        covered[n] = 0
            if old_r != 0:
                for n in range(old_r, target_L + 1, p):
                    covered[n] = 1
            T *= cooling
            continue
        
        # Accept/reject
        if new_L >= current_L:
            accept = True
        else:
            # Accept with probability exp(-delta / T)
            delta = new_L - current_L
            if delta < 0:
                prob = math.exp(delta / T)
                accept = rng.random() < prob
            else:
                accept = True
        
        if accept:
            assign[idx] = new_r
            current_cost += cost_delta
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_assign = assign[:]
        
        else:
            # Revert
            if new_r != 0:
                for n in range(new_r, target_L + 1, p):
                    if not any(assign[j] != 0 and n % _PRIMES[j] == assign[j] 
                               for j in range(NUM_PRIMES) if j != idx and assign[j] != 0):
                        covered[n] = 0
            if old_r != 0:
                for n in range(old_r, target_L + 1, p):
                    covered[n] = 1
            current_cost -= cost_delta
        
        T *= cooling
    
    return best_assign, best_L


def _local_search(assign, time_limit, target_L):
    """
    Hill-climbing local search: try single-residue changes that improve L.
    Returns (best_assign, best_L).
    """
    start = time.time()
    rng = random.Random(123)
    
    covered = bytearray(target_L + 1)
    for idx, r in enumerate(assign):
        if r != 0:
            p = _PRIMES[idx]
            for n in range(r, target_L + 1, p):
                covered[n] = 1
    
    current_L = 0
    for n in range(1, target_L + 1):
        if covered[n]:
            current_L = n
        else:
            break
    
    best_assign = assign[:]
    best_L = current_L
    current_cost = Fraction(0)
    for idx, r in enumerate(assign):
        if r != 0:
            current_cost += _RECIPROCAL[idx]
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        
        # Try modifying each small prime
        for idx in range(NUM_PRIMES):
            if _PRIMES[idx] > 150:
                break
            p = _PRIMES[idx]
            old_r = assign[idx]
            
            # Try all residues (and 0)
            candidates = list(range(0, p + 1))
            rng.shuffle(candidates)
            
            for new_r in candidates:
                if new_r == old_r:
                    continue
                
                # Check cost
                cost_delta = Fraction(0)
                if old_r != 0 and new_r == 0:
                    cost_delta = -_RECIPROCAL[idx]
                elif old_r == 0 and new_r != 0:
                    cost_delta = _RECIPROCAL[idx]
                
                if current_cost + cost_delta > Fraction(1, 1):
                    continue
                
                # Compute new coverage
                if old_r != 0:
                    for n in range(old_r, target_L + 1, p):
                        still_covered = False
                        for j in range(NUM_PRIMES):
                            if j == idx or assign[j] == 0:
                                continue
                            if n % _PRIMES[j] == assign[j]:
                                still_covered = True
                                break
                        if not still_covered:
                            covered[n] = 0
                
                if new_r != 0:
                    for n in range(new_r, target_L + 1, p):
                        covered[n] = 1
                
                new_L = 0
                for n in range(1, target_L + 1):
                    if covered[n]:
                        new_L = n
                    else:
                        break
                
                if new_L > current_L:
                    assign[idx] = new_r
                    current_cost += cost_delta
                    current_L = new_L
                    improved = True
                    if new_L > best_L:
                        best_L = new_L
                        best_assign = assign[:]
                    break  # Restart scanning after improvement
                else:
                    # Revert
                    if new_r != 0:
                        for n in range(new_r, target_L + 1, p):
                            if not any(assign[j] != 0 and n % _PRIMES[j] == assign[j] 
                                       for j in range(NUM_PRIMES) if j != idx and assign[j] != 0):
                                covered[n] = 0
                    if old_r != 0:
                        for n in range(old_r, target_L + 1, p):
                            covered[n] = 1
    
    return best_assign, best_L


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill
    
    # Start with seed solution
    best_assign = _initial_solution()
    best_L = _coverage_length(best_assign)
    
    if not _is_valid(best_assign):
        best_assign = [0] * NUM_PRIMES
        best_L = 0
    
    # Phase 1: Local search to quickly improve
    target_L = max(best_L + 200, 1200)
    time_left = deadline - time.time() - 20
    if time_left > 5:
        ls_assign, ls_L = _local_search(best_assign, time_left, target_L)
        if ls_L > best_L and _is_valid(ls_assign):
            best_assign = ls_assign
            best_L = ls_L
    
    # Phase 2: Simulated annealing for deeper exploration
    target_L = max(best_L + 300, 1500)
    time_left = deadline - time.time() - 10
    if time_left > 10:
        sa_assign, sa_L = _simulated_annealing(best_assign, time_left, target_L, seed=42)
        if sa_L > best_L and _is_valid(sa_assign):
            best_assign = sa_assign
            best_L = sa_L
    
    # Phase 3: Another local search pass
    target_L = max(best_L + 200, 1300)
    time_left = deadline - time.time() - 5
    if time_left > 5:
        ls_assign, ls_L = _local_search(best_assign, time_left, target_L)
        if ls_L > best_L and _is_valid(ls_assign):
            best_assign = ls_assign
            best_L = ls_L
    
    # Final validation
    if best_assign is not None and _is_valid(best_assign):
        return [[_PRIMES[i], r] for i, r in enumerate(best_assign) if r != 0]
    else:
        # Fallback to seed
        primes_from_13 = [p for p in _PRIMES if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_from_13)]
