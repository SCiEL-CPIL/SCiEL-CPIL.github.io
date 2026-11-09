"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt, exp

P = {'n': 256}

def _is_prime(n):
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

def _can_add(candidate, current_set):
    """Check if candidate can be added to current_set without violating the constraint.
    For every three distinct members a,b,c in the set, a must not divide b*c."""
    cs = list(current_set)
    n = len(cs)
    
    # Case 1: candidate plays role of 'a': candidate ∤ b*c for all distinct b,c in current_set
    for i in range(n):
        for j in range(i+1, n):
            if (cs[i] * cs[j]) % candidate == 0:
                return False
    
    # Case 2: candidate plays role of 'b' (or 'c'): for each a in current_set, a ∤ candidate * b
    # where b is another distinct element from current_set
    for i in range(n):
        a = cs[i]
        for j in range(n):
            if j == i:
                continue
            b = cs[j]
            if (candidate * b) % a == 0:
                return False
    return True

def _greedy_construct():
    """Greedy construction starting from primes."""
    primes = [a for a in range(2, P['n']+1) if _is_prime(a)]
    current = set(primes)
    
    # Try adding composites in increasing order
    composites = [a for a in range(2, P['n']+1) if not _is_prime(a)]
    for cand in composites:
        if _can_add(cand, current):
            current.add(cand)
    return current

def _simulated_annealing(initial_set, deadline):
    """Simulated Annealing to improve the set size."""
    best_set = set(initial_set)
    current_set = set(initial_set)
    best_size = len(best_set)
    
    T = 1.0
    cooling = 0.999
    
    all_elements = list(range(2, P['n']+1))
    
    while time.time() < deadline - 5 and T > 0.01:
        # Pick a random element to remove
        if not current_set:
            break
        u = random.choice(list(current_set))
        
        # Pick a random candidate to add
        # Bias towards composites? Or just random.
        v = random.choice(all_elements)
        
        if v in current_set:
            continue
            
        trial_set = current_set - {u}
        
        if _can_add(v, trial_set):
            new_set = trial_set | {v}
            new_size = len(new_set)
            
            delta = new_size - len(current_set)
            
            # Accept if better, or with probability exp(delta/T)
            if delta >= 0 or random.random() < exp(delta / T):
                current_set = new_set
                if new_size > best_size:
                    best_set = set(new_set)
                    best_size = new_size
        
        T *= cooling
        
    return best_set

def construct():
    deadline = time.time() + 50  # safety margin
    
    # Phase 1: Greedy construction
    greedy_set = _greedy_construct()
    
    # Phase 2: Simulated Annealing
    result = _simulated_annealing(greedy_set, deadline)
    
    return [int(x) for x in sorted(result)]
# EVOLVE-BLOCK-END
