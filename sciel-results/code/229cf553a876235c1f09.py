import time
import random
import numpy as np

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)
_L_LIMIT = 10000

def _build_count(residues):
    count = np.zeros(_L_LIMIT + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        if start <= _L_LIMIT:
            count[start::_PRIMES[i]] += 1
    return count

def _find_L(count):
    below = np.where(count[1:_L_LIMIT+1] < 2)[0]
    if len(below) == 0:
        return _L_LIMIT
    return int(below[0])

def _greedy_init():
    """
    Greedy initialization: for each x from 1 upward, ensure at least 2 primes cover x.
    We assign residues to primes such that x is covered.
    """
    residues = [0] * _NPRIMES
    # Track coverage count for each x
    # We'll build this incrementally
    
    # For efficiency, let's use a different approach:
    # Start with all residues = 0 (which covers multiples of p)
    # Then for x from 1 to some limit, if coverage < 2, assign a prime to cover x
    
    # Actually, let's use a smarter greedy:
    # For x from 1 to L_LIMIT, we want at least 2 primes p where r_p = x % p
    # A prime p "can cover" x if we set r_p = x % p
    # But setting r_p = x % p means p covers all numbers congruent to x mod p
    
    # Strategy: Process x from 1 upward. For each x, check how many primes currently cover it.
    # If < 2, pick a prime that doesn't already cover x and set its residue to x % p.
    # Prefer primes where x % p is small or where the prime is large (less "waste").
    
    # Let's track which primes cover each x.
    # Initially all r_p = 0, so p covers multiples of p.
    
    # Build initial count
    count = _build_count(residues)
    
    # For each x from 1 to L_LIMIT, ensure count[x] >= 2
    for x in range(1, _L_LIMIT + 1):
        if count[x] >= 2:
            continue
        
        # Need to add coverage for x
        # Find primes that don't currently cover x
        # A prime p covers x if (x - r_p) % p == 0, i.e., x % p == r_p
        
        # We want to pick a prime p and set r_p = x % p
        # This will change what p covers: previously it covered numbers ≡ old_r mod p,
        # now it covers numbers ≡ x % p mod p
        
        # For simplicity, pick a prime that currently doesn't cover x
        # and where the change is "cheap" (doesn't reduce coverage of other important numbers)
        
        best_i = -1
        best_score = -1
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            new_r = x % p
            old_r = residues[i]
            
            if new_r == old_r:
                # Already covers x
                continue
            
            # Check if p currently covers x
            if (x - old_r) % p == 0:
                # Already covers x, but we said count[x] < 2, so this shouldn't happen
                # unless there's a bug
                continue
            
            # Score: prefer primes where old_r is 0 (default) or where the prime is large
            # Large primes have sparser coverage, so changing them is less disruptive
            score = p  # Prefer large primes
            
            if score > best_score:
                best_score = score
                best_i = i
        
        if best_i >= 0:
            i = best_i
            p = _PRIMES[i]
            old_r = residues[i]
            new_r = x % p
            
            # Update count
            start_old = p if old_r == 0 else old_r
            if start_old <= _L_LIMIT:
                count[start_old::p] -= 1
            
            start_new = p if new_r == 0 else new_r
            if start_new <= _L_LIMIT:
                count[start_new::p] += 1
            
            residues[i] = new_r
        
        # If we still can't cover x, just move on
    
    return residues

def _local_search(residues, deadline):
    """
    Local search to improve the solution.
    """
    count = _build_count(residues)
    current_L = _find_L(count)
    best_L = current_L
    best_residues = list(residues)
    
    stagnation = 0
    max_stag = 100
    
    while time.time() < deadline - 5:
        if current_L >= _L_LIMIT:
            break
        
        gap = current_L + 1
        
        # Try to fix the gap
        best_move = None
        min_cost = float('inf')
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            new_r = gap % p
            old_r = residues[i]
            
            if new_r == old_r:
                continue
            
            # Cost: number of positions in [1, current_L] that drop below 2
            start_old = p if old_r == 0 else old_r
            if start_old > current_L:
                cost = 0
            else:
                # Positions where old_r covers
                pos = np.arange(start_old, current_L + 1, p)
                if len(pos) == 0:
                    cost = 0
                else:
                    cost = int(np.sum(count[pos] == 2))
            
            if cost < min_cost:
                min_cost = cost
                best_move = (i, new_r)
                if cost == 0:
                    break
        
        if best_move:
            i, new_r = best_move
            p = _PRIMES[i]
            old_r = residues[i]
            start_old = p if old_r == 0 else old_r
            start_new = p if new_r == 0 else new_r
            
            count[start_old::p] -= 1
            count[start_new::p] += 1
            residues[i] = new_r
            
            new_L = _find_L(count)
            
            if new_L > current_L:
                current_L = new_L
                stagnation = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = list(residues)
            else:
                # Revert
                count[start_new::p] -= 1
                count[start_old::p] += 1
                residues[i] = old_r
                stagnation += 1
        else:
            stagnation += 1
        
        if stagnation > max_stag:
            break
    
    return best_residues, best_L

def _simulated_annealing(residues, deadline):
    """
    Simulated annealing to escape local optima.
    """
    count = _build_count(residues)
    current_L = _find_L(count)
    best_L = current_L
    best_residues = list(residues)
    
    temperature = 100.0
    cooling = 0.995
    
    while time.time() < deadline - 5 and temperature > 0.1:
        # Pick a random prime to change
        i = random.randint(0, _NPRIMES - 1)
        p = _PRIMES[i]
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        start_old = p if old_r == 0 else old_r
        start_new = p if new_r == 0 else new_r
        
        count[start_old::p] -= 1
        count[start_new::p] += 1
        residues[i] = new_r
        
        new_L = _find_L(count)
        
        # Accept if better, or with probability exp(-delta/T)
        if new_L >= current_L:
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
        else:
            # Revert
            count[start_new::p] -= 1
            count[start_old::p] += 1
            residues[i] = old_r
        
        temperature *= cooling
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    random.seed(42)
    
    # Phase 1: Greedy initialization
    residues = _greedy_init()
    
    # Phase 2: Local search
    residues, L = _local_search(residues, deadline)
    
    # Phase 3: Simulated annealing with multiple restarts
    while time.time() < deadline - 10:
        # Perturb the current solution
        perturbed = list(residues)
        for _ in range(5):
            i = random.randint(0, _NPRIMES - 1)
            p = _PRIMES[i]
            perturbed[i] = random.randint(0, p - 1)
        
        # Local search from perturbed state
        residues, L = _local_search(perturbed, deadline)
        
        # If we have time, do a bit of simulated annealing
        if time.time() < deadline - 15:
            residues, L = _simulated_annealing(residues, deadline)
    
    return [int(r) for r in residues]
