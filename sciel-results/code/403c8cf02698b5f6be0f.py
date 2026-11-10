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
        count[start::_PRIMES[i]] += 1
    return count

def _find_L(count):
    below = np.where(count[1:_L_LIMIT+1] < 2)[0]
    if len(below) == 0:
        return _L_LIMIT
    return int(below[0])

def _greedy_init_small_primes():
    """Initialize with small primes prioritized for coverage."""
    residues = [0] * _NPRIMES
    # Sort primes: small primes first (higher density impact)
    prime_indices = list(range(_NPRIMES))
    prime_indices.sort(key=lambda i: _PRIMES[i])
    
    assigned = [False] * _NPRIMES
    # For each x from 1 upward, assign 2 small primes to cover x
    for x in range(1, _L_LIMIT + 1):
        found = 0
        for i in prime_indices:
            if assigned[i]:
                continue
            p = _PRIMES[i]
            r = x % p
            residues[i] = r
            assigned[i] = True
            found += 1
            if found >= 2:
                break
        if found < 2:
            break
    
    for i in range(_NPRIMES):
        if not assigned[i]:
            residues[i] = 0
    
    return residues

def _simulated_annealing(residues, deadline):
    """SA with vectorized count updates."""
    count = _build_count(residues)
    current_L = _find_L(count)
    best_L = current_L
    best_residues = list(residues)
    
    T = 100.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline - 5 and T > T_min:
        # Pick a random prime to change
        i = random.randint(0, _NPRIMES - 1)
        p = _PRIMES[i]
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        start_old = p if old_r == 0 else old_r
        start_new = p if new_r == 0 else new_r
        
        # Compute delta in L
        count[start_old::p] -= 1
        count[start_new::p] += 1
        
        new_L = _find_L(count)
        
        if new_L >= current_L:
            residues[i] = new_r
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
        else:
            # Accept with probability exp((new_L - current_L) / T)
            delta = new_L - current_L
            if random.random() < np.exp(delta / T):
                residues[i] = new_r
                current_L = new_L
            else:
                # Revert
                count[start_new::p] -= 1
                count[start_old::p] += 1
        
        T *= cooling
    
    return best_L, best_residues

def _targeted_local_search(residues, deadline):
    """Targeted search to fix gaps."""
    count = _build_count(residues)
    current_L = _find_L(count)
    best_L = current_L
    best_residues = list(residues)
    
    max_iters = 500
    for _ in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        if current_L >= _L_LIMIT:
            break
        
        gap = current_L + 1
        if count[gap] >= 2:
            while gap <= _L_LIMIT and count[gap] >= 2:
                current_L = gap
                gap += 1
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
            continue
        
        # Find best move to cover gap
        best_move = None
        min_cost = float('inf')
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            new_r = gap % p
            old_r = residues[i]
            if new_r == old_r:
                continue
            
            start_old = p if old_r == 0 else old_r
            if start_old > current_L:
                cost = 0
            else:
                pos = np.arange(start_old, current_L + 1, p)
                cost = np.sum(count[pos] == 2)
            
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
                if current_L > best_L:
                    best_L = current_L
                    best_residues = list(residues)
            else:
                count[start_new::p] -= 1
                count[start_old::p] += 1
                residues[i] = old_r
    
    return best_L, best_residues

def _pairwise_swaps(residues, deadline):
    """Try swapping residues between pairs of primes."""
    count = _build_count(residues)
    current_L = _find_L(count)
    best_L = current_L
    best_residues = list(residues)
    
    max_iters = 200
    for _ in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        if current_L >= _L_LIMIT:
            break
        
        i1 = random.randint(0, _NPRIMES - 1)
        i2 = random.randint(0, _NPRIMES - 1)
        if i1 == i2:
            continue
        
        p1, p2 = _PRIMES[i1], _PRIMES[i2]
        r1, r2 = residues[i1], residues[i2]
        
        start1_old = p1 if r1 == 0 else r1
        start2_old = p2 if r2 == 0 else r2
        
        count[start1_old::p1] -= 1
        count[start2_old::p2] -= 1
        count[start2_old::p1] += 1  # Wrong - need to use correct mod
        # Actually swap means: prime i1 gets r2 % p1, prime i2 gets r1 % p2
        # Let me redo this properly
        count[start2_old::p1] -= 1  # revert the wrong addition
        count[start1_old::p2] -= 1  # revert wrong
        
        new_r1 = r2 % p1
        new_r2 = r1 % p2
        
        start1_new = p1 if new_r1 == 0 else new_r1
        start2_new = p2 if new_r2 == 0 else new_r2
        
        count[start1_old::p1] -= 1
        count[start2_old::p2] -= 1
        count[start1_new::p1] += 1
        count[start2_new::p2] += 1
        
        new_L = _find_L(count)
        
        if new_L >= current_L:
            residues[i1] = new_r1
            residues[i2] = new_r2
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
        else:
            # Revert
            count[start1_new::p1] -= 1
            count[start2_new::p2] -= 1
            count[start1_old::p1] += 1
            count[start2_old::p2] += 1
    
    return best_L, best_residues

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    random.seed(42)
    
    best_L_overall = 0
    best_residues_overall = None
    
    # Multiple restarts with different strategies
    n_restarts = 5
    time_per_restart = (deadline - start_time - 10) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Greedy initialization with small primes prioritized
        residues = _greedy_init_small_primes()
        
        # SA phase
        sa_deadline = min(time.time() + time_per_restart * 0.5, deadline - 10)
        L_sa, residues_sa = _simulated_annealing(residues, sa_deadline)
        
        # Targeted local search
        ls_deadline = min(time.time() + time_per_restart * 0.3, deadline - 10)
        L_ls, residues_ls = _targeted_local_search(residues_sa, ls_deadline)
        
        # Pairwise swaps
        swap_deadline = min(time.time() + time_per_restart * 0.2, deadline - 10)
        L_swap, residues_swap = _pairwise_swaps(residues_ls, swap_deadline)
        
        if L_swap > best_L_overall:
            best_L_overall = L_swap
            best_residues_overall = residues_swap
    
    if best_residues_overall is None:
        best_residues_overall = [0] * _NPRIMES
    
    return [int(r) for r in best_residues_overall]
