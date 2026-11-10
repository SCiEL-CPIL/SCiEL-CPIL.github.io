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

def _build_count_np(residues, L_MAX):
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = int(residues[i])
        start = p if r == 0 else r
        if start <= L_MAX:
            count[start::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    if L_MAX < 1:
        return 0
    below = np.where(count[1:L_MAX+1] < 2)[0]
    if len(below) == 0:
        return L_MAX
    return int(below[0])

def _apply_move(count, residues, i, new_r):
    p = _PRIMES[i]
    old_r = int(residues[i])
    if old_r == new_r:
        return
    start_old = p if old_r == 0 else old_r
    start_new = p if new_r == 0 else new_r
    count[start_old::p] -= 1
    count[start_new::p] += 1
    residues[i] = new_r

def _revert_move(count, residues, i, old_r):
    p = _PRIMES[i]
    new_r = int(residues[i])
    start_new = p if new_r == 0 else new_r
    start_old = p if old_r == 0 else old_r
    count[start_new::p] -= 1
    count[start_old::p] += 1
    residues[i] = old_r

def _greedy_init(L_SMALL=500):
    residues = [0] * _NPRIMES
    count = np.zeros(L_SMALL + 1, dtype=np.int32)
    
    for i, p in enumerate(_PRIMES):
        if p > L_SMALL:
            break
            
        best_r = 0
        best_score = -1
        
        for r in range(p):
            start = p if r == 0 else r
            if start > L_SMALL:
                continue
                
            positions = np.arange(start, L_SMALL + 1, p)
            if len(positions) == 0:
                continue
                
            current_counts = count[positions]
            would_reach_2 = np.sum(current_counts == 1)
            stays_2 = np.sum(current_counts >= 2)
            
            # Heuristic: prioritize closing gaps (1->2 is good, 0->1 is bad)
            score = would_reach_2 * 10 - np.sum(current_counts == 0) * 2 + stays_2
            
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[i] = best_r
        start = p if best_r == 0 else best_r
        if start <= L_SMALL:
            count[start::p] += 1
    
    return residues

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    random.seed(42)
    
    # Phase 1: Greedy Initialization
    residues = _greedy_init(500)
    
    # Phase 2: Build initial state
    L_LIMIT = 10000
    count = _build_count_np(residues, L_LIMIT)
    current_L = _find_prefix_L_np(count, L_LIMIT)
    
    best_L = current_L
    best_residues = [int(r) for r in residues]
    
    stagnation_count = 0
    max_stagnation = 20
    no_improve_streak = 0
    
    # Phase 3: Main Loop
    while time.time() < deadline - 5:
        # Extend window if needed
        if current_L > L_LIMIT - 500:
            new_limit = min(L_LIMIT * 2, 100000)
            count = _build_count_np(residues, new_limit)
            L_LIMIT = new_limit
            current_L = _find_prefix_L_np(count, L_LIMIT)
        
        gap = current_L + 1
        
        if gap > L_LIMIT:
            new_limit = min(L_LIMIT * 2, 100000)
            count = _build_count_np(residues, new_limit)
            L_LIMIT = new_limit
            current_L = _find_prefix_L_np(count, L_LIMIT)
            continue
        
        # Check if gap is covered
        if count[gap] >= 2:
            while gap <= L_LIMIT and count[gap] >= 2:
                current_L = gap
                gap += 1
            if current_L > best_L:
                best_L = current_L
                best_residues = [int(r) for r in residues]
                no_improve_streak = 0
            stagnation_count = 0
            continue
        
        # Gap not covered. Find best move to fix it.
        best_move = None
        min_cost = float('inf')
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            old_r = int(residues[i])
            new_r = gap % p
            
            if old_r == new_r:
                continue
            
            # Calculate cost: number of x in [1, current_L] that drop from 2 to 1
            start_x = p if old_r == 0 else old_r
            if start_x > current_L:
                cost = 0
            else:
                positions = np.arange(start_x, current_L + 1, p)
                cost = int(np.sum(count[positions] == 2))
            
            if cost < min_cost:
                min_cost = cost
                best_move = (i, new_r)
                if cost == 0:
                    break
        
        improved = False
        if best_move is not None:
            i, new_r = best_move
            old_r = int(residues[i])
            
            _apply_move(count, residues, i, new_r)
            
            new_L = _find_prefix_L_np(count, L_LIMIT)
            
            if new_L > current_L:
                current_L = new_L
                improved = True
                stagnation_count = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = [int(r) for r in residues]
                    no_improve_streak = 0
            else:
                _revert_move(count, residues, i, old_r)
                stagnation_count += 1
        else:
            stagnation_count += 1
            
        if not improved:
            no_improve_streak += 1
        else:
            no_improve_streak = 0

        # Phase 4: Simulated Annealing if Stagnant
        if stagnation_count > max_stagnation:
            T = 10.0
            cooling = 0.95
            min_T = 1.0
            sa_start_time = time.time()
            
            while time.time() < sa_start_time + 3.0 and T > min_T:
                i = random.randint(0, _NPRIMES - 1)
                p = _PRIMES[i]
                old_r = int(residues[i])
                new_r = random.randint(0, p - 1)
                
                if new_r == old_r:
                    continue
                
                _apply_move(count, residues, i, new_r)
                
                new_L = _find_prefix_L_np(count, L_LIMIT)
                delta = new_L - current_L
                
                if delta > 0:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = [int(r) for r in residues]
                        no_improve_streak = 0
                        stagnation_count = 0
                else:
                    prob = 2.718281828 ** (delta / T)
                    if random.random() < prob:
                        current_L = new_L
                    else:
                        _revert_move(count, residues, i, old_r)
                
                T *= cooling
            
            stagnation_count = 0

    # Final Verification
    count_final = _build_count_np(best_residues, 100000)
    final_L = _find_prefix_L_np(count_final, 100000)
    
    return [int(r) for r in best_residues]
