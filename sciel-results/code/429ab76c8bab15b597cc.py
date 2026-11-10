# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

PRIMES = _primes_upto(1000)
NPRIMES = len(PRIMES)  # 168
WORK_L = 20000

def _score(residues, primes, cap=WORK_L):
    """Return the largest L such that every integer 1..L has count >= 2."""
    counts = np.zeros(cap + 1, dtype=np.int32)
    for p, r in zip(primes, residues):
        start = r
        if start < 1:
            start += ((1 - start + p - 1) // p) * p
        if start <= cap:
            counts[start::p] += 1
    below = np.where(counts[1:cap+1] < 2)[0]
    if len(below) == 0:
        return cap
    return int(below[0])

def _score_fast(residues, primes, cap=WORK_L):
    """Faster score using vectorized approach."""
    counts = np.zeros(cap + 1, dtype=np.int32)
    for p, r in zip(primes, residues):
        start = r
        if start < 1:
            start += ((1 - start + p - 1) // p) * p
        if start <= cap:
            counts[start::p] += 1
    below = np.where(counts[1:cap+1] < 2)[0]
    if len(below) == 0:
        return cap
    return int(below[0])

def _delta_score(residues, primes, old_r, new_r, p_idx, cap=WORK_L):
    """Compute score change when changing residue of prime p_idx from old_r to new_r."""
    p = primes[p_idx]
    # Remove old coverage
    old_start = old_r
    if old_start < 1:
        old_start += ((1 - old_start + p - 1) // p) * p
    # Add new coverage
    new_start = new_r
    if new_start < 1:
        new_start += ((1 - new_start + p - 1) // p) * p
    
    # We need to compute the full score to be safe, but we can optimize
    # by only checking the affected region
    # For now, just compute full score (it's fast enough with numpy)
    residues[p_idx] = new_r
    sc = _score_fast(residues, primes, cap)
    residues[p_idx] = old_r
    return sc

def _initial_structured():
    """Create a structured initial assignment that covers small integers densely."""
    residues = [0] * NPRIMES
    # Assign residues to create dense coverage in 1..200
    # Each integer 1..200 should be covered by at least 2 primes
    # Strategy: for each prime, assign it to cover a target integer
    targets = list(range(1, 201))
    random.shuffle(targets)
    for i, p in enumerate(PRIMES):
        # Cycle through targets
        t = targets[i % len(targets)]
        residues[i] = t % p
    return residues

def _greedy_repair(residues, primes, cap=WORK_L, max_iters=200):
    """Greedy repair: find uncovered integers and fix them."""
    best_sc = 0
    best_res = residues[:]
    
    for iteration in range(max_iters):
        sc = _score_fast(residues, primes, cap)
        if sc > best_sc:
            best_sc = sc
            best_res = residues[:]
        
        if sc >= cap:
            break
        
        # Find first gap
        check_L = min(sc + 500, cap)
        counts = np.zeros(check_L + 1, dtype=np.int32)
        for p, r in zip(primes, residues):
            start = r
            if start < 1:
                start += ((1 - start + p - 1) // p) * p
            if start <= check_L:
                counts[start::p] += 1
        below = np.where(counts[1:check_L+1] < 2)[0]
        if len(below) == 0:
            continue
        bad_x = int(below[0]) + 1
        
        # Try to fix by changing one residue
        fixed = False
        # Try all primes, preferring larger ones (less disruption)
        order = sorted(range(NPRIMES), key=lambda i: -primes[i])
        for idx in order:
            old_r = residues[idx]
            new_r = bad_x % primes[idx]
            if old_r == new_r:
                continue
            residues[idx] = new_r
            new_sc = _score_fast(residues, primes, cap)
            if new_sc >= sc:
                fixed = True
                break
            else:
                residues[idx] = old_r
        if not fixed:
            # Try smaller primes
            for idx in range(min(40, NPRIMES)):
                old_r = residues[idx]
                new_r = bad_x % primes[idx]
                if old_r == new_r:
                    continue
                residues[idx] = new_r
                new_sc = _score_fast(residues, primes, cap)
                if new_sc >= sc:
                    fixed = True
                    break
                else:
                    residues[idx] = old_r
        if not fixed:
            break
    
    return best_res, best_sc

def _simulated_annealing(residues, primes, deadline, initial_temp=1000.0, cooling=0.9995):
    """Simulated annealing with incremental evaluation."""
    best_res = residues[:]
    best_sc = _score_fast(residues, primes, WORK_L)
    current_res = residues[:]
    current_sc = best_sc
    temp = initial_temp
    
    while time.time() < deadline - 10:
        # Pick 1-2 random primes to modify
        k = random.randint(1, 2)
        indices = random.sample(range(NPRIMES), k)
        old_values = [current_res[i] for i in indices]
        
        for idx in indices:
            current_res[idx] = random.randint(0, primes[idx] - 1)
        
        new_sc = _score_fast(current_res, primes, WORK_L)
        
        if new_sc >= current_sc:
            current_sc = new_sc
            if new_sc > best_sc:
                best_sc = new_sc
                best_res = current_res[:]
        else:
            # Accept with probability exp(-(delta)/temp)
            delta = current_sc - new_sc
            if delta < 100:  # Only accept small degradations
                prob = np.exp(-delta / temp)
                if random.random() < prob:
                    current_sc = new_sc
            else:
                # Reject: revert
                for idx, old_v in zip(indices, old_values):
                    current_res[idx] = old_v
        
        temp *= cooling
        if temp < 1.0:
            temp = 1.0
    
    return best_res, best_sc

def construct():
    deadline = time.time() + 135  # Safety margin
    
    primes = PRIMES
    
    # Phase 1: Structured initialization + greedy repair
    best_res = None
    best_score = 0
    
    # Try multiple structured initializations
    for attempt in range(5):
        if time.time() > deadline - 30:
            break
        if attempt == 0:
            res = _initial_structured()
        else:
            res = [random.randint(0, p - 1) for p in primes]
        
        res, sc = _greedy_repair(res, primes, WORK_L, max_iters=100)
        if sc > best_score:
            best_score = sc
            best_res = res[:]
    
    # Phase 2: Simulated annealing from best
    if best_res is not None and time.time() < deadline - 20:
        sa_res, sa_score = _simulated_annealing(best_res, primes, deadline)
        if sa_score > best_score:
            best_score = sa_score
            best_res = sa_res
    
    # Phase 3: Final greedy repair pass
    if best_res is not None and time.time() < deadline - 5:
        final_res, final_score = _greedy_repair(best_res, primes, WORK_L, max_iters=100)
        if final_score > best_score:
            best_score = final_score
            best_res = final_res
    
    # Ensure all residues are Python integers
    if best_res is not None:
        return [int(r) for r in best_res]
    
    # Fallback
    return [((i + 2) // 2) % p for i, p in enumerate(primes)]
# EVOLVE-BLOCK-END
