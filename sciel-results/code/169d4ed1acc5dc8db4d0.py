"""Optimized doubly-covered prefix: Constraint-satisfaction with targeted repair."""
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
_PRIMES_ARR = np.array(_PRIMES)

def _build_count_np(residues, L_MAX):
    """Build count array for x in [1, L_MAX] using NumPy."""
    count = np.zeros(L_MAX + 1, dtype=np.int8)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        if start <= L_MAX:
            count[start:L_MAX+1:p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    valid = count[1:L_MAX+1] >= 2
    if not np.any(~valid):
        return L_MAX
    first_invalid = np.where(~valid)[0][0]
    return first_invalid

def _score(residues):
    """Compute the actual score (largest L with full double coverage)."""
    L_MAX = 100000
    count = _build_count_np(residues, L_MAX)
    return _find_prefix_L_np(count, L_MAX)

def _greedy_construct(deadline):
    """
    Greedy construction: build residues to maximize double coverage.
    Strategy: for each prime, choose residue to help cover the current gap.
    """
    residues = [0] * _NPRIMES
    L_MAX = 5000
    count = [0] * (L_MAX + 1)
    
    # Process primes in order of increasing size (small primes give more coverage)
    prime_order = sorted(range(_NPRIMES), key=lambda i: _PRIMES[i])
    
    current_L = 0
    for idx in prime_order:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        best_r = 0
        best_score = -1
        
        # The gap is at current_L + 1
        gap = current_L + 1
        candidates = set()
        candidates.add(gap % p)
        candidates.add(0)
        candidates.add(1)
        candidates.add(2)
        for dx in range(-5, 6):
            x = gap + dx
            if x >= 1:
                candidates.add(x % p)
        
        for r in candidates:
            if r >= p:
                continue
            start = p if r == 0 else r
            coverage_gain = 0
            for x in range(start, L_MAX + 1, p):
                if count[x] <= 1:
                    coverage_gain += 1
                elif count[x] == 2:
                    coverage_gain -= 0.5
            
            if gap % p == r:
                coverage_gain += 100
            
            if coverage_gain > best_score:
                best_score = coverage_gain
                best_r = r
        
        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
        
        while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
            current_L += 1
    
    return residues, current_L

def _targeted_repair(residues, count, L_MAX, deadline):
    """
    Targeted repair: identify the first uncovered point and fix it by
    reassigning primes that can cover it. This is a constraint-satisfaction
    approach that directly addresses bottlenecks rather than random moves.
    """
    residues = list(residues)
    count = count.copy()
    
    improved = True
    repair_count = 0
    max_repairs = 50
    
    while improved and time.time() < deadline and repair_count < max_repairs:
        improved = False
        repair_count += 1
        
        # Find the first point with count < 2
        current_L = _find_prefix_L_np(count, L_MAX)
        if current_L >= L_MAX:
            break
        
        gap = current_L + 1  # The first uncovered point
        
        # Find which primes currently cover gap
        covering_primes = []
        for i, p in enumerate(_PRIMES):
            if gap % p == residues[i]:
                covering_primes.append(i)
        
        # We need at least 2 primes to cover gap
        # If we have < 2, we need to reassign some primes to cover gap
        
        if len(covering_primes) < 2:
            # Find candidate primes to reassign to cover gap
            # Priority: primes that don't currently cover gap, and where
            # reassigning doesn't break other critical coverage
            
            candidates = []
            for i, p in enumerate(_PRIMES):
                if i in covering_primes:
                    continue
                
                new_r = gap % p
                old_r = residues[i]
                
                if new_r == old_r:
                    continue
                
                # Estimate the impact of reassigning prime i
                old_start = p if old_r == 0 else old_r
                new_start = p if new_r == 0 else new_r
                
                # Count how many points in [1, L_MAX] would lose coverage
                loss = 0
                for x in range(old_start, L_MAX + 1, p):
                    if count[x] == 1:
                        loss += 1
                
                # Count how many points would gain coverage
                gain = 0
                for x in range(new_start, L_MAX + 1, p):
                    if count[x] == 0:
                        gain += 1
                
                # Net impact
                net = gain - loss * 0.5
                
                # Bonus if this helps cover points near the gap
                for dx in range(-10, 11):
                    x = gap + dx
                    if 1 <= x <= L_MAX and count[x] == 1 and x % p == new_r:
                        net += 2
                
                candidates.append((net, i, new_r))
            
            # Sort by net impact and try the best candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            applied = False
            for net, i, new_r in candidates[:10]:  # Try top 10 candidates
                if time.time() > deadline:
                    break
                
                p = _PRIMES[i]
                old_r = residues[i]
                old_start = p if old_r == 0 else old_r
                new_start = p if new_r == 0 else new_r
                
                # Apply the change
                count[old_start:L_MAX+1:p] -= 1
                count[new_start:L_MAX+1:p] += 1
                residues[i] = new_r
                
                # Check if this improved the prefix
                new_L = _find_prefix_L_np(count, L_MAX)
                if new_L > current_L:
                    improved = True
                    applied = True
                    break
                else:
                    # Revert
                    count[new_start:L_MAX+1:p] -= 1
                    count[old_start:L_MAX+1:p] += 1
                    residues[i] = old_r
            
            if not applied:
                # Try a different strategy: look for points just after gap
                # that have count == 1 and try to boost them
                for offset in range(1, 20):
                    x = gap + offset
                    if x > L_MAX:
                        break
                    
                    if count[x] == 1:
                        # Find which prime covers x
                        covering = []
                        for i, p in enumerate(_PRIMES):
                            if x % p == residues[i]:
                                covering.append(i)
                        
                        if len(covering) == 1:
                            # Find a prime to also cover x
                            for i, p in enumerate(_PRIMES):
                                if i in covering:
                                    continue
                                
                                new_r = x % p
                                old_r = residues[i]
                                
                                if new_r == old_r:
                                    continue
                                
                                old_start = p if old_r == 0 else old_r
                                new_start = p if new_r == 0 else new_r
                                
                                # Check impact
                                loss = 0
                                for xx in range(old_start, L_MAX + 1, p):
                                    if count[xx] == 1:
                                        loss += 1
                                
                                if loss <= 2:  # Acceptable loss
                                    count[old_start:L_MAX+1:p] -= 1
                                    count[new_start:L_MAX+1:p] += 1
                                    residues[i] = new_r
                                    improved = True
                                    applied = True
                                    break
                                else:
                                    # Revert (didn't apply)
                                    pass
                            
                            if applied:
                                break
                
                if not applied:
                    break
    
    return residues, count, repair_count

def _hybrid_optimize(residues, deadline):
    """
    Hybrid optimization: alternate between targeted repair and simulated annealing.
    Targeted repair fixes obvious bottlenecks, SA explores the space.
    """
    residues = list(residues)
    L_MAX_SEARCH = 10000
    count = _build_count_np(residues, L_MAX_SEARCH)
    
    best_residues = list(residues)
    best_L = _find_prefix_L_np(count, L_MAX_SEARCH)
    
    T = 50.0
    T_min = 0.5
    cooling_rate = 0.998
    
    start_time = time.time()
    phase = 0
    
    while time.time() < deadline and T > T_min:
        phase += 1
        
        # Alternate between repair and SA
        if phase % 5 == 0:
            # Do targeted repair
            residues, count, _ = _targeted_repair(residues, count, L_MAX_SEARCH, deadline - 5)
            current_L = _find_prefix_L_np(count, L_MAX_SEARCH)
            
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
        else:
            # Do SA move
            idx = random.randint(0, _NPRIMES - 1)
            p = _PRIMES[idx]
            old_r = residues[idx]
            
            if p == 2:
                new_r = 1 - old_r
            else:
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
            
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r
            
            count[old_start:L_MAX_SEARCH+1:p] -= 1
            count[new_start:L_MAX_SEARCH+1:p] += 1
            
            new_L = _find_prefix_L_np(count, L_MAX_SEARCH)
            
            # Simple acceptance: accept if better or with probability exp(delta/T)
            if new_L >= best_L:
                if new_L > best_L:
                    best_L = new_L
                    best_residues = list(residues)
            else:
                # Metropolis
                if random.random() < np.exp((new_L - best_L) / T):
                    pass  # Keep the change
                else:
                    # Revert
                    count[new_start:L_MAX_SEARCH+1:p] -= 1
                    count[old_start:L_MAX_SEARCH+1:p] += 1
                    residues[idx] = old_r
            
            residues[idx] = new_r
        
        T *= cooling_rate
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_residues = None
    best_score = 0
    
    # 1. Greedy Initialization
    residues, current_L = _greedy_construct(deadline - 30)
    
    # 2. Hybrid Optimization (targeted repair + SA)
    residues, hybrid_L = _hybrid_optimize(residues, deadline - 10)
    
    # 3. Final Score Calculation
    if time.time() < deadline - 5:
        actual_score = _score(residues)
        if actual_score > best_score:
            best_score = actual_score
            best_residues = list(residues)
    
    # 4. If time remains, try another restart with random init + hybrid
    if time.time() < deadline - 30:
        random.seed(456)
        residues_rand = [random.randint(0, _PRIMES[i] - 1) for i in range(_NPRIMES)]
        residues_rand, hybrid_L_rand = _hybrid_optimize(residues_rand, deadline - 10)
        if time.time() < deadline - 5:
            actual_score_rand = _score(residues_rand)
            if actual_score_rand > best_score:
                best_score = actual_score_rand
                best_residues = list(residues_rand)

    if best_residues is None:
        # Fallback: simple seed
        best_residues = [(i + 1) % p for i, p in enumerate(_PRIMES)]
    
    return best_residues
