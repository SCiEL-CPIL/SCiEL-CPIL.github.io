"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _score(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return int(below_two[0])

def _greedy_cover(primes, max_check=100000):
    """
    Greedy set-cover heuristic: process primes in increasing order, selecting
    for each prime the residue that maximizes the number of integers in [1, max_check]
    that are covered by exactly one congruence (i.e., converting single-covered to double-covered).
    """
    counts = np.zeros(max_check + 1, dtype=np.int32)
    residues = np.zeros(len(primes), dtype=np.int32)
    
    for idx, p in enumerate(primes):
        best_r = 0
        best_gain = -1
        
        # For small primes, try all residues; for large, sample
        if p <= 200:
            candidates = list(range(p))
        else:
            candidates = list(range(0, p, max(1, p // 50)))
            candidates.append(0)
        
        for r in candidates:
            # Compute gain: how many positions in [1, max_check] go from count==1 to count==2
            start = r if r > 0 else p
            if start > max_check:
                gain = 0
            else:
                # Positions hit by this congruence
                positions = np.arange(start, max_check + 1, p)
                # Gain = number of positions where count == 1 (will become 2)
                gain = np.sum(counts[positions] == 1)
                # Penalty = number of positions where count >= 2 (will become >=3, but that's fine)
                # Actually we only care about extending the prefix, so focus on count==1
        
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        residues[idx] = best_r
        # Update counts
        start = best_r if best_r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    
    return residues

def _hill_climb(residues, primes, max_check=100000, deadline=None):
    """Local search: hill-climbing on individual residues."""
    residues = residues.copy()
    best_score = _score(residues, primes, max_check)
    n_primes = len(primes)
    max_passes = 20
    
    for iteration in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        for i in range(n_primes):
            if deadline is not None and time.time() > deadline:
                break
            
            p = primes[i]
            current_r = residues[i]
            best_r = current_r
            best_local_score = best_score
            
            if p <= 100:
                candidates = list(range(p))
            else:
                candidates = [current_r]
                if p <= 500:
                    candidates.extend(np.random.randint(0, p, size=min(20, p)).tolist())
                else:
                    candidates.extend(np.random.randint(0, p, size=10).tolist())
            
            for r in candidates:
                if r == current_r:
                    continue
                
                old_r = residues[i]
                residues[i] = r
                new_score = _score(residues, primes, max_check)
                residues[i] = old_r
                
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_r = r
            
            if best_r != current_r:
                residues[i] = best_r
                best_score = best_local_score
                improved = True
        
        if not improved:
            break
    
    return residues, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    max_check = 100000
    
    # Phase 1: Greedy set-cover heuristic
    greedy_residues = _greedy_cover(primes, max_check)
    greedy_score = _score(greedy_residues, primes, max_check)
    
    # Phase 2: Hill-climbing refinement
    refined_residues, refined_score = _hill_climb(greedy_residues, primes, max_check, deadline)
    
    # Also try hill-climbing from the original seed for comparison
    seed_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int32)
    seed_refined, seed_score = _hill_climb(seed_residues, primes, max_check, deadline)
    
    # Pick the best result
    if refined_score >= seed_score:
        return refined_residues.tolist()
    else:
        return seed_refined.tolist()
# EVOLVE-BLOCK-END
