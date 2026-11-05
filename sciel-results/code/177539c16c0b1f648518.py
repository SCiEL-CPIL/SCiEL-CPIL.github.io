"""Optimized solver for the covering problem."""
import time
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _score(residues, primes, max_L=100000):
    """
    Compute the largest L such that every integer 1..L is covered at least twice.
    Uses efficient counting.
    """
    n = len(primes)
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for i in range(n):
        r = residues[i]
        p = primes[i]
        if r == 0:
            r = p
        if r <= max_L:
            positions = np.arange(r, max_L + 1, p)
            counts[positions] += 1
    
    deficient = np.where(counts[1:] < 2)[0]
    if len(deficient) == 0:
        return max_L
    return deficient[0]

def _score_fast(residues, primes, max_L=5000):
    """Faster scoring for search - uses early termination."""
    n = len(primes)
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for i in range(n):
        r = residues[i]
        p = primes[i]
        if r == 0:
            r = p
        if r <= max_L:
            positions = np.arange(r, max_L + 1, p)
            counts[positions] += 1
    
    # Find first position where count < 2
    for pos in range(1, max_L + 1):
        if counts[pos] < 2:
            return pos - 1
    return max_L

def _try_improve(residues, primes, idx, max_L_check, deadline, current_score):
    """Try to improve residue at position idx. Returns (new_score, new_residue, improved)."""
    p = primes[idx]
    old_r = residues[idx]
    
    # Try residues that cover the most deficient positions
    # First, identify where coverage is thin
    counts = np.zeros(max_L_check + 1, dtype=np.int32)
    for i in range(len(primes)):
        r = residues[i]
        pi = primes[i]
        if r == 0:
            r = pi
        if r <= max_L_check:
            positions = np.arange(r, max_L_check + 1, pi)
            counts[positions] += 1
    
    # Find positions with count < 2 (deficient)
    deficient_positions = []
    for pos in range(1, max_L_check + 1):
        if counts[pos] < 2:
            deficient_positions.append(pos)
        else:
            break
    
    if not deficient_positions:
        return current_score, old_r, False
    
    # Try residues that cover deficient positions
    tried = set()
    best_new_score = current_score
    best_new_r = old_r
    
    for pos in deficient_positions[:20]:  # Try first 20 deficient positions
        if time.time() > deadline:
            break
        
        # For this position, what residue would cover it?
        target_r = pos % p
        if target_r == 0:
            target_r = p  # But we need 0 <= r < p, so 0
            target_r = 0
        
        if target_r not in tried:
            tried.add(target_r)
            residues[idx] = target_r
            new_score = _score_fast(residues, primes, max_L_check)
            if new_score > best_new_score:
                best_new_score = new_score
                best_new_r = target_r
    
    # Also try a few random residues
    rng = np.random.RandomState(hash(idx) % 2**32)
    for _ in range(3):
        if time.time() > deadline:
            break
        new_r = rng.randint(0, p - 1)
        if new_r not in tried:
            tried.add(new_r)
            residues[idx] = new_r
            new_score = _score_fast(residues, primes, max_L_check)
            if new_score > best_new_score:
                best_new_score = new_score
                best_new_r = new_r
    
    residues[idx] = best_new_r
    return best_new_score, best_new_r, best_new_score > current_score

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    primes = _get_primes(1000)
    n = len(primes)
    
    # Try multiple initial heuristics
    initial_candidates = []
    
    # Heuristic 1: Original seed
    r1 = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    initial_candidates.append(r1)
    
    # Heuristic 2: r_i = i % p
    r2 = [i % p for i, p in enumerate(primes)]
    initial_candidates.append(r2)
    
    # Heuristic 3: r_i = (i * 3 + 1) % p
    r3 = [(i * 3 + 1) % p for i, p in enumerate(primes)]
    initial_candidates.append(r3)
    
    # Heuristic 4: r_i = (i * 7 + 2) % p
    r4 = [(i * 7 + 2) % p for i, p in enumerate(primes)]
    initial_candidates.append(r4)
    
    # Heuristic 5: r_i = p - 1 - (i % p)
    r5 = [(p - 1 - (i % p)) % p for i, p in enumerate(primes)]
    initial_candidates.append(r5)
    
    # Evaluate all initial candidates
    max_L_check = 5000
    best_residues = None
    best_score = -1
    
    for candidate in initial_candidates:
        score = _score_fast(candidate, primes, max_L_check)
        if score > best_score:
            best_score = score
            best_residues = candidate[:]
    
    # Local search: iteratively improve
    max_iterations = 200
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        # Focus on small primes first (more impact)
        indices = sorted(range(n), key=lambda i: primes[i])
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            p = primes[idx]
            # Skip very large primes if we're running out of time
            if p > 500 and time.time() > deadline - 20:
                continue
            
            new_score, new_r, was_improved = _try_improve(
                best_residues, primes, idx, max_L_check, deadline, best_score
            )
            
            if was_improved:
                best_score = new_score
                improved = True
        
        if not improved:
            # Try random restarts
            if time.time() > deadline - 30:
                break
            
            trial = best_residues.copy()
            rng = np.random.RandomState(iteration)
            for _ in range(5):
                idx = rng.randint(0, n - 1)
                p = primes[idx]
                trial[idx] = rng.randint(0, p - 1)
            
            trial_score = _score_fast(trial, primes, max_L_check)
            if trial_score > best_score:
                best_residues = trial
                best_score = trial_score
    
    # Final evaluation
    final_score = _score(best_residues, primes, 100000)
    return best_residues
