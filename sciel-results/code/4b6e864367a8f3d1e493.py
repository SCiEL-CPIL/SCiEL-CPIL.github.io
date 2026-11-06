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

def _get_L(counts, max_check=100000):
    """Find the largest L such that counts[1..L] >= 2."""
    below_two = np.where(counts[1:max_check+1] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return int(below_two[0])

def _compute_counts(residues, primes, max_check=100000):
    """Compute coverage counts for all integers 1..max_check."""
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    return counts

def _greedy_construct(primes, order, max_check=100000):
    """
    Greedy set-cover heuristic.
    For each prime in 'order', pick residue to cover the first gap.
    """
    counts = np.zeros(max_check + 1, dtype=np.int32)
    residues = np.zeros(len(primes), dtype=np.int32)
    current_L = 0
    
    for p in order:
        if current_L < max_check:
            start_search = current_L + 1
            if start_search > max_check:
                g1 = max_check + 1
            else:
                segment = counts[start_search:]
                below_two = np.where(segment < 2)[0]
                if len(below_two) == 0:
                    g1 = max_check + 1
                else:
                    g1 = start_search + int(below_two[0])
        else:
            g1 = max_check + 1
            
        if g1 > max_check:
            r = 0
        else:
            r = g1 % p
        
        idx = primes.index(p)
        residues[idx] = r
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
            
        new_L = _get_L(counts, max_check)
        current_L = new_L
        
    return residues, current_L

def _fast_hill_climb(residues, primes, counts, current_L, max_check=100000, deadline=None):
    """
    Fast local search using incremental gap logic.
    Combines single-residue moves with strategic candidates.
    """
    n_primes = len(primes)
    improved = True
    passes = 0
    
    while improved and passes < 20:
        passes += 1
        improved = False
        
        for i in range(n_primes):
            if deadline and time.time() > deadline:
                return residues, counts, current_L
                
            p = primes[i]
            current_r = residues[i]
            
            # Generate candidates
            candidates = set()
            
            # Nearby residues
            for delta in range(-3, 4):
                candidates.add((current_r + delta) % p)
                
            # Strategic: try residues that cover the first gap
            if current_L < max_check:
                start_search = current_L + 1
                segment = counts[start_search:]
                below_two = np.where(segment < 2)[0]
                if len(below_two) > 0:
                    g1 = start_search + int(below_two[0])
                    target_r = g1 % p
                    candidates.add(target_r)
                    
                # Try to cover a few positions after current_L
                for offset in range(1, 5):
                    if current_L + offset <= max_check:
                        candidates.add((current_L + offset) % p)
            
            # Random candidates for larger primes
            if p > 20:
                for _ in range(5):
                    candidates.add(np.random.randint(0, p))
            
            # For small primes, try all residues
            if p <= 30:
                candidates = set(range(p))
            
            best_r = current_r
            best_L = current_L
            
            for r in candidates:
                if r == current_r:
                    continue
                
                # Apply change
                old_start = current_r if current_r > 0 else p
                if old_start <= max_check:
                    counts[old_start::p] -= 1
                
                new_start = r if r > 0 else p
                if new_start <= max_check:
                    counts[new_start::p] += 1
                
                new_L = _get_L(counts, max_check)
                
                if new_L > best_L:
                    best_L = new_L
                    best_r = r
                else:
                    # Revert
                    if new_start <= max_check:
                        counts[new_start::p] -= 1
                    if old_start <= max_check:
                        counts[old_start::p] += 1
                    
            # Apply best
            if best_r != current_r:
                # Re-apply best_r
                old_start = current_r if current_r > 0 else p
                if old_start <= max_check:
                    counts[old_start::p] -= 1
                new_start = best_r if best_r > 0 else p
                if new_start <= max_check:
                    counts[new_start::p] += 1
                residues[i] = best_r
                current_L = best_L
                improved = True
                
    return residues, counts, current_L

def _pairwise_moves(residues, primes, counts, current_L, max_check=100000, deadline=None):
    """
    Multi-residue moves to escape local optima.
    """
    n_primes = len(primes)
    
    for _ in range(100):
        if deadline and time.time() > deadline:
            break
            
        # Pick two random positions to modify
        i1, i2 = np.random.choice(n_primes, size=2, replace=False)
        p1, p2 = primes[i1], primes[i2]
        r1, r2 = residues[i1], residues[i2]
        
        # Try random new values for both
        for _ in range(10):
            if deadline and time.time() > deadline:
                break
                
            new_r1 = np.random.randint(0, p1)
            new_r2 = np.random.randint(0, p2)
            
            if new_r1 == r1 and new_r2 == r2:
                continue
            
            # Apply changes
            old_start1 = r1 if r1 > 0 else p1
            old_start2 = r2 if r2 > 0 else p2
            new_start1 = new_r1 if new_r1 > 0 else p1
            new_start2 = new_r2 if new_r2 > 0 else p2
            
            if old_start1 <= max_check:
                counts[old_start1::p1] -= 1
            if old_start2 <= max_check:
                counts[old_start2::p2] -= 1
            if new_start1 <= max_check:
                counts[new_start1::p1] += 1
            if new_start2 <= max_check:
                counts[new_start2::p2] += 1
            
            new_L = _get_L(counts, max_check)
            
            if new_L > current_L:
                residues[i1] = new_r1
                residues[i2] = new_r2
                current_L = new_L
                break
            else:
                # Revert
                if new_start1 <= max_check:
                    counts[new_start1::p1] -= 1
                if new_start2 <= max_check:
                    counts[new_start2::p2] -= 1
                if old_start1 <= max_check:
                    counts[old_start1::p1] += 1
                if old_start2 <= max_check:
                    counts[old_start2::p2] += 1
                    
    return residues, counts, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    max_check = 100000
    
    best_residues = None
    best_score = 0
    
    # Try multiple different greedy orders
    orders = [
        primes,  # Increasing
        primes[::-1],  # Decreasing
        list(np.random.permutation(primes)),
        sorted(primes, key=lambda p: -np.log(p)),  # Decreasing log
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
            
        residues, L = _greedy_construct(primes, order, max_check)
        
        # Refine with hill climbing
        counts = _compute_counts(residues, primes, max_check)
        
        residues, counts, L = _fast_hill_climb(residues, primes, counts, L, max_check, deadline)
        
        # Add pairwise moves for escape
        if time.time() < deadline - 5:
            residues, counts, L = _pairwise_moves(residues, primes, counts, L, max_check, deadline)
            
        if L > best_score:
            best_score = L
            best_residues = residues.copy()
            
    if best_residues is None:
        # Fallback
        best_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int32)
        
    return best_residues.tolist()
# EVOLVE-BLOCK-END
