import time
import numpy as np
import random

# EVOLVE-BLOCK-START

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
    """Build count array for x in [1, L_MAX] using NumPy."""
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        if start <= L_MAX:
            count[start::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    # Find first index where count < 2
    # count[0] is unused, we care about 1..L_MAX
    # np.where returns indices
    # We want the first index >= 1 where count < 2
    # If no such index, L = L_MAX
    bad_indices = np.where(count[1:L_MAX+1] < 2)[0]
    if len(bad_indices) == 0:
        return L_MAX
    first_bad = bad_indices[0] + 1
    return first_bad - 1

def _score(residues):
    """Compute the actual score (largest L with full double coverage)."""
    L_MAX = 100000
    count = _build_count_np(residues, L_MAX)
    return _find_prefix_L_np(count, L_MAX)

def _greedy_construct(deadline):
    """
    Greedy construction using NumPy for speed.
    """
    residues = [0] * _NPRIMES
    L_MAX = 2000
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    
    # Process primes in increasing order
    prime_order = list(range(_NPRIMES))
    
    current_L = 0
    for idx in prime_order:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        best_r = 0
        best_score = -1
        
        # Try a subset of residues to cover the gap
        gap = current_L + 1
        candidates = set()
        candidates.add(gap % p)
        candidates.add(0)
        candidates.add(1)
        candidates.add(2)
        # Add some random candidates
        for _ in range(5):
            candidates.add(random.randint(0, p-1))
            
        for r in candidates:
            if r >= p:
                continue
            # Simulate
            # We don't want to update the full array for every candidate in greedy
            # Just estimate gain: how many x in [1, L_MAX] with count[x] < 2 are covered by this r?
            start = p if r == 0 else r
            # Vectorized check
            if start > L_MAX:
                gain = 0
            else:
                # Count positions x in [start, L_MAX] step p where count[x] < 2
                # This is a bit slow if done for every candidate.
                # Approximation: count of (L_MAX - start) // p
                # Better: just pick the one that covers the gap.
                pass
            
            # Simplified heuristic: prioritize covering the gap
            score = 0
            if gap % p == r:
                score += 100
            # Prefer residues that add coverage to under-covered spots
            # This is expensive. Let's just use the gap coverage for now in greedy.
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        if start <= L_MAX:
            count[start::p] += 1
        
        # Update current_L
        # Find first gap
        bad_indices = np.where(count[1:L_MAX+1] < 2)[0]
        if len(bad_indices) == 0:
            current_L = L_MAX
        else:
            current_L = bad_indices[0]

    return residues, current_L

def _local_search_np(residues, deadline, L_MAX=2000, use_random_restarts=True):
    """
    Local search with NumPy acceleration and optional random restarts.
    """
    residues = list(residues)
    count = _build_count_np(residues, L_MAX)
    current_L = _find_prefix_L_np(count, L_MAX)
    
    best_residues = list(residues)
    best_L = current_L
    
    stuck_count = 0
    max_stuck = 50
    
    while time.time() < deadline:
        if current_L >= L_MAX:
            break
            
        if stuck_count > max_stuck:
            if use_random_restarts:
                # Kick: randomly change 2-3 residues
                for _ in range(3):
                    idx = random.randint(0, _NPRIMES - 1)
                    p = _PRIMES[idx]
                    old_r = residues[idx]
                    new_r = random.randint(0, p - 1)
                    if old_r != new_r:
                        old_start = p if old_r == 0 else old_r
                        new_start = p if new_r == 0 else new_r
                        if old_start <= L_MAX:
                            count[old_start::p] -= 1
                        if new_start <= L_MAX:
                            count[new_start::p] += 1
                        residues[idx] = new_r
                current_L = _find_prefix_L_np(count, L_MAX)
                stuck_count = 0
                continue
            else:
                break
        
        improved = False
        gap = current_L + 1
        
        # Try 1-flips
        # Prioritize primes that don't cover the gap
        # Or just try all small primes
        small_prime_indices = [i for i in range(_NPRIMES) if _PRIMES[i] <= 100]
        
        best_gain = 0
        best_move = None
        
        for i in small_prime_indices:
            if time.time() > deadline:
                break
            p = _PRIMES[i]
            current_r = residues[i]
            
            # Candidate residues: gap % p, and a few others
            candidates = {gap % p}
            # Add current_r's neighbors? No, just gap and 0,1
            candidates.add(0)
            candidates.add(1)
            
            for new_r in candidates:
                if new_r == current_r:
                    continue
                if new_r >= p:
                    continue
                
                old_start = p if current_r == 0 else current_r
                new_start = p if new_r == 0 else new_r
                
                # Apply
                if old_start <= L_MAX:
                    count[old_start::p] -= 1
                if new_start <= L_MAX:
                    count[new_start::p] += 1
                
                new_L = _find_prefix_L_np(count, L_MAX)
                
                # Undo
                if new_start <= L_MAX:
                    count[new_start::p] -= 1
                if old_start <= L_MAX:
                    count[old_start::p] += 1
                
                gain = new_L - current_L
                if gain > best_gain:
                    best_gain = gain
                    best_move = (i, new_r, new_L)
        
        # If no 1-flip improved, try 2-flips on a small subset
        if best_gain == 0:
            # Pick 2 small primes
            if len(small_prime_indices) >= 2:
                i1, i2 = random.sample(small_prime_indices, 2)
                p1, p2 = _PRIMES[i1], _PRIMES[i2]
                r1, r2 = residues[i1], residues[i2]
                
                # Try a few combinations
                for nr1 in {gap % p1, 0, 1}:
                    for nr2 in {gap % p2, 0, 1}:
                        if (nr1 == r1 and nr2 == r2):
                            continue
                        if nr1 >= p1 or nr2 >= p2:
                            continue
                        
                        os1 = p1 if r1 == 0 else r1
                        ns1 = p1 if nr1 == 0 else nr1
                        os2 = p2 if r2 == 0 else r2
                        ns2 = p2 if nr2 == 0 else nr2
                        
                        # Apply
                        if os1 <= L_MAX: count[os1::p1] -= 1
                        if ns1 <= L_MAX: count[ns1::p1] += 1
                        if os2 <= L_MAX: count[os2::p2] -= 1
                        if ns2 <= L_MAX: count[ns2::p2] += 1
                        
                        new_L = _find_prefix_L_np(count, L_MAX)
                        
                        # Undo
                        if ns1 <= L_MAX: count[ns1::p1] -= 1
                        if os1 <= L_MAX: count[os1::p1] += 1
                        if ns2 <= L_MAX: count[ns2::p2] -= 1
                        if os2 <= L_MAX: count[os2::p2] += 1
                        
                        gain = new_L - current_L
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (i1, nr1, i2, nr2, new_L)
        
        if best_gain > 0:
            if len(best_move) == 3:
                i, new_r, new_L = best_move
                p = _PRIMES[i]
                old_r = residues[i]
                old_start = p if old_r == 0 else old_r
                new_start = p if new_r == 0 else new_r
                
                if old_start <= L_MAX: count[old_start::p] -= 1
                if new_start <= L_MAX: count[new_start::p] += 1
                
                residues[i] = new_r
                current_L = new_L
            else:
                i1, nr1, i2, nr2, new_L = best_move
                p1, p2 = _PRIMES[i1], _PRIMES[i2]
                old_r1, old_r2 = residues[i1], residues[i2]
                
                os1 = p1 if old_r1 == 0 else old_r1
                ns1 = p1 if nr1 == 0 else nr1
                os2 = p2 if old_r2 == 0 else old_r2
                ns2 = p2 if nr2 == 0 else nr2
                
                if os1 <= L_MAX: count[os1::p1] -= 1
                if ns1 <= L_MAX: count[ns1::p1] += 1
                if os2 <= L_MAX: count[os2::p2] -= 1
                if ns2 <= L_MAX: count[ns2::p2] += 1
                
                residues[i1] = nr1
                residues[i2] = nr2
                current_L = new_L
            
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
            
            stuck_count = 0
        else:
            stuck_count += 1
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Phase 1: Greedy
    residues, L = _greedy_construct(deadline - 100)
    best_residues = list(residues)
    best_L = L
    
    # Phase 2: Local Search with NumPy
    if time.time() < deadline - 10:
        residues, L = _local_search_np(residues, deadline - 10, L_MAX=2000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 3: Final Polish with larger L_MAX if time permits
    if time.time() < deadline - 5:
        # Re-evaluate with larger L_MAX to see if we can push further
        # But local search is bounded by L_MAX.
        # Let's try one more local search pass with L_MAX=3000
        residues, L = _local_search_np(best_residues, deadline - 5, L_MAX=3000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Final Verification
    # The evaluator will compute the score, but we can do a quick check
    # if time permits, though it's expensive.
    # We trust the local search L.
    
    return best_residues

# EVOLVE-BLOCK-END
