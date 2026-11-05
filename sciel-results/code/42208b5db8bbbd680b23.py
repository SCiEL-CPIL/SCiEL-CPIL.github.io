"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

N = 5000
DEADLINE = 320  # seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


_ALL_PRIMES = _primes(N + 1)


def _is_admissible(points):
    """Check if points form an admissible tuple."""
    m = len(points)
    for p in _ALL_PRIMES:
        if p > m:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def _is_admissible_with_candidate(points, candidate):
    """Check if adding candidate to points maintains admissibility."""
    m_new = len(points) + 1
    for p in _ALL_PRIMES:
        if p > m_new:
            break
        r = candidate % p
        residues = set(v % p for v in points)
        if len(residues) == p - 1 and r not in residues:
            # Adding this would make all p residues present
            return False
        elif len(residues) == p:
            # Already not admissible
            return False
    return True


def _repair(points):
    changed = True
    while changed:
        changed = False
        m = len(points)
        for p in _ALL_PRIMES:
            if p > m:
                break
            counts = {}
            for v in points:
                r = v % p
                counts[r] = counts.get(r, 0) + 1
            if len(counts) == p:
                min_class = min(counts, key=lambda r: counts[r])
                points = [v for v in points if v % p != min_class]
                changed = True
                break
    return points


def _greedy_construct(deadline):
    """Greedy construction starting from a small seed."""
    t0 = time.time()
    
    # Start with primes > sqrt(N) plus 1
    initial = [1] + [p for p in _ALL_PRIMES if p * p > N]
    initial = _repair(initial)
    
    point_set = set(initial)
    
    # Score candidates by how many small-prime residue classes they cover
    small_primes = [p for p in _ALL_PRIMES if p <= 300]
    scores = [0] * (N + 1)
    for p in small_primes:
        for multiple in range(0, N + 1, p):
            scores[multiple] += 1
    
    candidates = [v for v in range(N + 1) if v not in point_set]
    candidates.sort(key=lambda x: (-scores[x], x))
    
    # Track residue counts for primes up to a limit
    max_track_prime = 1500
    track_primes = [p for p in _ALL_PRIMES if p <= max_track_prime]
    
    counts = {}
    for p in track_primes:
        counts[p] = [0] * p
    for v in initial:
        for p in track_primes:
            counts[p][v % p] += 1
    
    for v in candidates:
        if time.time() - t0 > deadline:
            break
        
        m_new = len(initial) + 1
        ok = True
        for p in track_primes:
            if p > m_new:
                break
            r = v % p
            # After adding v, class r gets one more count
            # We need at least one class with count 0 after adding
            # Class i has count 0 after adding iff counts[p][i] == 0 and i != r
            # So no violation iff there exists i != r with counts[p][i] == 0
            zero_count = 0
            zero_at_r = False
            for i in range(p):
                if counts[p][i] == 0:
                    zero_count += 1
                    if i == r:
                        zero_at_r = True
            
            if zero_count >= 2:
                continue  # No violation
            elif zero_count == 1 and not zero_at_r:
                continue  # No violation
            else:
                ok = False
                break
        
        if ok:
            initial.append(v)
            point_set.add(v)
            for p in track_primes:
                counts[p][v % p] += 1
    
    return _repair(initial)


def _try_insert(points, deadline):
    """Try to insert more elements into the tuple."""
    t0 = time.time()
    point_set = set(points)
    candidates = [v for v in range(N + 1) if v not in point_set]
    
    # Score by coverage of small prime residue classes
    small_primes = [p for p in _ALL_PRIMES if p <= 200]
    scores = [0] * (N + 1)
    for p in small_primes:
        for multiple in range(0, N + 1, p):
            scores[multiple] += 1
    
    candidates.sort(key=lambda x: (-scores[x], x))
    
    max_track_prime = 2000
    track_primes = [p for p in _ALL_PRIMES if p <= max_track_prime]
    
    counts = {}
    for p in track_primes:
        counts[p] = [0] * p
    for v in points:
        for p in track_primes:
            counts[p][v % p] += 1
    
    for v in candidates:
        if time.time() - t0 > deadline:
            break
        
        m_new = len(points) + 1
        ok = True
        for p in track_primes:
            if p > m_new:
                break
            r = v % p
            zero_count = 0
            zero_at_r = False
            for i in range(p):
                if counts[p][i] == 0:
                    zero_count += 1
                    if i == r:
                        zero_at_r = True
            
            if zero_count >= 2:
                continue
            elif zero_count == 1 and not zero_at_r:
                continue
            else:
                ok = False
                break
        
        if ok:
            points.append(v)
            point_set.add(v)
            for p in track_primes:
                counts[p][v % p] += 1
    
    return _repair(points)


def _randomized_local_search(points, deadline):
    """Randomized multi-start local search: try inserting and removing."""
    t0 = time.time()
    best = list(points)
    best_len = len(best)
    point_set = set(best)
    
    # Precompute scores
    small_primes = [p for p in _ALL_PRIMES if p <= 150]
    scores = [0] * (N + 1)
    for p in small_primes:
        for multiple in range(0, N + 1, p):
            scores[multiple] += 1
    
    # Multiple restarts with random perturbations
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() - t0 > deadline:
            break
        
        # Start from current best with some random removals
        current = list(best)
        if restart > 0 and len(current) > 10:
            # Remove some random elements
            n_remove = min(5, len(current) // 10)
            to_remove = set(random.sample(current, n_remove))
            current = [v for v in current if v not in to_remove]
            point_set = set(current)
        
        # Try to add elements back
        candidates = [v for v in range(N + 1) if v not in point_set]
        candidates.sort(key=lambda x: (-scores[x], x))
        
        max_track_prime = 1500
        track_primes = [p for p in _ALL_PRIMES if p <= max_track_prime]
        
        counts = {}
        for p in track_primes:
            counts[p] = [0] * p
        for v in current:
            for p in track_primes:
                counts[p][v % p] += 1
        
        for v in candidates:
            if time.time() - t0 > deadline:
                break
            
            m_new = len(current) + 1
            ok = True
            for p in track_primes:
                if p > m_new:
                    break
                r = v % p
                zero_count = 0
                zero_at_r = False
                for i in range(p):
                    if counts[p][i] == 0:
                        zero_count += 1
                        if i == r:
                            zero_at_r = True
                
                if zero_count >= 2:
                    continue
                elif zero_count == 1 and not zero_at_r:
                    continue
                else:
                    ok = False
                    break
            
            if ok:
                current.append(v)
                point_set.add(v)
                for p in track_primes:
                    counts[p][v % p] += 1
        
        current = _repair(current)
        if len(current) > best_len:
            best = current
            best_len = len(current)
    
    return best


def construct():
    t0 = time.time()
    
    # Phase 1: Greedy construction
    points = _greedy_construct(DEADLINE - 200)
    
    # Phase 2: Try to add more
    points = _try_insert(points, DEADLINE - 100)
    
    # Phase 3: Randomized local search
    points = _randomized_local_search(points, DEADLINE - 30)
    
    return points
# EVOLVE-BLOCK-END
