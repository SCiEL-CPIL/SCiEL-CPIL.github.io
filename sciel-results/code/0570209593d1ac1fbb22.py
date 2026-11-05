"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

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


def _greedy_with_tracking():
    sqrt_N = math.isqrt(N)
    initial = [1] + [p for p in _ALL_PRIMES if p * p > N]
    initial = _repair(initial)
    
    small_primes = [p for p in _ALL_PRIMES if p <= 500]
    scores = [0] * (N + 1)
    for p in small_primes:
        for multiple in range(0, N + 1, p):
            scores[multiple] += 1
    
    point_set = set(initial)
    candidates = [v for v in range(N + 1) if v not in point_set]
    candidates.sort(key=lambda x: (-scores[x], x))
    
    max_track_prime = 2000
    track_primes = [p for p in _ALL_PRIMES if p <= max_track_prime]
    
    counts = {}
    for p in track_primes:
        counts[p] = [0] * p
    for v in initial:
        for p in track_primes:
            counts[p][v % p] += 1
    
    t0 = time.time()
    for v in candidates:
        if time.time() - t0 > 100:
            break
        
        m_new = len(initial) + 1
        ok = True
        for p in track_primes:
            if p > m_new:
                break
            r = v % p
            # Check if all classes other than r have count > 0
            # i.e., min of counts[p][i] for i != r is > 0
            # This is equivalent to: the global min of counts[p] is > 0, OR
            # the global min is 0 and it's not at index r
            min_val = min(counts[p])
            if min_val > 0:
                ok = False
                break
            else:
                # min_val == 0, check if there's a zero at index r
                # If counts[p][r] == 0, then after adding, it becomes 1,
                # and we need to check if all other zeros are gone
                # Actually, we need: after adding, is there any class with count 0?
                # After adding, counts[p][r] += 1
                # So class i has count 0 after adding iff counts[p][i] == 0 and i != r
                # So no violation iff there exists i != r with counts[p][i] == 0
                # i.e., the number of zeros in counts[p] is >= 2, or
                # the number of zeros is 1 and it's at index r (then after adding, no zeros)
                # Wait, we want NO violation, so we want at least one class with count 0 after adding
                # After adding, class i has count 0 iff counts[p][i] == 0 and i != r
                # So no violation iff there exists i != r with counts[p][i] == 0
                # This is: (number of zeros in counts[p]) >= 2, OR
                # (number of zeros == 1 AND the zero is at index r is FALSE, i.e., zero is not at r)
                # Actually simpler: count zeros in counts[p], if zeros >= 2, ok. If zeros == 1, check if it's at r.
                # If zeros == 0, not ok (but we already handled min_val > 0)
                
                # Count zeros
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
                    continue  # No violation (the zero is not at r, so it stays 0)
                else:
                    # zero_count == 0 (handled) or zero_count == 1 and zero_at_r
                    # After adding, the zero at r becomes 1, no more zeros -> violation
                    ok = False
                    break
        
        if ok:
            initial.append(v)
            point_set.add(v)
            for p in track_primes:
                counts[p][v % p] += 1
    
    return _repair(initial)


def _try_add_more(points):
    t0 = time.time()
    point_set = set(points)
    candidates = [v for v in range(N + 1) if v not in point_set]
    
    small_primes = [p for p in _ALL_PRIMES if p <= 500]
    scores = [0] * (N + 1)
    for p in small_primes:
        for multiple in range(0, N + 1, p):
            scores[multiple] += 1
    
    candidates.sort(key=lambda x: (-scores[x], x))
    
    max_track_prime = 3000
    track_primes = [p for p in _ALL_PRIMES if p <= max_track_prime]
    
    counts = {}
    for p in track_primes:
        counts[p] = [0] * p
    for v in points:
        for p in track_primes:
            counts[p][v % p] += 1
    
    for v in candidates:
        if time.time() - t0 > DEADLINE - 20:
            break
        
        m_new = len(points) + 1
        ok = True
        for p in track_primes:
            if p > m_new:
                break
            r = v % p
            min_val = min(counts[p])
            if min_val > 0:
                ok = False
                break
            else:
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


def construct():
    t0 = time.time()
    points = _greedy_with_tracking()
    points = _try_add_more(points)
    return points
# EVOLVE-BLOCK-END
