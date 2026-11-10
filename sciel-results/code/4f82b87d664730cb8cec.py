"""Frontier-based greedy with local search for prime residue class covering."""
import time
import math
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from fractions import Fraction

def _sieve_primes(limit):
    """Generate all primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _compute_frontier(covered_arr, start=1, cap=100000):
    """Find the largest L such that all positions 1..L are covered."""
    # covered_arr is a numpy boolean array, index 0 unused
    # Find first False from position 1 onwards
    if len(covered_arr) < 2:
        return 0
    # Use np.argmax on ~covered_arr[1:start+1000] to find first gap
    # But we need the first False, so flip
    segment = ~covered_arr[start:min(start+100000, len(covered_arr))]
    if not np.any(segment):
        return min(start + 100000, len(covered_arr) - 1)
    first_gap = np.argmax(segment)
    return start + first_gap - 1

def _mark_coverage(covered_arr, p, r, cap=100000):
    """Mark positions r mod p, r+p, r+2p, ... in covered_arr."""
    r_mod = r % p
    if r_mod == 0:
        start = p
    else:
        start = r_mod
    if start <= cap:
        covered_arr[start::p] = True

def _unmark_coverage(covered_arr, p, r, cap=100000):
    """Unmark positions r mod p, r+p, r+2p, ... in covered_arr."""
    r_mod = r % p
    if r_mod == 0:
        start = p
    else:
        start = r_mod
    if start <= cap:
        # We can't simply unmark because other classes might cover those positions
        # This function is only used when we know this is the sole coverer
        covered_arr[start::p] = False

def _score_classes(classes, cap=100000):
    """Compute the score (largest L) for a set of classes."""
    if not classes:
        return 0
    covered = np.zeros(cap + 1, dtype=bool)
    for p, r in classes:
        _mark_coverage(covered, p, r, cap)
    return _compute_frontier(covered, 1, cap)

def _greedy_construct(primes, deadline, cap=100000):
    """
    Frontier-based greedy: at each step, add the (p, r) that extends the frontier the most.
    """
    classes = []
    used_primes = set()
    budget = Fraction(0, 1)
    current_frontier = 0
    
    # Precompute 1/p for all primes
    inv_primes = {p: Fraction(1, p) for p in primes}
    
    # Sort primes by 1/p descending (smaller primes first)
    sorted_primes = sorted(primes, key=lambda p: -inv_primes[p])
    
    while time.time() < deadline:
        best_gain = 0
        best_p = None
        best_r = None
        
        for p in sorted_primes:
            if p in used_primes:
                continue
            if budget + inv_primes[p] > Fraction(1, 1):
                continue
            
            # Try a subset of remainders
            # For small primes, try all; for large, try strategic ones
            if p <= 50:
                candidates = range(p)
            elif p <= 200:
                candidates = list(range(0, p, max(1, p // 25))) + [0, 1, 2, 3, p-1]
            else:
                # Strategic remainders near the frontier
                candidates = [0, 1, 2, 3, p//2, p-1]
                if current_frontier > 0:
                    candidates.extend([
                        (current_frontier + 1) % p,
                        (current_frontier + 2) % p,
                        (current_frontier + 3) % p,
                    ])
            
            best_r_local = 0
            best_gain_local = 0
            seen = set()
            
            for r in candidates:
                r = r % p
                if r in seen:
                    continue
                seen.add(r)
                
                # Test adding (p, r)
                test_classes = classes + [[p, r]]
                s = _score_classes(test_classes, cap)
                gain = s - current_frontier
                if gain > best_gain_local:
                    best_gain_local = gain
                    best_r_local = r
            
            if best_gain_local > best_gain:
                best_gain = best_gain_local
                best_p = p
                best_r = best_r_local
        
        if best_p is None:
            break
        
        classes.append([best_p, best_r])
        used_primes.add(best_p)
        budget += inv_primes[best_p]
        current_frontier = _score_classes(classes, cap)
    
    return classes, _score_classes(classes, cap)

def _local_search(classes, primes, deadline, cap=100000):
    """
    Hill-climbing: try to improve by changing remainders or swapping primes.
    """
    best_classes = [list(c) for c in classes]
    best_score = _score_classes(best_classes, cap)
    used = set(c[0] for c in best_classes)
    budget = sum(Fraction(1, c[0]) for c in best_classes)
    
    inv_primes = {p: Fraction(1, p) for p in primes}
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try adjusting each remainder
        for i in range(len(best_classes)):
            if time.time() > deadline:
                break
            
            p, old_r = best_classes[i]
            best_r = old_r
            best_local = best_score
            
            # Determine candidate remainders
            if p <= 60:
                candidates = range(p)
            elif p <= 200:
                step = max(1, p // 30)
                candidates = list(range(0, p, step)) + [0, 1, 2, 3, p-1, old_r]
            else:
                candidates = [0, 1, 2, 3, p//2, p-1, old_r]
                if best_score > 0:
                    frontier = best_score + 1
                    candidates.extend([frontier % p, (frontier+1) % p, (frontier+2) % p])
            
            seen = set()
            for r in candidates:
                r = r % p
                if r in seen or r == old_r:
                    continue
                seen.add(r)
                
                test = [list(c) for c in best_classes]
                test[i] = [p, r]
                s = _score_classes(test, cap)
                if s > best_local:
                    best_local = s
                    best_r = r
            
            if best_r != old_r:
                best_classes[i] = [p, best_r]
                best_score = best_local
                improved = True
        
        # Phase 2: Try swapping a prime
        if time.time() > deadline:
            break
        
        for i in range(len(best_classes)):
            if time.time() > deadline:
                break
            
            p_old, r_old = best_classes[i]
            
            # Try removing this prime and adding a different one
            remaining = [list(c) for c in best_classes if c[0] != p_old]
            rem_budget = sum(Fraction(1, c[0]) for c in remaining)
            
            best_new_p = None
            best_new_r = None
            best_s = best_score
            
            for p_new in primes:
                if p_new in used and p_new != p_old:
                    continue
                if p_new == p_old:
                    continue
                if rem_budget + inv_primes[p_new] > Fraction(1, 1):
                    continue
                
                # Try a few remainders
                if p_new <= 50:
                    candidates = range(p_new)
                else:
                    candidates = [0, 1, 2, 3, p_new//2, p_new-1]
                    if best_score > 0:
                        candidates.extend([(best_score+1) % p_new, (best_score+2) % p_new])
                
                seen = set()
                for r in candidates:
                    r = r % p_new
                    if r in seen:
                        continue
                    seen.add(r)
                    test = remaining + [[p_new, r]]
                    s = _score_classes(test, cap)
                    if s > best_s:
                        best_s = s
                        best_new_p = p_new
                        best_new_r = r
            
            if best_new_p is not None:
                best_classes[i] = [best_new_p, best_new_r]
                used = set(c[0] for c in best_classes)
                budget = sum(Fraction(1, c[0]) for c in best_classes)
                best_score = best_s
                improved = True
                break
        
        # Phase 3: Try adding a new prime
        if time.time() > deadline:
            break
        
        for p in sorted(primes, key=lambda x: -inv_primes[x]):
            if time.time() > deadline:
                break
            if p in used:
                continue
            if budget + inv_primes[p] > Fraction(1, 1):
                continue
            
            best_r = 0
            best_s = best_score
            
            if p <= 50:
                candidates = range(p)
            else:
                candidates = [0, 1, 2, 3, p//2, p-1]
                if best_score > 0:
                    candidates.extend([(best_score+1) % p, (best_score+2) % p])
            
            seen = set()
            for r in candidates:
                r = r % p
                if r in seen:
                    continue
                seen.add(r)
                test = best_classes + [[p, r]]
                s = _score_classes(test, cap)
                if s > best_s:
                    best_s = s
                    best_r = r
            
            if best_s > best_score:
                best_classes.append([p, best_r])
                used.add(p)
                budget += inv_primes[p]
                best_score = best_s
                improved = True
                break
    
    return best_classes, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    cap = 100000
    
    primes = _sieve_primes(1000)
    
    # Phase 1: Greedy construction
    classes, score = _greedy_construct(primes, deadline, cap)
    
    # Phase 2: Local search
    classes, score = _local_search(classes, primes, deadline, cap)
    
    # Final validation
    final_classes = []
    seen_primes = set()
    final_budget = Fraction(0, 1)
    for p, r in classes:
        if p in seen_primes:
            continue
        r = r % p
        if 0 <= r < p:
            final_classes.append([p, r])
            seen_primes.add(p)
            final_budget += Fraction(1, p)
    
    while final_budget > Fraction(1, 1) and final_classes:
        final_classes.sort(key=lambda x: -x[0])
        p, r = final_classes.pop()
        final_budget -= Fraction(1, p)
    
    return final_classes
# EVOLVE-BLOCK-END
