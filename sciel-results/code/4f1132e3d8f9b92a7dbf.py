"""Sums and differences of finite sets - optimized construction."""
import time
import math
import random
import numpy as np
from itertools import combinations

# Deadline guard
_START_TIME = time.time()
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop

def _time_left():
    return DEADLINE - SAFETY_MARGIN - (time.time() - _START_TIME)

def _score(U):
    """Compute the lower bound score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if not U or max(U) == 0:
        return 1.0
    U_set = set(U)
    n = len(U_set)
    max_u = max(U_set)
    
    # Compute sumset U+U
    sumset = set()
    diffset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 1.0:
        return 1.0
    
    log_base = math.log(2 * max_u + 1, 2)
    if log_base <= 0:
        return 1.0
    
    return 1.0 + math.log(ratio, 2) / log_base

def _greedy_prime_step():
    """Build U using prime-step gaps with greedy score optimization."""
    # Start with 0
    U = [0]
    
    # Generate a list of small primes
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = []
    n = 2
    while len(primes) < 200:
        if is_prime(n):
            primes.append(n)
        n += 1
    
    # Greedily add elements
    best_score = _score(U)
    
    # Try different starting sequences
    for start_prime_idx in range(min(10, len(primes))):
        U_candidate = [0]
        current = 0
        used_primes = set()
        
        for p_idx in range(len(primes)):
            if _time_left() < 5.0:
                break
            p = primes[p_idx]
            if p in used_primes:
                continue
            new_elem = current + p
            U_candidate.append(new_elem)
            current = new_elem
            used_primes.add(p)
            
            # Check if score is improving
            s = _score(U_candidate)
            if s <= best_score:
                # Don't add this element if it doesn't help
                U_candidate.pop()
                current -= p
                used_primes.discard(p)
            else:
                best_score = s
        
        if best_score > 1.0:
            U = U_candidate
    
    return U

def _random_search():
    """Random search for good sets."""
    best_U = [0, 1, 3]
    best_score = _score(best_U)
    
    random.seed(42)
    
    for attempt in range(200):
        if _time_left() < 5.0:
            break
        
        # Random size
        n = random.randint(3, 20)
        
        # Generate random increasing sequence starting with 0
        U = [0]
        current = 0
        for _ in range(n - 1):
            step = random.randint(1, 50)
            current += step
            U.append(current)
        
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    return best_U

def _structured_search():
    """Try structured patterns like powers, factorials, etc."""
    candidates = []
    
    # Powers of 2
    U1 = [2**i for i in range(10)]
    candidates.append(U1)
    
    # Powers of 3
    U2 = [3**i for i in range(7)]
    candidates.append(U2)
    
    # Mersenne-like
    U3 = [2**i - 1 for i in range(1, 10)]
    candidates.append(U3)
    
    # Mixed
    U4 = [0, 1, 3, 7, 15, 31, 63, 127, 255]
    candidates.append(U4)
    
    # Sparse arithmetic
    for step in [2, 3, 5, 7, 11]:
        U5 = [i * step for i in range(15)]
        candidates.append(U5)
    
    best_U = [0, 1, 3]
    best_score = _score(best_U)
    
    for U in candidates:
        if _time_left() < 5.0:
            break
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    return best_U

def _local_optimization(U):
    """Try to improve U by adding/removing elements."""
    U = sorted(set(U))
    best_score = _score(U)
    
    if _time_left() < 5.0:
        return U
    
    # Try adding elements
    max_u = max(U) if U else 0
    for candidate in range(1, max_u + 100):
        if _time_left() < 5.0:
            break
        if candidate in U:
            continue
        U_new = U + [candidate]
        s = _score(U_new)
        if s > best_score:
            best_score = s
            U = U_new
            max_u = max(U)
    
    # Try removing elements (keep 0)
    changed = True
    while changed and _time_left() > 5.0:
        changed = False
        for i in range(1, len(U)):
            if _time_left() < 5.0:
                break
            U_new = U[:i] + U[i+1:]
            s = _score(U_new)
            if s > best_score:
                best_score = s
                U = U_new
                changed = True
                break
    
    return U

def _hybrid_search():
    """Combine multiple strategies."""
    results = []
    
    # Prime step greedy
    U_prime = _greedy_prime_step()
    results.append(U_prime)
    
    # Structured
    U_struct = _structured_search()
    results.append(U_struct)
    
    # Random
    U_rand = _random_search()
    results.append(U_rand)
    
    # Local optimization on best
    best_U = max(results, key=lambda u: _score(u))
    best_U = _local_optimization(best_U)
    
    return best_U

def construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    global _START_TIME
    _START_TIME = time.time()
    
    # Quick baseline
    baseline = [0, 1, 3]
    baseline_score = _score(baseline)
    
    # Try hybrid search
    try:
        U = _hybrid_search()
    except Exception:
        U = baseline
    
    # Ensure 0 is in U and all elements are nonnegative integers
    U = sorted(set(int(x) for x in U if isinstance(x, (int, np.integer)) and x >= 0))
    if 0 not in U:
        U.insert(0, 0)
    
    # If no improvement, return baseline
    if _score(U) <= baseline_score:
        return baseline
    
    return U
