"""Covering 1..L by prime residue classes with reciprocal budget 1."""
import time
from fractions import Fraction
import numpy as np

_DEADLINE = 110.0

def _primes_up_to(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _coverage_length(covered, cap=100000):
    if cap <= 0:
        return 0
    if not np.all(covered[1:cap+1]):
        uncovered = np.where(covered[1:cap+1] == False)[0]
        if len(uncovered) > 0:
            return int(uncovered[0])
    return cap

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    primes = _primes_up_to(1000)
    MAX_L = 100000
    covered = np.zeros(MAX_L + 1, dtype=bool)
    
    budget = Fraction(1, 1)
    used = Fraction(0, 1)
    selected = []
    
    WORK_L = 2000
    work_covered = np.zeros(WORK_L + 1, dtype=bool)
    
    max_iters = 300
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if used >= budget:
            break
        
        best_score = 0
        best_p = None
        best_r = None
        
        for p in primes:
            if time.time() > deadline:
                break
            cost = Fraction(1, p)
            if used + cost > budget:
                continue
            
            for r in range(p):
                start = r if r > 0 else p
                if start > WORK_L:
                    continue
                positions = np.arange(start, WORK_L + 1, p)
                gain = int(np.sum(~work_covered[positions]))
                score = gain * p
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
        
        if best_p is None:
            break
        
        p = best_p
        r = best_r
        selected.append([p, r])
        used += Fraction(1, p)
        
        start = r if r > 0 else p
        positions = np.arange(start, WORK_L + 1, p)
        work_covered[positions] = True
        positions_full = np.arange(start, MAX_L + 1, p)
        covered[positions_full] = True
    
    L = _coverage_length(covered, MAX_L)
    return selected
