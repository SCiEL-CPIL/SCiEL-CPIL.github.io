"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

_DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _elapsed(t0):
    return time.time() - t0

def _time_exhausted(t0, margin=2.0):
    return _elapsed(t0) > (_DEADLINE - margin)

def _score(chosen):
    """Compute sum 1/(512 - a) as float."""
    s = 0.0
    for a in chosen:
        s += 1.0 / (512.0 - a)
    return s

def _is_coprime_set(chosen):
    """Check if all pairs in chosen are coprime."""
    n = len(chosen)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(chosen[i], chosen[j]) != 1:
                return False
    return True

def _greedy_init():
    """Greedy: pick largest a first that is coprime to all chosen."""
    chosen = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    return chosen

def construct():
    t0 = time.time()
    chosen = _greedy_init()
    best_score = _score(chosen)
    
    # Local search: try swapping each chosen element with an unchosen one
    # to improve the score while maintaining pairwise coprimality.
    all_vals = set(range(1, 512))
    
    improved = True
    while improved and not _time_exhausted(t0):
        improved = False
        chosen_set = set(chosen)
        
        # For each position, try replacing with a candidate not in chosen
        for i in range(len(chosen)):
            if _time_exhausted(t0):
                break
            current = chosen[i]
            # Try candidates in descending order of 1/(512-a), i.e., descending a
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                # Check if replacing current with a maintains coprimality
                # a must be coprime to all other chosen elements (excluding current)
                ok = True
                for j in range(len(chosen)):
                    if j == i:
                        continue
                    if gcd(a, chosen[j]) != 1:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Compute new score
                new_score = best_score - 1.0 / (512.0 - current) + 1.0 / (512.0 - a)
                if new_score > best_score + 1e-15:
                    # Accept the swap
                    chosen[i] = a
                    chosen_set.remove(current)
                    chosen_set.add(a)
                    best_score = new_score
                    improved = True
                    break  # restart inner loop for this position
    
    return chosen
# EVOLVE-BLOCK-END
