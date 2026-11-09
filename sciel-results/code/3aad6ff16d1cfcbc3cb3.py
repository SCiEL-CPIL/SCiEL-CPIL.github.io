import time
import numpy as np
from math import log
from itertools import combinations

def _compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    n = len(U)
    if n == 0 or U[0] != 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for i, a in enumerate(U):
        for b in U[i:]:  # a+b = b+a
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def _try_local_search(U, max_iter=5000, deadline=None):
    """Try to improve U via local search."""
    U = sorted(set(U))
    best_score = _compute_score(U)
    best_U = U[:]
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        n = len(best_U)
        max_u = max(best_U)
        
        # Try adding a new element
        # Candidate positions: near existing elements or at strategic gaps
        candidates = set()
        for x in best_U:
            for d in [1, 2, 3, 5, 8, 13, 21]:
                candidates.add(x + d)
                candidates.add(x - d)
        # Also try some random positions
        for _ in range(10):
            candidates.add(int(np.random.randint(1, 2 * max_u + 10)))
        
        best_add_score = best_score
        best_add_elem = None
        for c in candidates:
            if c < 0:
                continue
            if c in best_U:
                continue
            new_U = sorted(best_U + [c])
            s = _compute_score(new_U)
            if s > best_add_score + 1e-12:
                best_add_score = s
                best_add_elem = c
        
        if best_add_elem is not None:
            best_U = sorted(best_U + [best_add_elem])
            best_score = best_add_score
        
        # Try removing an element (if n > 2)
        if n > 2:
            best_rem_score = best_score
            best_rem_idx = None
            for i in range(n):
                if i == 0:  # keep 0
                    continue
                new_U = best_U[:i] + best_U[i+1:]
                s = _compute_score(new_U)
                if s > best_rem_score + 1e-12:
                    best_rem_score = s
                    best_rem_idx = i
            
            if best_rem_idx is not None:
                best_U = best_U[:best_rem_idx] + best_U[best_rem_idx+1:]
                best_score = best_rem_score
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin
    
    # Start with several seed sets
    seeds = [
        [0, 1, 3],
        [0, 1, 4, 9],
        [0, 1, 3, 9, 27],
        [0, 1, 4, 16, 64],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 6, 10, 15],  # triangular-ish
    ]
    
    best_U = None
    best_score = 0.0
    
    for seed in seeds:
        if time.time() > deadline:
            break
        U, s = _try_local_search(seed, max_iter=3000, deadline=deadline)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Also try a pure random restart approach
    for _ in range(20):
        if time.time() > deadline:
            break
        # Random set
        size = int(np.random.randint(3, 8))
        U = [0] + sorted(np.random.randint(1, 100, size - 1).tolist())
        U = sorted(set(U))
        U, s = _try_local_search(U, max_iter=2000, deadline=deadline)
        if s > best_score:
            best_score = s
            best_U = U
    
    if best_U is None:
        return [0, 1, 3]
    
    return best_U
