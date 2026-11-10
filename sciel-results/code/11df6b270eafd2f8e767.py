"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _compute_max_discrepancy(signs):
    """Compute maximum absolute signed sum over all arithmetic progressions in [1, n]."""
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1
    
    for step in range(1, n + 1):
        for r in range(step):
            if r >= n:
                break
            subseq = arr[r::step]
            if len(subseq) == 0:
                continue
            prefix_sums = np.cumsum(subseq)
            current_max = np.max(np.abs(prefix_sums))
            if current_max > max_disc:
                max_disc = int(current_max)
    
    return max_disc

def _greedy_init(n, rng):
    """Greedy initialization: build signs one by one, choosing sign that minimizes current max discrepancy."""
    signs = [0] * n
    # Start with random first sign
    signs[0] = rng.choice([-1, 1])
    
    for i in range(1, n):
        # Try both signs, pick the one that gives lower max discrepancy
        best_sign = None
        best_disc = float('inf')
        for s in [-1, 1]:
            signs[i] = s
            # Only compute discrepancy for progressions involving position i
            # For efficiency, compute full discrepancy (n=256 is small enough)
            disc = _compute_max_discrepancy(signs)
            if disc < best_disc:
                best_disc = disc
                best_sign = s
        signs[i] = best_sign
    
    return signs

def _pairwise_local_search(signs, time_limit=55.0, stagnation_threshold=200):
    """Local search with pairwise flips and adaptive restart on stagnation."""
    n = len(signs)
    signs = list(signs)
    current_disc = _compute_max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    start_time = time.time()
    rng = random.Random(42)
    
    stagnation_count = 0
    T = 5.0
    alpha = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick two distinct positions to flip
        i, j = rng.sample(range(n), 2)
        si, sj = signs[i], signs[j]
        signs[i] = -si
        signs[j] = -sj
        
        new_disc = _compute_max_discrepancy(signs)
        delta = new_disc - current_disc
        
        if delta < 0 or (T > 0.01 and rng.random() < np.exp(-delta / T)):
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
        else:
            signs[i] = si
            signs[j] = sj
            stagnation_count += 1
        
        T *= alpha
        
        # If stuck for too long, try a greedy repair or restart
        if stagnation_count > stagnation_threshold:
            # Try a few greedy single flips to escape
            for _ in range(50):
                if time.time() - start_time >= time_limit:
                    break
                idx = rng.randint(0, n - 1)
                old = signs[idx]
                signs[idx] = -old
                d = _compute_max_discrepancy(signs)
                if d < current_disc:
                    current_disc = d
                    if d < best_disc:
                        best_disc = d
                        best_signs = signs.copy()
                        stagnation_count = 0
                    break
                else:
                    signs[idx] = old
            
            # Reset temperature and stagnation counter
            T = max(T, 2.0)
            stagnation_count = 0
    
    return best_signs, best_disc

def construct():
    n = 256
    start_time = time.time()
    deadline = 57.0  # Safety margin before 60s
    
    rng = random.Random(171)
    
    # Phase 1: Greedy initialization
    signs = _greedy_init(n, rng)
    best_signs, best_disc = signs.copy(), _compute_max_discrepancy(signs)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_signs
    
    # Phase 2: Pairwise local search from greedy start
    new_signs, new_disc = _pairwise_local_search(signs, time_limit=deadline - elapsed)
    if new_disc < best_disc:
        best_disc = new_disc
        best_signs = new_signs
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_signs
    
    # Phase 3: Try a few random restarts with pairwise search if time permits
    for restart in range(8):
        if time.time() - start_time > deadline:
            break
        signs = [rng.choice((-1, 1)) for _ in range(n)]
        new_signs, new_disc = _pairwise_local_search(signs, time_limit=deadline - (time.time() - start_time))
        if new_disc < best_disc:
            best_disc = new_disc
            best_signs = new_signs
    
    return best_signs
# EVOLVE-BLOCK-END
