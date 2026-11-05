"""
Low-discrepancy construction for arithmetic progressions on [1..256].
Uses a fast vectorized discrepancy evaluator plus a simulated-annealing /
multi-start local-search hybrid to drive the max absolute AP sum down.
"""
import time
import math
import random
import numpy as np

_N = 256

def _max_ap_discrepancy(signs):
    """
    Exact max |sum| over every contiguous sub-sequence of every
    arithmetic progression (start, step) contained in [0..n-1].
    Returns an integer.
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int32)
    best = 0
    # For each step d, each residue class r mod d
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                continue
            # indices: r, r+d, r+2d, ...
            idx = np.arange(r, n, d)
            sub = s[idx]
            # sliding window max subarray via prefix
            pref = np.concatenate(([0], np.cumsum(sub)))
            # max subarray sum = max(pref[j] - min_{i<j} pref[i])
            # vectorized: running min
            running_min = np.minimum.accumulate(pref)
            max_sub = int(np.max(pref - running_min))
            # min subarray sum = min(pref[j] - max_{i<j} pref[i])
            running_max = np.maximum.accumulate(pref)
            min_sub = int(np.min(pref - running_max))
            cur = max(abs(max_sub), abs(min_sub))
            if cur > best:
                best = cur
    return best

def _fast_max_ap_discrepancy(signs, max_step=None):
    """
    Slightly faster version: we can early-exit if best already >= current global.
    Also we skip very long APs that are unlikely to be worst? No, we need exact.
    But we can use int16 and avoid some overhead.
    Actually the main cost is the double loop over d and r. For n=256,
    number of (d,r) pairs is sum_{d=1}^{255} d = 255*256/2 = 32640.
    Each does a cumsum over length ~ n/d. Total work ~ sum_{d} d * (n/d) = n * num_d = 256*255 = 65280 elements.
    That's small. The Python loop overhead is the issue.
    Let's keep it simple and just call _max_ap_discrepancy.
    """
    return _max_ap_discrepancy(signs)

def _generate_initial_candidates(n=_N, num_random=8):
    """Generate a diverse set of starting candidates."""
    cands = []
    # Thue-Morse
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    cands.append(tm)
    # Alternating
    alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    cands.append(alt)
    # Sine-based
    for freq in [0.3, 0.5, 0.7, 1.0, 1.5]:
        sine = [1 if math.sin(2 * math.pi * freq * i / n) >= 0 else -1 for i in range(n)]
        cands.append(sine)
    # Random
    for seed in range(num_random):
        rng = random.Random(seed * 1000 + 7)
        rand = [rng.choice((-1, 1)) for _ in range(n)]
        cands.append(rand)
    # Block-based
    for bl in [4, 8, 16, 32]:
        blk = []
        for i in range(n):
            b = i // bl
            w = i % bl
            if b % 2 == 0:
                blk.append(1 if w < bl // 2 else -1)
            else:
                blk.append(-1 if w < bl // 2 else 1)
        cands.append(blk)
    return cands

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin
    n = _N
    rng = random.Random(12345)
    
    # Generate candidates
    candidates = _generate_initial_candidates(n)
    
    # Evaluate all candidates
    best_signs = None
    best_disc = float('inf')
    for c in candidates:
        d = _max_ap_discrepancy(c)
        if d < best_disc:
            best_disc = d
            best_signs = c[:]
    
    if best_signs is None:
        best_signs = [1] * n
        best_disc = n
    
    # Local search: simulated annealing / hill climbing
    # We'll do a combination: first pure hill climbing, then SA if time permits.
    improved = True
    no_improve_count = 0
    max_no_improve = 5
    
    while improved and time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        no_improve_count += 1
        
        # Try flipping each position
        for i in range(n):
            if time.time() > deadline:
                break
            old = best_signs[i]
            best_signs[i] = -old
            new_d = _max_ap_discrepancy(best_signs)
            if new_d < best_disc:
                best_disc = new_d
                improved = True
                no_improve_count = 0
            else:
                best_signs[i] = old
        
        # Also try some random flips to escape local minima
        for _ in range(20):
            if time.time() > deadline:
                break
            i = rng.randint(0, n - 1)
            old = best_signs[i]
            best_signs[i] = -old
            new_d = _max_ap_discrepancy(best_signs)
            if new_d < best_disc:
                best_disc = new_d
                improved = True
                no_improve_count = 0
            else:
                best_signs[i] = old
    
    # If we still have time, do a few random restarts with perturbation
    if time.time() < deadline - 30:
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            # Perturb the best solution
            perturbed = best_signs[:]
            for _ in range(rng.randint(5, 15)):
                i = rng.randint(0, n - 1)
                perturbed[i] = -perturbed[i]
            
            cur_disc = _max_ap_discrepancy(perturbed)
            if cur_disc < best_disc:
                best_signs = perturbed
                best_disc = cur_disc
            
            # Hill climb from perturbed
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                old = perturbed[i]
                perturbed[i] = -old
                new_d = _max_ap_discrepancy(perturbed)
                if new_d < best_disc:
                    best_disc = new_d
                    best_signs = perturbed[:]
                else:
                    perturbed[i] = old
    
    return best_signs
