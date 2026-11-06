import time
import random
import numpy as np

N = 256
DEADLINE = 85.0

def _max_abs_ap_sum(signs):
    """Fast computation of max absolute AP sum."""
    n = N
    s = signs  # list of -1/1
    max_abs = 0
    
    # For each step, for each residue class, compute prefix sums
    for step in range(1, n + 1):
        # For each starting position in 0..step-1
        for r in range(min(step, n)):
            # Build prefix sum for positions r, r+step, r+2*step, ...
            cumsum = 0
            for i in range(r, n, step):
                cumsum += s[i]
                a = abs(cumsum)
                if a > max_abs:
                    max_abs = a
                # Early exit if we can't beat current best significantly
                # (remaining elements can at most add (n-i)//step to abs)
    
    return max_abs

def _find_worst_ap_positions(signs):
    """Find positions in a worst AP. Returns list of 0-indexed positions."""
    n = N
    s = signs
    best_abs = -1
    best_positions = [0]
    
    for step in range(1, n + 1):
        for r in range(min(step, n)):
            cumsum = 0
            best_local = 0
            best_local_len = 0
            positions = []
            for i in range(r, n, step):
                cumsum += s[i]
                positions.append(i)
                a = abs(cumsum)
                if a > best_local:
                    best_local = a
                    best_local_len = len(positions)
            if best_local > best_abs:
                best_abs = best_local
                best_positions = positions[:best_local_len]
    
    return best_positions

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    n = N
    rng = random.Random(171)
    
    # Initial: random
    signs = [rng.choice((-1, 1)) for _ in range(n)]
    best_val = _max_abs_ap_sum(signs)
    best_signs = list(signs)
    
    # Try Thue-Morse
    tm = [1]
    for _ in range(8):
        tm = tm + [-x for x in tm]
    tm = tm[:n]
    val = _max_abs_ap_sum(tm)
    if val < best_val:
        best_val = val
        best_signs = list(tm)
    
    # Local search
    while time.time() < deadline - 5:
        # Find worst AP
        worst_pos = _find_worst_ap_positions(best_signs)
        
        # Try flipping each position in worst AP
        improved = False
        for pos in worst_pos:
            old = best_signs[pos]
            best_signs[pos] = -old
            val = _max_abs_ap_sum(best_signs)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_signs[pos] = old
        
        if not improved:
            # Try random flips
            for _ in range(20):
                pos = rng.randint(0, n - 1)
                old = best_signs[pos]
                best_signs[pos] = -old
                val = _max_abs_ap_sum(best_signs)
                if val < best_val:
                    best_val = val
                    improved = True
                else:
                    best_signs[pos] = old
            
            if not improved:
                # Perturb
                for _ in range(rng.randint(3, 8)):
                    pos = rng.randint(0, n - 1)
                    best_signs[pos] = -best_signs[pos]
                val = _max_abs_ap_sum(best_signs)
                if val < best_val:
                    best_val = val
    
    return best_signs
