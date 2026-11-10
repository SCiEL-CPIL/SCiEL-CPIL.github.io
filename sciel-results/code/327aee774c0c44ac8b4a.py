"""Rewritten program: simulated annealing with precomputed AP structure for min max|AP sum|."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_TOTAL_DEADLINE = 85  # wall-clock safety margin

def _precompute_aps(n):
    """Precompute all APs (start, step, length>=2) as index arrays."""
    aps = []
    for d in range(1, n):
        for r in range(d):
            idx = list(range(r, n, d))
            if len(idx) >= 2:
                aps.append(np.array(idx, dtype=np.int32))
    return aps

def _max_abs_ap_sum(signs, aps):
    """Compute max |sum| over all precomputed APs."""
    max_disc = 1
    for idx in aps:
        s = signs[idx].sum()
        a = abs(s)
        if a > max_disc:
            max_disc = a
    return max_disc

def _legendre_init(n):
    """Initialize using quadratic character mod 257 (prime)."""
    p = 257
    signs = np.zeros(n, dtype=np.int8)
    for i in range(n):
        x = (i + 1) % p
        if x == 0:
            signs[i] = 1
        else:
            signs[i] = 1 if pow(x, (p - 1) // 2, p) == 1 else -1
    return signs

def _simulated_annealing(signs, aps, max_time, rng):
    """Simulated annealing to minimize max |AP sum|."""
    n = len(signs)
    signs = signs.copy()
    current_max = _max_abs_ap_sum(signs, aps)
    best_max = current_max
    best_signs = signs.copy()
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # Temperature schedule
    t0 = max(current_max * 0.5, 2.0)
    t_end = 0.1
    n_steps_estimate = 50000
    cool_rate = (t_end / t0) ** (1.0 / n_steps_estimate)
    
    temp = t0
    step = 0
    
    while time.time() < deadline:
        # Batch of moves per temperature step for efficiency
        batch_size = 10
        for _ in range(batch_size):
            if time.time() > deadline:
                break
            pos = rng.randint(0, n - 1)
            old_sign = signs[pos]
            new_sign = -old_sign
            signs[pos] = new_sign
            
            new_max = _max_abs_ap_sum(signs, aps)
            delta = new_max - current_max
            
            if delta <= 0:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Accept with probability exp(-delta/temp)
                if temp > 0 and rng.random() < np.exp(-delta / temp):
                    current_max = new_max
                else:
                    signs[pos] = old_sign
        
        temp *= cool_rate
        step += 1
    
    return best_signs, best_max

def construct():
    """
    Construct 256 signs (-1 or +1) minimizing the maximum absolute signed sum
    over any arithmetic progression in [1, 256].
    """
    start_time = time.time()
    
    # Precompute APs once
    aps = _precompute_aps(_N)
    
    rng = random.Random(12345)
    
    # Multiple initializations
    inits = []
    # Quadratic character init
    inits.append(_legendre_init(_N))
    # Random inits
    for _ in range(3):
        inits.append(np.array([rng.choice((-1, 1)) for _ in range(_N)], dtype=np.int8))
    
    best_signs = None
    best_max = float('inf')
    
    for i, init_signs in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = _TOTAL_DEADLINE - elapsed - 5  # 5s safety margin
        if remaining <= 5:
            break
        
        # Allocate more time to earlier inits
        time_frac = 1.0 / (len(inits) - i)
        sa_time = min(remaining * time_frac * 0.9, remaining - 5)
        
        result_signs, result_max = _simulated_annealing(init_signs, aps, sa_time, rng)
        
        if result_max < best_max:
            best_max = result_max
            best_signs = result_signs
    
    # If we have time left, do a final polish on the best
    elapsed = time.time() - start_time
    remaining = _TOTAL_DEADLINE - elapsed - 5
    if remaining > 10 and best_signs is not None:
        final_signs, final_max = _simulated_annealing(best_signs, aps, remaining, rng)
        if final_max < best_max:
            best_max = final_max
            best_signs = final_signs
    
    if best_signs is None:
        best_signs = _legendre_init(_N)
    
    return [int(x) for x in best_signs]

# EVOLVE-BLOCK-END
