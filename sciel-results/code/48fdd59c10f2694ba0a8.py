# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256

def _legendre_sequence(n, p):
    """Generate Legendre symbol sequence for 1..n mod p."""
    # p should be prime > n
    # Legendre symbol (a/p) = a^((p-1)/2) % p
    # If result is 1, it's a quadratic residue (return 1)
    # If result is p-1 (which is -1 mod p), it's a non-residue (return -1)
    # For a=0, it's 0, but we assume 1..n where n < p, so a != 0
    exp = (p - 1) // 2
    signs = []
    for i in range(1, n + 1):
        res = pow(i, exp, p)
        if res == 1:
            signs.append(1)
        elif res == p - 1:
            signs.append(-1)
        else:
            # Should not happen for prime p and i not divisible by p
            signs.append(1)
    return signs

def _max_abs_ap_sum_limited(signs, limit):
    """Compute max |sum| over all APs. Returns early if sum > limit."""
    n = len(signs)
    s = np.asarray(signs, dtype=np.int32)
    best = 0
    for d in range(1, n):
        for r in range(d):
            seq = s[r::d]
            if len(seq) > 1:
                cs = np.cumsum(seq)
                # Check if any abs sum exceeds limit
                if np.max(np.abs(cs)) > limit:
                    return limit + 1  # Indicate it's worse than limit
                m = int(np.max(np.abs(cs)))
                if m > best:
                    best = m
    return best

def _local_search(signs, deadline, rng):
    """Local search: flip bits to reduce max AP discrepancy."""
    n = len(signs)
    signs = list(signs)
    current = _max_abs_ap_sum_limited(signs, 10**9)
    best_signs = signs[:]
    best_val = current
    start = time.time()
    
    step = 0
    while time.time() < deadline:
        # Pick a random position to flip
        i = rng.randint(0, n - 1)
        signs[i] = -signs[i]
        # Only accept if it's strictly better than current
        # Use limited check: if new_val > current, reject quickly
        new_val = _max_abs_ap_sum_limited(signs, current)
        if new_val < current:
            current = new_val
            if new_val < best_val:
                best_val = new_val
                best_signs = signs[:]
            step = 0
        else:
            signs[i] = -signs[i]  # revert
            step += 1
            if step > 200:
                # Reset step counter to allow more exploration
                step = 0
    return best_signs, best_val

def construct():
    # Safety margin: 50s to ensure return before 60s kill
    deadline = time.time() + 50
    rng = random.Random(42)
    
    best_signs = None
    best_val = 10**9
    
    # 1. Try Legendre sequence initialization (known good structure)
    # Closest prime > 256 is 257
    p = 257
    leg_signs = _legendre_sequence(_N, p)
    
    # Run local search on Legendre sequence
    trial_deadline = min(deadline, time.time() + 10)
    res, val = _local_search(leg_signs, trial_deadline, rng)
    if val < best_val:
        best_val = val
        best_signs = res[:]
    
    # 2. Try a few random restarts if time remains
    if time.time() < deadline - 5:
        for trial in range(10):
            if time.time() >= deadline - 2:
                break
            signs = [rng.choice((-1, 1)) for _ in range(_N)]
            trial_deadline = min(deadline, time.time() + 2)
            res, val = _local_search(signs, trial_deadline, rng)
            if val < best_val:
                best_val = val
                best_signs = res[:]
    
    # 3. Final polish on the best found so far
    if best_signs is not None and time.time() < deadline - 1:
        res, val = _local_search(best_signs, deadline, rng)
        if val < best_val:
            best_val = val
            best_signs = res[:]
    
    if best_signs is None:
        best_signs = [1] * _N
    
    return best_signs
# EVOLVE-BLOCK-END
