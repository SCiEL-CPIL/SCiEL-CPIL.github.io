import time
import random
import math
import numpy as np

def _max_discrepancy(signs):
    """Compute max |sum| over all arithmetic progressions in signs[0..n-1]."""
    n = len(signs)
    max_abs = 0
    # Precompute prefix sums for step=1
    prefix = np.zeros(n + 1)
    prefix[1:] = np.cumsum(signs)
    
    # Step = 1: all contiguous substrings
    for i in range(n):
        for j in range(i + 1, n + 1):
            s = abs(int(prefix[j] - prefix[i]))
            if s > max_abs:
                max_abs = s
                if max_abs >= n:
                    return max_abs
    
    # Step = d for d >= 2
    for d in range(2, n):
        for r in range(d):
            # Collect elements at positions r, r+d, r+2d, ...
            idx = r
            if idx >= n:
                continue
            # Build prefix for this residue class
            length = (n - 1 - r) // d + 1
            if length <= 1:
                continue
            ps = np.zeros(length + 1)
            k = 0
            idx = r
            while idx < n:
                ps[k + 1] = ps[k] + signs[idx]
                k += 1
                idx += d
            # Check all sub-APs within this residue class
            for a in range(length):
                for b in range(a + 1, length + 1):
                    s = abs(int(ps[b] - ps[a]))
                    if s > max_abs:
                        max_abs = s
                        if max_abs >= n:
                            return max_abs
    return max_abs

def _max_discrepancy_fast(signs):
    """Faster: only check steps where improvement is possible, use early termination."""
    n = len(signs)
    max_abs = 0
    prefix = np.zeros(n + 1)
    prefix[1:] = np.cumsum(signs)
    
    # Step = 1
    for i in range(n):
        for j in range(i + 1, n + 1):
            s = abs(int(prefix[j] - prefix[i]))
            if s > max_abs:
                max_abs = s
                if max_abs >= n:
                    return max_abs
    
    # Step = d for d >= 2, but only up to where it matters
    for d in range(2, n):
        for r in range(d):
            if r >= n:
                break
            idx = r
            length = (n - 1 - r) // d + 1
            if length <= 1:
                continue
            ps = np.zeros(length + 1)
            k = 0
            idx = r
            while idx < n:
                ps[k + 1] = ps[k] + signs[idx]
                k += 1
                idx += d
            for a in range(length):
                for b in range(a + 1, length + 1):
                    s = abs(int(ps[b] - ps[a]))
                    if s > max_abs:
                        max_abs = s
                        if max_abs >= n:
                            return max_abs
    return max_abs

def _local_search(signs, time_limit, rng):
    """Hill climbing with random restarts within time limit."""
    n = len(signs)
    signs = list(signs)
    current_val = _max_discrepancy_fast(signs)
    best_val = current_val
    best_signs = list(signs)
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    while time.time() < deadline:
        improved = False
        # Try flipping each position
        for i in range(n):
            signs[i] = -signs[i]
            new_val = _max_discrepancy_fast(signs)
            if new_val < current_val:
                current_val = new_val
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_signs = list(signs)
            else:
                signs[i] = -signs[i]  # revert
            if time.time() > deadline:
                break
        if not improved:
            # Perturb: flip 1-3 random positions
            num_flips = rng.randint(1, 3)
            for _ in range(num_flips):
                idx = rng.randint(0, n - 1)
                signs[idx] = -signs[idx]
            current_val = _max_discrepancy_fast(signs)
            if current_val < best_val:
                best_val = current_val
                best_signs = list(signs)
    
    return best_signs, best_val

def _spectral_init(n, rng):
    """Initialize using spectral smoothing: create a smooth signal then threshold."""
    # Generate a random low-frequency signal
    freqs = np.arange(n)
    # Emphasize low frequencies
    weights = 1.0 / (1.0 + freqs ** 2)
    phase = rng.uniform(0, 2 * math.pi, n)
    amp = rng.uniform(0, 1, n) * weights
    signal = np.zeros(n)
    for k in range(n):
        signal += amp[k] * np.cos(2 * math.pi * freqs[k] * freqs / n + phase[k])
    # Normalize and threshold
    signal = signal - np.mean(signal)
    std = np.std(signal)
    if std > 0:
        signal = signal / std
    # Threshold to ±1
    signs = np.sign(signal)
    # Ensure balance
    while np.sum(signs) > 0:
        idx = np.where(signs == 1)[0]
        if len(idx) == 0:
            break
        signs[rng.choice(idx)] = -1
    while np.sum(signs) < 0:
        idx = np.where(signs == -1)[0]
        if len(idx) == 0:
            break
        signs[rng.choice(idx)] = 1
    return list(signs.astype(int))

def construct():
    n = 256
    start_time = time.time()
    total_deadline = start_time + 80.0  # Leave 10s margin
    
    rng = random.Random(42)
    
    # Generate multiple initial candidates
    candidates = []
    
    # Spectral initialization
    for _ in range(3):
        candidates.append(_spectral_init(n, rng))
    
    # Random initialization
    candidates.append([rng.choice((-1, 1)) for _ in range(n)])
    
    # Balanced random
    half = n // 2
    arr = [1] * half + [-1] * (n - half)
    rng.shuffle(arr)
    candidates.append(arr)
    
    # Evaluate initial candidates
    best_signs = None
    best_val = float('inf')
    
    for cand in candidates:
        val = _max_discrepancy_fast(cand)
        if val < best_val:
            best_val = val
            best_signs = list(cand)
    
    # Local search from best candidate
    time_remaining = total_deadline - time.time()
    if time_remaining > 5:
        best_signs, best_val = _local_search(best_signs, time_remaining - 5, rng)
    
    # Try local search from other candidates if time permits
    for cand in candidates:
        if cand == best_signs:
            continue
        time_remaining = total_deadline - time.time()
        if time_remaining < 10:
            break
        val = _max_discrepancy_fast(cand)
        if val < best_val + 2:  # Only try if close
            ts, tv = _local_search(cand, min(time_remaining - 5, 20), rng)
            if tv < best_val:
                best_val = tv
                best_signs = ts
    
    # Final polish: try single-flip improvements
    time_remaining = total_deadline - time.time()
    if time_remaining > 3:
        for i in range(n):
            if time.time() > total_deadline - 2:
                break
            best_signs[i] = -best_signs[i]
            new_val = _max_discrepancy_fast(best_signs)
            if new_val < best_val:
                best_val = new_val
            else:
                best_signs[i] = -best_signs[i]
    
    return best_signs
