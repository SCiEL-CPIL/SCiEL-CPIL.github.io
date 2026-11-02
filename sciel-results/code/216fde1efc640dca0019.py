"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft

def _score_set_fft(U):
    """Compute the score for a finite set U using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute sumset using FFT
    # U + U: values range from 0 to 2*max_u
    sum_len = 2 * max_u + 1
    # Pad to next power of 2 for efficient FFT
    fft_len = 1
    while fft_len < 2 * sum_len:
        fft_len *= 2
    
    # Create indicator vector for U
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    # FFT to compute convolution (sumset)
    freq = fft(indicator)
    conv = ifft(freq * freq).real
    
    # Count nonzero entries in sumset (values 0 to 2*max_u)
    sumset_size = np.sum(conv[:sum_len] > 1e-9)
    
    # Compute difference set using FFT
    # U - U: values range from -max_u to max_u
    # Use autocorrelation: convolve indicator with its reverse
    # Place U at offset max_u in a vector, then autocorrelate
    diff_len = 2 * max_u + 1
    fft_len2 = 1
    while fft_len2 < 2 * diff_len:
        fft_len2 *= 2
    
    # Place U in a vector of length fft_len2, shifted by max_u
    diff_vec = np.zeros(fft_len2, dtype=np.float64)
    for u in U:
        diff_vec[u + max_u] = 1.0
    
    # Autocorrelation: FFT(v) * conj(FFT(v))
    freq_d = fft(diff_vec)
    diff_result = ifft(freq_d * np.conj(freq_d)).real
    
    # In autocorrelation, the difference d appears at position max_u + d
    # So differences range from -max_u to max_u, appearing at positions 0 to 2*max_u
    # Count all nonzero entries in the full range
    diffset_size = np.sum(np.abs(diff_result) > 1e-9)
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # For small sets, use direct computation
    if n <= 30:
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        if log_denom == 0:
            return 1.0
        return 1.0 + log_ratio / log_denom
    else:
        return _score_set_fft(U)

def _golomb_ruler(n):
    """Generate a Golomb ruler of order n using greedy construction."""
    if n <= 1:
        return [0]
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    }
    if n in known:
        return known[n]
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler

def _local_search(U, max_iters=2000, deadline=None):
    """Improve set U via local search (swap elements)."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 200  # wider search space for swaps
    
    stall_count = 0
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try swapping one element with wider neighborhood
        for i in range(len(U)):
            if U[i] == 0:
                continue  # must keep 0
            old_val = U[i]
            # Try several new values around old_val and random positions
            candidates = set()
            # Wider local neighborhood
            for offset in range(-50, 51):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            # Add random candidates across the range
            for _ in range(15):
                candidates.add(random.randint(1, upper_bound))
            # Try values that create interesting gap patterns
            for other in U:
                if other != old_val and other != 0:
                    for mult in [2, 3]:
                        nv = other * mult
                        if nv > 0 and nv <= upper_bound:
                            candidates.add(nv)
                        nv = abs(other - old_val)
                        if nv > 0:
                            candidates.add(nv)
            
            for new_val in sorted(candidates):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    max_u = max(U)
                    upper_bound = max_u + 200
                    improved = True
                    stall_count = 0
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(U) < 300:
                for _ in range(30):
                    new_val = random.randint(1, upper_bound)
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
                        stall_count = 0
                        break
            
            if not improved:
                # Try removing an element (if size > 2)
                if len(U) > 2:
                    for i in range(len(U)):
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i+1:]
                        s = _score_set(new_U)
                        if s > best_score + 1e-9:
                            best_score = s
                            best_U = new_U[:]
                            U = new_U[:]
                            max_u = max(U)
                            upper_bound = max_u + 200
                            improved = True
                            stall_count = 0
                            break
        
        if not improved:
            stall_count += 1
            # Try a larger random perturbation
            if random.random() < 0.3 or stall_count > 5:
                new_U = U[:]
                # Randomly change a few elements
                n_changes = min(random.randint(2, 8), len(new_U) - 1)
                for _ in range(n_changes):
                    idx = random.randint(1, len(new_U) - 1)  # skip index 0 (value 0)
                    new_U[idx] = random.randint(1, upper_bound)
                new_U = sorted(set(new_U))
                if 0 in new_U and len(new_U) >= 2:
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
                        stall_count = 0
            
            if not improved and stall_count > 20:
                break
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Prioritize small, high-scoring Golomb rulers
    # Known optimal or near-optimal Golomb rulers for small N
    known_grs = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    }
    
    candidates = []
    for n, gr in known_grs.items():
        candidates.append(gr)
        
    # Add a few random and structured sets for diversity
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 50:
            break
        n = random.randint(3, 15)
        max_val = random.randint(20, 200)
        if n - 1 <= max_val:
            U = [0] + random.sample(range(1, max_val + 1), n - 1)
            candidates.append(sorted(U))
            
    # Evaluate candidates
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    for U in candidates:
        if time.time() > deadline - 40:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # Perform a focused local search on the best candidate
    # Only if we have time
    if time.time() < deadline - 10:
        best_U, best_score = _local_search(best_U, max_iters=2000, deadline=deadline - 10)
        
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
