"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft

# Global deadline tracking
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 140  # 10s safety margin before 150s
    return _DEADLINE

def _time_remaining():
    return max(0, _get_deadline() - time.time())

def _next_pow2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

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
    fft_len = _next_pow2(2 * sum_len)
    
    # Create indicator vector for U
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    # FFT to compute convolution (sumset)
    freq = fft(indicator)
    conv = ifft(freq * freq).real
    
    # Count nonzero entries in sumset (values 0 to 2*max_u)
    sumset_size = int(np.sum(conv[:sum_len] > 1e-9))
    
    # Compute difference set using FFT
    # U - U: values range from -max_u to max_u
    # Use autocorrelation: place U at offset max_u, then autocorrelate
    diff_len = 2 * max_u + 1
    fft_len2 = _next_pow2(2 * diff_len)
    
    # Place U in a vector of length fft_len2, shifted by max_u
    diff_vec = np.zeros(fft_len2, dtype=np.float64)
    for u in U:
        diff_vec[u + max_u] = 1.0
    
    # Autocorrelation via FFT
    freq_d = fft(diff_vec)
    diff_result = ifft(freq_d * np.conj(freq_d)).real
    
    # The difference d = a - b appears at position (max_u + d) in diff_result
    # So differences range from -max_u to max_u, appearing at positions 0 to 2*max_u
    diffset_size = int(np.sum(np.abs(diff_result[:diff_len]) > 1e-9))
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set_fast(U):
    """Fast scoring using Python sets - good for medium-sized sets."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    for i in range(n):
        ui = U[i]
        for j in range(n):
            sumset.add(ui + U[j])
            diffset.add(ui - U[j])
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set_exact(U):
    """Exact O(n^2) score computation for small sets (high precision)."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    for i in range(n):
        for j in range(n):
            sumset.add(U[i] + U[j])
            diffset.add(U[i] - U[j])
    ratio = len(diffset) / len(sumset)
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
    
    # Choose method based on set size and max value
    if n <= 50 and max_u <= 5000:
        return _score_set_fast(U)
    elif max_u <= 200000:
        return _score_set_fft(U)
    else:
        # For very large max_u with small n, use fast method
        return _score_set_fast(U)

def _golomb_ruler(n):
    """Generate a Golomb ruler of order n using known optimal + greedy construction."""
    if n <= 1:
        return [0]
    # Known optimal Golomb rulers (shortest length)
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
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86],
        16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95],
        17: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104],
        18: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104, 113],
        19: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104, 113, 122],
        20: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86, 95, 104, 113, 122, 131],
    }
    if n in known:
        return known[n]
    # Greedy with backtracking for larger n
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

def _local_search(U, max_iters=5000, deadline=None):
    """Improve set U via local search with simulated annealing, focused on Golomb structure."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 200  # tighter search space
    
    current_score = best_score
    current_U = U[:]
    
    # Simulated annealing parameters
    T = 0.03  # lower initial temperature for more exploitation
    cooling_rate = 0.998
    min_T = 0.0005
    
    no_improve_count = 0
    max_no_improve = 40
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if _time_remaining() < 5:
            break
        if no_improve_count >= max_no_improve and T < min_T:
            break
        
        improved = False
        # Try swapping one element - prioritize keeping Golomb property
        indices_to_try = list(range(1, len(U)))  # skip 0
        random.shuffle(indices_to_try)
        
        for i in indices_to_try:
            old_val = U[i]
            # Try nearby values with focus on maintaining spread
            candidates = set()
            for offset in range(-20, 21):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            # Add some random candidates in range
            for _ in range(10):
                candidates.add(random.randint(1, upper_bound))
            
            # Try closest first
            for new_val in sorted(candidates, key=lambda x: abs(x - old_val)):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                delta_s = s - current_score
                if delta_s > 1e-10:
                    current_score = s
                    current_U = new_U[:]
                    U = new_U[:]
                    if s > best_score:
                        best_score = s
                        best_U = new_U[:]
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    if T > min_T and random.random() < math.exp(delta_s / T):
                        current_score = s
                        current_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            # Try adding an element - prefer positions that maintain Golomb-like spread
            if len(U) < 50:
                # Try positions that would maintain distinct differences
                tried = set()
                for _ in range(30):
                    new_val = random.randint(1, upper_bound)
                    if new_val in U or new_val in tried:
                        continue
                    tried.add(new_val)
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    delta_s = s - current_score
                    if delta_s > 1e-10 or (T > min_T and random.random() < math.exp(delta_s / T)):
                        current_score = s
                        current_U = new_U[:]
                        U = new_U[:]
                        if s > best_score:
                            best_score = s
                            best_U = new_U[:]
                        improved = True
                        no_improve_count = 0
                        break
            
            if not improved:
                # Try removing an element
                if len(U) > 4:
                    for i in range(1, len(U)):
                        new_U = U[:i] + U[i+1:]
                        s = _score_set(new_U)
                        delta_s = s - current_score
                        if delta_s > 1e-10 or (T > min_T and random.random() < math.exp(delta_s / T)):
                            current_score = s
                            current_U = new_U[:]
                            U = new_U[:]
                            if s > best_score:
                                best_score = s
                                best_U = new_U[:]
                            improved = True
                            no_improve_count = 0
                            break
        
        if not improved:
            # Try larger perturbation - swap two elements
            if random.random() < 0.15 and len(U) >= 4:
                new_U = U[:]
                i1, i2 = random.sample(range(1, len(new_U)), 2)
                new_U[i1] = random.randint(1, upper_bound)
                new_U[i2] = random.randint(1, upper_bound)
                if len(set(new_U)) == len(new_U):
                    s = _score_set(new_U)
                    delta_s = s - current_score
                    if delta_s > 1e-10 or (T > min_T and random.random() < math.exp(delta_s / T)):
                        current_score = s
                        current_U = new_U[:]
                        U = new_U[:]
                        if s > best_score:
                            best_score = s
                            best_U = new_U[:]
                        improved = True
                        no_improve_count = 0
        
        # Cool down
        T = max(min_T, T * cooling_rate)
    
    return sorted(set(best_U)), best_score

def _polish(U, deadline=None):
    """High-precision polish pass using exact scoring for small sets."""
    U = sorted(set(U))
    n = len(U)
    if n > 60:
        # Too large for exact O(n^2) scoring; use FFT-based local search instead
        return _local_search(U, max_iters=3000, deadline=deadline)
    
    best_score = _score_set_exact(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 100
    
    no_improve = 0
    max_no_improve = 30
    
    for it in range(10000):
        if deadline and time.time() > deadline:
            break
        if _time_remaining() < 3:
            break
        if no_improve >= max_no_improve:
            break
        
        improved = False
        indices_to_try = list(range(1, len(U)))  # skip 0
        random.shuffle(indices_to_try)
        
        for i in indices_to_try:
            old_val = U[i]
            # Try nearby values with wider range for better exploration
            for offset in range(-30, 31):
                nv = old_val + offset
                if nv <= 0:
                    continue
                new_U = U[:]
                new_U[i] = nv
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set_exact(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
            # Try add/remove with structured candidates
            if len(U) < 80:
                # Try adding at positions that maintain good spread
                for _ in range(40):
                    nv = random.randint(1, upper_bound)
                    if nv in U:
                        continue
                    new_U = U + [nv]
                    s = _score_set_exact(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
            if not improved and len(U) > 4:
                for i in range(1, len(U)):
                    new_U = U[:i] + U[i+1:]
                    s = _score_set_exact(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
    
    return sorted(set(best_U)), best_score

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 140  # safety margin before 150s limit
    deadline = _DEADLINE
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Try different initial constructions
    candidates = []
    
    # Golomb rulers - primary focus since they maximize |U-U|/|U+U| ratio
    for n in range(3, 35):
        if time.time() > deadline - 80:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)
    
    # Random sets with various sizes
    random.seed(42)
    for trial in range(30):
        if time.time() > deadline - 80:
            break
        n = random.randint(3, 50)
        max_val = random.randint(30, 1000)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Structured sets: arithmetic progressions with gaps
    for base in [2, 3, 5, 7, 11, 13, 17]:
        if time.time() > deadline - 80:
            break
        U = [0]
        for i in range(1, 30):
            U.append(i * base)
        candidates.append(U[:min(len(U), 30)])
    
    # Powers-based sets
    for p in [2, 3, 4, 5]:
        if time.time() > deadline - 80:
            break
        U = [0]
        for i in range(1, 20):
            U.append(p ** i)
            if U[-1] > 20000:
                break
        candidates.append(U)
    
    # Sparse random sets (higher sparsity)
    for trial in range(20):
        if time.time() > deadline - 80:
            break
        n = random.randint(5, 40)
        max_val = random.randint(200, 3000)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Mixed structured sets
    for trial in range(15):
        if time.time() > deadline - 80:
            break
        n = random.randint(10, 40)
        base = random.randint(2, 10)
        U = [0]
        for i in range(1, min(n, 15)):
            U.append(i * base)
        remaining = n - len(U)
        if remaining > 0:
            max_val = max(U) + random.randint(50, 300)
            extra = random.sample(range(1, max_val + 1), remaining)
            U.extend(extra)
        U = sorted(set(U))
        candidates.append(U)
    
    # Evaluate candidates and keep top ones
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline - 60:
            break
        s = _score_set(U)
        scored_candidates.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Local search on best candidate
    if time.time() < deadline - 40:
        best_U, best_score = _local_search(best_U, max_iters=15000, deadline=deadline - 40)
    
    # Try local search on top few other candidates
    for s, U in scored_candidates[:10]:
        if time.time() > deadline - 40:
            break
        if U == best_U:
            continue
        if s > best_score - 0.05:  # only if close
            improved_U, improved_score = _local_search(U, max_iters=5000, deadline=deadline - 40)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # Final high-precision polish pass
    if time.time() < deadline - 10:
        best_U, best_score = _polish(best_U, deadline=deadline - 5)
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
