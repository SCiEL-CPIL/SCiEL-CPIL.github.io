"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft
from itertools import combinations

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
    sumset_size = int(np.sum(conv[:sum_len] > 1e-9))
    
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
    diffset_size = int(np.sum(np.abs(diff_result) > 1e-9))
    
    if sumset_size == 0:
        return 0.0
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _greedy_prime_init(max_val=200, target_size=15, deadline=None):
    """Build a high-quality set using greedy selection maximizing incremental score."""
    start_time = time.time()
    U = [0]
    current_score = 1.0
    
    # Candidate pool: primes and structured values
    candidates = []
    # Add primes up to max_val
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    for i in range(2, max_val + 1):
        if is_prime(i):
            candidates.append(i)
    
    # Add some composite values for diversity
    for i in range(2, max_val + 1):
        if not is_prime(i) and i % 2 != 0:
            candidates.append(i)
    
    candidates = sorted(set(candidates))
    
    for _ in range(target_size - 1):
        if deadline and time.time() > deadline:
            break
        best_candidate = None
        best_score = current_score
        
        for cand in candidates:
            if cand in U:
                continue
            new_U = U + [cand]
            s = _score_set(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_candidate = cand
        
        if best_candidate is not None:
            U.append(best_candidate)
            current_score = best_score
        else:
            break
    
    return sorted(U), current_score

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
    """Improve set U via local search with targeted moves."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 300  # wider search space for swaps
    
    stall_count = 0
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Strategy 1: Try swapping one element with targeted neighborhood
        for i in range(len(U)):
            if U[i] == 0:
                continue  # must keep 0
            old_val = U[i]
            # Build targeted candidate set
            candidates = set()
            # Local neighborhood (tighter for efficiency)
            for offset in range(-30, 31):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            # Values derived from other elements
            for other in U:
                if other != old_val and other != 0:
                    for mult in [2, 3, 5]:
                        nv = other * mult
                        if 0 < nv <= upper_bound:
                            candidates.add(nv)
                    nv = abs(other - old_val)
                    if 0 < nv <= upper_bound:
                        candidates.add(nv)
                    nv = other + old_val
                    if 0 < nv <= upper_bound:
                        candidates.add(nv)
            # Random candidates
            for _ in range(10):
                candidates.add(random.randint(1, upper_bound))
            
            # Sort candidates by proximity to old_val for faster convergence
            for new_val in sorted(candidates, key=lambda x: abs(x - old_val)):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    max_u = max(U)
                    upper_bound = max_u + 300
                    improved = True
                    stall_count = 0
                    break
            if improved:
                break
        
        if not improved:
            # Strategy 2: Try adding an element from prime candidates
            if len(U) < 25:
                def is_prime(n):
                    if n < 2:
                        return False
                    if n == 2:
                        return True
                    if n % 2 == 0:
                        return False
                    for i in range(3, int(math.sqrt(n)) + 1, 2):
                        if n % i == 0:
                            return False
                    return True
                
                # Try primes in range
                for p in range(2, upper_bound + 1):
                    if p in U:
                        continue
                    if not is_prime(p):
                        continue
                    new_U = U + [p]
                    s = _score_set(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 300
                        improved = True
                        stall_count = 0
                        break
                if not improved:
                    # Try random additions
                    for _ in range(20):
                        new_val = random.randint(1, upper_bound)
                        if new_val in U:
                            continue
                        new_U = U + [new_val]
                        s = _score_set(new_U)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_U = new_U[:]
                            U = new_U[:]
                            max_u = max(U)
                            upper_bound = max_u + 300
                            improved = True
                            stall_count = 0
                            break
            
            if not improved:
                # Strategy 3: Try removing an element (if size > 3)
                if len(U) > 3:
                    for i in range(len(U)):
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i+1:]
                        s = _score_set(new_U)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_U = new_U[:]
                            U = new_U[:]
                            max_u = max(U)
                            upper_bound = max_u + 300
                            improved = True
                            stall_count = 0
                            break
        
        if not improved:
            stall_count += 1
            # Strategy 4: Larger random perturbation when stalled
            if stall_count >= 3 or (stall_count >= 1 and random.random() < 0.2):
                new_U = U[:]
                n_changes = min(random.randint(1, 4), len(new_U) - 1)
                for _ in range(n_changes):
                    idx = random.randint(1, len(new_U) - 1)
                    new_U[idx] = random.randint(1, upper_bound)
                new_U = sorted(set(new_U))
                if 0 in new_U and len(new_U) >= 2:
                    s = _score_set(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 300
                        improved = True
                        stall_count = 0
            
            if not improved and stall_count > 30:
                break
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Phase 1: Evaluate known high-quality initial sets
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
    
    for n, gr in known_grs.items():
        s = _score_set(gr)
        if s > best_score:
            best_score = s
            best_U = gr[:]
    
    # Phase 2: Greedy prime-based initialization (shown to improve scores)
    if time.time() < deadline - 60:
        greedy_U, greedy_score = _greedy_prime_init(max_val=200, target_size=15, deadline=deadline - 60)
        if greedy_score > best_score:
            best_score = greedy_score
            best_U = greedy_U
    
    # Phase 3: Local search refinement
    if time.time() < deadline - 15:
        best_U, best_score = _local_search(best_U, max_iters=3000, deadline=deadline - 15)
    
    # Phase 4: Final polish - try small perturbations and scaling
    if time.time() < deadline - 5:
        # Try removing the largest element if it hurts
        if len(best_U) > 3:
            test_U = best_U[:-1]
            s = _score_set(test_U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = test_U
        
        # Try scaling down
        max_u = max(best_U)
        if max_u > 30:
            for factor in [2, 3]:
                scaled = sorted(set([0] + [max(1, round(u / factor)) for u in best_U if u > 0]))
                if 0 in scaled and len(scaled) >= 2:
                    s = _score_set(scaled)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = scaled
        
        # One more round of local search with tighter tolerance
        if time.time() < deadline - 3:
            best_U, best_score = _local_search(best_U, max_iters=500, deadline=deadline - 3)
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
