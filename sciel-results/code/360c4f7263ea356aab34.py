"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft

def _score_set(U):
    """Compute the score for a finite set U containing 0, using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    # For very small sets, direct computation is faster
    if n <= 12:
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

    # FFT-based computation for larger sets
    # Sumset: U + U, values in [0, 2*max_u]
    sum_range = 2 * max_u + 1
    fft_len = 1
    while fft_len < 2 * sum_range:
        fft_len *= 2

    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0

    freq = fft(indicator)
    conv = ifft(freq * freq).real
    sumset_size = int(np.sum(conv[:sum_range] > 1e-9))

    # Difference set: U - U, values in [-max_u, max_u]
    diff_range = 2 * max_u + 1
    fft_len2 = 1
    while fft_len2 < 2 * diff_range:
        fft_len2 *= 2

    diff_vec = np.zeros(fft_len2, dtype=np.float64)
    for u in U:
        diff_vec[u + max_u] = 1.0

    freq_d = fft(diff_vec)
    autocorr = ifft(freq_d * np.conj(freq_d)).real

    valid_mask = np.zeros(fft_len2, dtype=bool)
    valid_mask[0] = True
    valid_mask[1:max_u+1] = True
    valid_mask[fft_len2-max_u:fft_len2] = True
    diffset_size = int(np.sum(autocorr[valid_mask] > 1e-9))

    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _is_golomb(U):
    """Check if U is a Golomb ruler (all pairwise differences are distinct)."""
    diffs = set()
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _compress_golomb(U, max_steps=2000):
    """Try to reduce the span of a Golomb ruler by moving elements left."""
    U = sorted(set(U))
    if not _is_golomb(U):
        return U
    improved = True
    while improved:
        improved = False
        # Try moving each element (except 0) to the left
        for i in range(len(U) - 1, 0, -1):
            original = U[i]
            for new_val in range(original - 1, -1, -1):
                if new_val == 0 and i != 0:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U = sorted(set(new_U))
                if len(new_U) != len(U):
                    continue
                if _is_golomb(new_U):
                    U = new_U
                    improved = True
                    break
            if improved:
                break
    return U

def _exhaustive_search_small(n, max_val, deadline):
    """Exhaustive search for small sets to find optimal configurations."""
    best_U = None
    best_score = -1.0
    from itertools import combinations
    for combo in combinations(range(1, max_val + 1), n - 1):
        if time.time() > deadline:
            break
        U = sorted([0] + list(combo))
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    return best_U, best_score

def _local_search(U, deadline, max_iters=5000):
    """Improve set U via local search with add/remove/swap operations."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        max_u = max(U)
        upper_bound = max_u + 30
        
        # Try swapping one element
        random.shuffle(U)
        for i in range(len(U)):
            old_val = U[i]
            for new_val in range(0, upper_bound + 1):
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
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(U) < 25:
                for new_val in range(0, upper_bound + 1):
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        improved = True
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
                        improved = True
                        break
        
        if not improved:
            break
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    # Known optimal Golomb rulers (span minimized)
    known_rulers = {
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
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 73],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 73, 85],
    }

    # Generate candidates
    candidates = []
    
    # Use known rulers
    for n in range(1, 15):
        if n in known_rulers:
            candidates.append(known_rulers[n][:])

    # Generate greedy rulers for larger n
    for n in range(15, 22):
        if time.time() > deadline:
            break
        U = [0]
        for i in range(1, n):
            pos = 1
            while True:
                U.append(pos)
                if _is_golomb(U):
                    break
                U.pop()
                pos += 1
        candidates.append(U)

    # Evaluate and compress
    for U in candidates:
        if time.time() > deadline:
            break
        # Compress to reduce M
        U_comp = _compress_golomb(U)
        s = _score_set(U_comp)
        if s > best_score:
            best_score = s
            best_U = U_comp[:]

    # Exhaustive search for small sets
    if time.time() < deadline:
        # Search for optimal sets of size 3-5
        for n in range(3, 6):
            if time.time() > deadline:
                break
            max_val = 30 if n <= 4 else 25
            U_found, s_found = _exhaustive_search_small(n, max_val, deadline)
            if U_found and s_found > best_score:
                best_score = s_found
                best_U = U_found[:]

    # Local search on the best candidate
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, deadline, max_iters=3000)

    # Try local search on a few other candidates
    for U in candidates[:8]:
        if time.time() > deadline:
            break
        if U == best_U:
            continue
        s = _score_set(U)
        if s > best_score - 0.02:  # only if close
            improved_U, improved_score = _local_search(U, deadline, max_iters=1500)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
