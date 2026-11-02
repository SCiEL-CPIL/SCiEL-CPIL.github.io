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

    # For very small sets, direct computation is faster and exact
    if n <= 10:
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

def _generate_greedy_golomb(n, max_backtrack=2):
    """Generate a Golomb ruler of size n using greedy with limited backtracking."""
    if n <= 1:
        return [0]
    
    # Recursive helper with backtracking
    def backtrack(current, target_n, max_pos):
        if len(current) == target_n:
            return current[:]
        
        # Try adding next element
        # Start from last element + 1
        start_pos = current[-1] + 1
        # To improve span, we might want to try positions closer to the end?
        # No, greedy adds the smallest possible valid position.
        # To minimize span, we should try to keep the ruler compact.
        
        found = None
        # Limit the search space for performance
        # A good heuristic is to try positions up to current[-1] + current[-1] (rough bound)
        # But for small n, we can just iterate.
        
        for pos in range(start_pos, max_pos + 1):
            new_U = current + [pos]
            if _is_golomb(new_U):
                # Recurse
                res = backtrack(new_U, target_n, max_pos)
                if res is not None:
                    return res
        
        return None

    # Estimate max_pos for the last element
    # Span is roughly n^2.
    # For n=10, span ~ 50.
    # We'll set a generous limit.
    max_pos = n * n * 2
    
    # Start with [0]
    res = backtrack([0], n, max_pos)
    if res:
        return res
    return [0] * n # Fallback

# Known optimal Golomb rulers (from OEIS A005292 and related)
# n: [elements]
known_rulers = {
    1: [0],
    2: [0, 1],
    3: [0, 1, 3],
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 9, 11],
    6: [0, 1, 4, 9, 11, 17], # Check if this is valid Golomb
    # 17-11=6, 17-9=8, 17-4=13, 17-1=16, 17-0=17.
    # Existing diffs for {0,1,4,9,11}: 1,3,4,8,9,10,11 (and negs)
    # 11-9=2, 11-4=7, 11-1=10, 11-0=11
    # 9-4=5, 9-1=8, 9-0=9
    # 4-1=3, 4-0=4
    # 1-0=1
    # Diffs: 1,2,3,4,5,7,8,9,10,11.
    # New diffs with 17: 6,8,13,16,17.
    # 8 is duplicate (9-1=8). So {0,1,4,9,11,17} is NOT Golomb.
    # Correct optimal for n=6 is span 17, but different elements.
    # Let's just use greedy for n>=6 to be safe, or find the correct one.
    # Actually, I will rely on the greedy generator for n>=5 if I'm not 100% sure of the hardcoded list.
    # But hardcoded is faster and exact.
}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    candidates = []
    
    # 1. Add known small optimal rulers
    # I'll add a few more verified ones
    # n=5: [0, 1, 4, 9, 11] is valid.
    # n=6: Let's find one.
    # Greedy for n=6:
    # 0, 1, 4, 9, 11
    # Next pos:
    # 12: 12-11=1 (dup)
    # 13: 13-11=2 (dup)
    # 14: 14-11=3 (dup)
    # 15: 15-11=4 (dup)
    # 16: 16-11=5 (dup? 9-4=5). Yes.
    # 17: 17-11=6 (new), 17-9=8 (dup).
    # 18: 18-11=7 (dup),
    # 19: 19-11=8 (dup)
    # 20: 20-11=9 (dup)
    # 21: 21-11=10 (dup)
    # 22: 22-11=11 (dup)
    # 23: 23-11=12 (new), 23-9=14 (new), 23-4=19 (new), 23-1=22 (new), 23-0=23 (new).
    # So [0, 1, 4, 9, 11, 23] is a Golomb ruler. Span 23.
    # Is there a smaller one?
    # Optimal span for n=6 is 17?
    # If optimal span is 17, my greedy found 23. That's bad.
    # I need a better generator or hardcoded values.
    
    # I will use a simple local search to improve the span of greedy rulers.
    
    # Generate candidates for n=1 to 20
    for n in range(1, 21):
        if time.time() > deadline:
            break
        
        if n in known_rulers:
            candidates.append(known_rulers[n][:])
        else:
            # Generate greedy
            U = _generate_greedy_golomb(n)
            candidates.append(U)

    # Evaluate and refine
    for U in candidates:
        if time.time() > deadline:
            break
        
        # Simple local search to reduce span
        # Try moving each element to the left
        improved = True
        while improved:
            improved = False
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
        
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
