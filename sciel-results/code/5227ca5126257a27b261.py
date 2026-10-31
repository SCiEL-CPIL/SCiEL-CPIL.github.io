import time
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    best_score = -1
    best_set = [0, 1, 3]

    def compute_score(U):
        """Compute score using FFT for O(M log M) sumset/diffset size computation."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        if len(U) < 2:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        
        n = len(U)
        # For small sets, naive is faster
        if n <= 15:
            sumset = set()
            for a in U:
                for b in U:
                    sumset.add(a + b)
            diffset = set()
            for a in U:
                for b in U:
                    diffset.add(a - b)
            n_sum = len(sumset)
            n_diff = len(diffset)
        else:
            # FFT-based computation
            # Sumset: |U+U| = number of positions where autocorrelation of indicator > 0
            # We need a transform size large enough to hold U+U (range 0 to 2*max_u)
            # and U-U (range -max_u to max_u)
            
            # For sumset: create indicator array, FFT, square, IFFT
            # Size must be > 2*max_u to avoid wrap-around
            size_sum = 1
            while size_sum <= 2 * max_u:
                size_sum *= 2
            
            # Check if size is reasonable (avoid memory issues)
            if size_sum > 2**24:
                # Fall back to naive for very large sets
                sumset = set()
                for a in U:
                    for b in U:
                        sumset.add(a + b)
                diffset = set()
                for a in U:
                    for b in U:
                        diffset.add(a - b)
                n_sum = len(sumset)
                n_diff = len(diffset)
            else:
                # Indicator vector for U
                f = np.zeros(size_sum, dtype=np.float64)
                for x in U:
                    f[x] = 1.0
                
                # FFT of indicator
                F = np.fft.fft(f)
                
                # |U+U|: positions where convolution f*f has non-zero values
                # Convolution via FFT: IFFT(|F|^2) gives cyclic convolution
                conv_sum = np.real(np.fft.ifft(F * F))
                # Round to handle floating point errors
                n_sum = int(np.sum(conv_sum > 0.5))
            
            # For diffset: |U-U| = |U+(-U)|
            # Create indicator for -U (shifted by max_u to make non-negative)
            # Actually, U-U range is [-max_u, max_u], so we need size > 2*max_u
            # We can compute it as: shift U by max_u, then compute sumset of shifted set
            # Or directly: diffset size = number of positions where autocorrelation is non-zero
            # Autocorrelation: IFFT(|F|^2) gives the same as sumset autocorrelation
            # Wait, let me think more carefully.
            
            # U-U = {a-b : a,b in U}
            # Let g[i] = 1 if i in U, else 0
            # Then (g * g_rev)[k] = sum_i g[i] * g_rev[k-i] = sum_i g[i] * g[k-i]
            # This is the cross-correlation, which equals the sumset of U and -U shifted.
            
            # Actually, for difference set:
            # |U-U| = number of distinct values of a-b for a,b in U
            # We can compute this by: create array h where h[x + max_u] = 1 if x in U
            # Then h * h_rev gives us the difference counts shifted.
            
            # Simpler approach: use the fact that |U-U| = |U + (-U)|
            # Create indicator for -U shifted: let v[i] = 1 if (i - max_u) in U, i.e., i in U + max_u
            # Then U-U shifted by max_u = (U + max_u) + (-U + max_u) - max_u... this is getting complex.
            
            # Let me just use: the autocorrelation R[k] = sum_x f[x] * f[x-k] for k in [-max_u, max_u]
            # |U-U| = number of k where R[k] > 0
            # R[k] for k>=0 is the same as the sumset autocorrelation we already computed.
            # R[k] for k<0 = R[-k] by symmetry (since we're dealing with real signals and the set)
            # Wait no, R[k] = sum_x f[x]f[x-k], R[-k] = sum_x f[x]f[x+k] = sum_y f[y-k]f[y] = R[k]
            # So R is symmetric! Thus |U-U| = number of k in [-max_u, max_u] where R[k] > 0
            # = 1 + 2 * (number of k in [1, max_u] where R[k] > 0)
            
            # But we computed conv_sum = IFFT(|F|^2) which gives the circular autocorrelation.
            # For size_sum > 2*max_u, the circular autocorrelation equals the linear autocorrelation
            # for lags in [-size_sum/2, size_sum/2), which covers [-max_u, max_u].
            
            # conv_sum[k] = sum_x f[x] * f[(x-k) mod size_sum]
            # For |k| <= max_u and size_sum > 2*max_u, this equals the linear autocorrelation.
            
            # The autocorrelation at lag 0 is conv_sum[0] = n (number of elements)
            # For lag k > 0, it's conv_sum[k]
            # For lag k < 0, it's conv_sum[size_sum + k] = conv_sum[-k] by symmetry
            
            # So |U-U| = 1 + 2 * (number of k in 1..max_u where conv_sum[k] > 0.5)
            # But wait, we also need to check that conv_sum[k] represents the actual count.
            
            n_diff = 1 + 2 * int(np.sum(conv_sum[1:max_u+1] > 0.5))
            
            # Double-check: for a Golomb ruler, |U-U| = n*(n-1)+1
            # Let's verify with a simple case
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if len(cand) < 2:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    def verify_score(U):
        """Verify score using naive computation for debugging."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        if len(U) < 2:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def is_golomb(u):
        """Check if u is a Golomb ruler (all pairwise differences distinct)."""
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def local_search(base_u, time_limit):
        """Local search: try adding, removing, or swapping elements while maintaining Golomb property."""
        start_time = time.time()
        best_u = list(base_u)
        best_score_local = compute_score(best_u)
        improved = True
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            best_candidate = None
            best_candidate_score = best_score_local
            
            # Try adding a new element
            current_max = max(best_u)
            n = len(best_u)
            # Wider search range for larger sets
            search_hi = current_max + n * n * 3
            for new_mark in range(current_max + 1, search_hi + 1):
                candidate = sorted(best_u + [new_mark])
                if not is_golomb(candidate):
                    continue
                s = compute_score(candidate)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = candidate
                if (time.time() - start_time) > time_limit:
                    break
            
            # Try removing an element (if size > 3)
            if best_u and len(best_u) > 3 and (time.time() - start_time) < time_limit:
                for i in range(1, len(best_u)):  # keep 0
                    candidate = [x for j, x in enumerate(best_u) if j != i]
                    if not is_golomb(candidate):
                        continue
                    s = compute_score(candidate)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate
                    if (time.time() - start_time) > time_limit:
                        break
            
            # Try swapping: replace one element with another
            if best_u and (time.time() - start_time) < time_limit:
                for i in range(1, len(best_u)):  # don't swap 0
                    old_val = best_u[i]
                    # Try replacing with nearby values - wider range for larger sets
                    for delta in range(1, max(15, len(best_u) * 5)):
                        for new_val in [old_val - delta, old_val + delta]:
                            if new_val < 0:
                                continue
                            if new_val in best_u:
                                continue
                            candidate = sorted([x if x != old_val else new_val for x in best_u])
                            if not is_golomb(candidate):
                                continue
                            s = compute_score(candidate)
                            if s > best_candidate_score:
                                best_candidate_score = s
                                best_candidate = candidate
                        if (time.time() - start_time) > time_limit:
                            break
            
            if best_candidate is not None:
                best_u = best_candidate
                best_score_local = best_candidate_score
                improved = True
        
        return best_u, best_score_local

    # Known verified Golomb rulers (optimal or near-optimal)
    known_golomb = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 10, 12, 17, 20, 22],
        [0, 1, 4, 10, 12, 17, 20, 22, 24],
        [0, 1, 4, 10, 12, 17, 20, 22, 24, 26],
    ]
    
    # First, evaluate known Golomb rulers
    for gu in known_golomb:
        try_set(gu)
    
    # Verify FFT computation against naive for a test case
    test_u = [0, 1, 4, 10, 12, 17]
    fft_score = compute_score(test_u)
    naive_score = verify_score(test_u)
    if abs(fft_score - naive_score) > 1e-6:
        # FFT has a bug; fall back to naive for all computations
        def compute_score(U):
            return verify_score(U)
    
    # Run local search from multiple starting points
    start_points = [
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 10, 12, 17, 20, 22],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 6],
    ]
    
    for sp in start_points:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        # Allocate time proportionally - give more time to larger starting sets
        time_for_this = min(remaining * 0.3, 40)
        if time_for_this < 5:
            time_for_this = max(5, remaining * 0.5)
        ext_u, ext_score = local_search(sp, time_for_this)
        if ext_u and len(ext_u) >= 3:
            try_set(ext_u)
    
    # Exhaustive search for small sets to ensure we don't miss anything
    max_n = 6
    max_m = 30
    
    for n in range(2, max_n + 1):
        if time.time() > deadline - 15:
            break
        # Choose n-1 elements from 1..max_m
        for combo in combinations(range(1, max_m + 1), n - 1):
            if time.time() > deadline - 15:
                break
            cand = [0] + list(combo)
            try_set(cand)
    
    # Additional local search from the best set found so far, with extended time
    remaining = deadline - time.time()
    if remaining > 10 and len(best_set) >= 3:
        ext_u, ext_score = local_search(best_set, remaining * 0.7)
        if ext_u and len(ext_u) >= 3:
            try_set(ext_u)
    
    return list(best_set)
