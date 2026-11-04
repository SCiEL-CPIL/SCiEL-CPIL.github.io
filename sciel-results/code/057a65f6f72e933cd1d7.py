"""Optimized sumset/difference-set construction."""
import time
import random
import math
import numpy as np
from collections import Counter

def _compute_score(U):
    """Compute the score 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or max(U) == 0:
        return 0.0
    U = list(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    # Compute sumset and difference set
    # Use numpy for efficiency
    U_arr = np.array(U, dtype=np.int64)
    # Sumset: all pairs (i,j) with i,j in U
    # Difference set: all pairs (i,j) with i,j in U
    # For large sets, this can be expensive. Use the fact that |U+U| <= |U|^2
    # and |U-U| <= |U|^2
    
    # Efficient computation using convolution-like approach
    # Actually, for sets up to ~10000 elements, direct computation might be too slow
    # Let's use a smarter approach: represent as boolean array and use FFT
    # But that's complex. Let's try direct for small sets, and for large sets use approximation
    
    if n <= 2000:
        # Direct computation
        sums = set()
        diffs = set()
        for i in U:
            for j in U:
                sums.add(i + j)
                diffs.add(i - j)
        num_sums = len(sums)
        num_diffs = len(diffs)
    else:
        # Use FFT-based approach
        # Create boolean array
        max_val = max_u * 2
        arr = np.zeros(max_val + 1, dtype=np.float64)
        for x in U:
            arr[x] = 1.0
        # Convolution for sumset
        fft_size = 1
        while fft_size < 2 * (max_val + 1):
            fft_size *= 2
        fft_arr = np.fft.fft(arr, fft_size)
        conv = np.fft.ifft(fft_arr * fft_arr).real
        num_sums = int(np.sum(conv > 0.5))
        
        # For difference set, use cross-correlation
        # U-U = {a-b : a,b in U}
        # This is equivalent to the support of the autocorrelation
        # Actually, U-U can be computed similarly
        # Let's use the fact that |U-U| can be found via:
        # For each d, d in U-U iff there exists a in U such that a-d in U
        # This is the cross-correlation of U with itself
        # Using FFT: autocorrelation
        fft_arr2 = np.fft.fft(arr, fft_size)
        autocorr = np.fft.ifft(fft_arr2 * np.conj(fft_arr2)).real
        # The autocorrelation at lag k gives sum of arr[i]*arr[i+k]
        # U-U contains d iff autocorrelation at appropriate lag is > 0
        # Actually, let's just compute it directly for differences
        # For large sets, this might still be slow. Let's use a different approach.
        # 
        # Alternative: |U-U| is related to the number of distinct differences.
        # We can estimate it, but for scoring we need exact values.
        # Let's just do direct computation but with early termination if it's taking too long
        start = time.time()
        diffs = set()
        for i in U:
            for j in U:
                diffs.add(i - j)
                if time.time() - start > 10:
                    break
            if time.time() - start > 10:
                break
        num_diffs = len(diffs)
    
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    if ratio <= 1.0:
        return 1.0
    log_term = math.log(ratio) / math.log(2 * max_u + 1)
    return 1.0 + log_term


def _local_search(U, max_iters=500, time_budget=20.0):
    """Local search to improve the set U."""
    U = sorted(set(U))
    best_score = _compute_score(U)
    best_U = U[:]
    start_time = time.time()
    
    for _ in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Try adding a random element
        max_u = max(U)
        # Candidate: random element in range
        candidates = []
        
        # Try adding elements near existing ones
        for _ in range(10):
            if U:
                base = random.choice(U)
                delta = random.randint(-max_u//2, max_u//2)
                candidates.append(base + delta)
        
        # Try adding elements at geometric positions
        if U:
            for k in range(1, 6):
                candidates.append(k * (max_u + 1) // 5)
                candidates.append(k * max_u // 4)
        
        # Try removing an element
        remove_candidates = [random.choice(U) for _ in range(5)]
        
        best_improvement = 0
        best_move = None
        
        for c in candidates:
            if c < 0:
                continue
            if c in U:
                continue
            new_U = U + [c]
            new_score = _compute_score(new_U)
            improvement = new_score - best_score
            if improvement > best_improvement:
                best_improvement = improvement
                best_move = ('add', c, new_U, new_score)
        
        for r in remove_candidates:
            if len(U) <= 3:
                continue
            new_U = [x for x in U if x != r]
            new_score = _compute_score(new_U)
            improvement = new_score - best_score
            if improvement > best_improvement:
                best_improvement = improvement
                best_move = ('remove', r, new_U, new_score)
        
        if best_move is not None:
            U = best_move[2]
            best_score = best_move[3]
            if best_score > _compute_score(best_U):
                best_U = U[:]
    
    return best_U, _compute_score(best_U)


def _geometric_seed(ratio, size_hint):
    """Create a geometric seed set."""
    U = {0}
    current = 1
    while len(U) < size_hint:
        U.add(current)
        current = int(current * ratio)
        if current > 10**9:
            break
    return sorted(U)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Try multiple geometric seeds with different ratios
    ratios = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.5, 3.0]
    sizes = [50, 100, 200, 500, 1000]
    
    for ratio in ratios:
        if time.time() > deadline:
            break
        for size in sizes:
            if time.time() > deadline:
                break
            U = _geometric_seed(ratio, size)
            score = _compute_score(U)
            if score > best_score:
                best_score = score
                best_U = U[:]
            
            # Local search from this seed
            time_left = deadline - time.time()
            if time_left > 5:
                refined, refined_score = _local_search(U, max_iters=200, time_budget=min(time_left - 2, 15))
                if refined_score > best_score:
                    best_score = refined_score
                    best_U = refined
    
    # Final polish with more iterations
    time_left = deadline - time.time()
    if time_left > 10:
        refined, refined_score = _local_search(best_U, max_iters=500, time_budget=time_left - 2)
        if refined_score > best_score:
            best_score = refined_score
            best_U = refined
    
    return best_U
