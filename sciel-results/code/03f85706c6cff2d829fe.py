"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _max_abs_ap_sum(signs):
    """
    Compute the maximum absolute signed sum over ALL arithmetic progressions
    contained in [1, n]. Returns (max_val, argmax_info).
    
    For each step d >= 1 and each residue class r mod d (r in [0, d)),
    the APs with step d starting at positions r, r+d, r+2d, ... are contiguous
    subsequences of the sequence signs[r], signs[r+d], signs[r+2d], ...
    So we need the max absolute subarray sum of each such subsequence.
    """
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_disc = 1  # singletons always give |sum| = 1
    best_info = (0, 1, 1)  # (start, step, length) of worst AP
    
    # For each step d from 1 to n-1
    for d in range(1, n):
        for r in range(d):
            # Extract the subsequence: indices r, r+d, r+2d, ... < n
            # Using numpy slicing for speed
            seq = signs[r::d]
            if len(seq) <= 1:
                continue
            
            # Compute max absolute subarray sum using Kadane's variant
            # We need max over all contiguous subarrays of |sum|
            # This is max(max_subarray_sum, -min_subarray_sum)
            
            # Max subarray sum
            max_ending_here = 0
            max_so_far = 0
            min_ending_here = 0
            min_so_far = 0
            
            for val in seq:
                max_ending_here = max(0, max_ending_here + val)
                max_so_far = max(max_so_far, max_ending_here)
                min_ending_here = min(0, min_ending_here + val)
                min_so_far = min(min_so_far, min_ending_here)
            
            candidate = max(max_so_far, -min_so_far)
            if candidate > max_disc:
                max_disc = candidate
                # We don't track exact AP for speed in evaluation, but for search we might
                # For now just return the max value
    
    return max_disc

def _max_abs_ap_sum_with_argmax(signs):
    """
    Compute max abs AP sum and also return info about which AP achieves it.
    Returns (max_val, (start, step, length, residue)) where start is 1-indexed.
    """
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_disc = 1
    best = (1, 1, 1)  # (start_1indexed, step, length)
    
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) <= 1:
                continue
            
            # Find max absolute subarray sum with position tracking
            # Max subarray
            max_ending = 0
            max_so_far = 0
            max_start_idx = 0
            max_end_idx = 0
            cur_start = 0
            
            min_ending = 0
            min_so_far = 0
            min_start_idx = 0
            min_end_idx = 0
            cur_min_start = 0
            
            for i, val in enumerate(seq):
                if max_ending + val < 0:
                    max_ending = 0
                    cur_start = i + 1
                else:
                    max_ending += val
                if max_ending > max_so_far:
                    max_so_far = max_ending
                    max_start_idx = cur_start
                    max_end_idx = i
                
                if min_ending + val > 0:
                    min_ending = 0
                    cur_min_start = i + 1
                else:
                    min_ending += val
                if min_ending < min_so_far:
                    min_so_far = min_ending
                    min_start_idx = cur_min_start
                    min_end_idx = i
            
            # Max positive subarray
            if max_so_far > max_disc:
                max_disc = max_so_far
                # seq index max_start_idx corresponds to original index r + max_start_idx * d
                # 1-indexed start: r + max_start_idx * d + 1
                length = max_end_idx - max_start_idx + 1
                best = (r + max_start_idx * d + 1, d, length)
            
            # Max negative subarray (absolute value)
            if -min_so_far > max_disc:
                max_disc = -min_so_far
                length = min_end_idx - min_start_idx + 1
                best = (r + min_start_idx * d + 1, d, length)
    
    return max_disc, best

def _greedy_construct():
    """
    Greedy construction: build the sign sequence one position at a time,
    choosing the sign that minimizes the worst AP discrepancy involving
    the new position.
    """
    signs = [0] * _N
    rng = random.Random(42)
    
    for i in range(_N):
        # Try both signs
        best_sign = 1
        best_disc = float('inf')
        
        for s in (1, -1):
            signs[i] = s
            # Evaluate only APs that include position i (0-indexed)
            # This is a partial evaluation for speed
            disc = 0
            for d in range(1, _N):
                for r in range(d):
                    if (i - r) % d != 0 or i < r:
                        continue
                    # Position i is in the residue class r mod d
                    seq_start = r
                    seq_indices = list(range(r, _N, d))
                    # Find where i is in this sequence
                    pos_in_seq = (i - r) // d
                    if pos_in_seq < 0 or pos_in_seq >= len(seq_indices):
                        continue
                    # Compute sum of this full AP
                    s_val = 0
                    for j in seq_indices:
                        if j < _N:
                            s_val += signs[j]
                    disc = max(disc, abs(s_val))
                    if disc >= best_disc:
                        break
                if disc >= best_disc:
                    break
            
            if disc < best_disc:
                best_disc = disc
                best_sign = s
        
        signs[i] = best_sign
    
    return signs

def _local_search(signs, max_time=30.0):
    """
    Local search: repeatedly find the worst AP, then try flipping signs
    in that AP to reduce the discrepancy.
    """
    signs = list(signs)
    best_disc = _max_abs_ap_sum(signs)
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        disc, (start, step, length) = _max_abs_ap_sum_with_argmax(signs)
        if disc == 1:
            break
        
        # Try flipping each sign in the worst AP
        improved = False
        indices = [start - 1 + k * step for k in range(length) if start - 1 + k * step < _N]
        
        for idx in indices:
            old_sign = signs[idx]
            signs[idx] = -old_sign
            new_disc = _max_abs_ap_sum(signs)
            if new_disc < best_disc:
                best_disc = new_disc
                improved = True
            else:
                signs[idx] = old_sign
        
        if not improved:
            # Try random flips to escape local minima
            for _ in range(5):
                idx = random.randint(0, _N - 1)
                signs[idx] = -signs[idx]
                new_disc = _max_abs_ap_sum(signs)
                if new_disc < best_disc:
                    best_disc = new_disc
                    improved = True
                else:
                    signs[idx] = -signs[idx]
            if not improved:
                break
    
    return signs, best_disc

def _simulated_annealing(signs, max_time=20.0):
    """
    Simulated annealing to escape local minima.
    """
    signs = list(signs)
    current_disc = _max_abs_ap_sum(signs)
    best_disc = current_disc
    best_signs = list(signs)
    
    start_time = time.time()
    temp = 5.0
    cooling = 0.999
    
    while time.time() - start_time < max_time and temp > 0.1:
        # Pick a random position to flip
        idx = random.randint(0, _N - 1)
        signs[idx] = -signs[idx]
        new_disc = _max_abs_ap_sum(signs)
        
        delta = new_disc - current_disc
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = list(signs)
        else:
            signs[idx] = -signs[idx]
        
        temp *= cooling
    
    return best_signs, best_disc

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    signs = _greedy_construct()
    disc = _max_abs_ap_sum(signs)
    
    # Phase 2: Local search
    if time.time() - start_time < _DEADLINE - 30:
        signs, disc = _local_search(signs, max_time=_DEADLINE - 35 - (time.time() - start_time))
    
    # Phase 3: Simulated annealing
    if time.time() - start_time < _DEADLINE - 5:
        sa_time = min(15.0, _DEADLINE - 5 - (time.time() - start_time))
        if sa_time > 1:
            signs2, disc2 = _simulated_annealing(signs, max_time=sa_time)
            if disc2 < disc:
                signs = signs2
                disc = disc2
    
    return signs

# EVOLVE-BLOCK-END
