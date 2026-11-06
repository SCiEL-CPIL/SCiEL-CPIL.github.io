"""
Optimized solver for minimizing monochromatic 4-term arithmetic progressions.
Uses delta-update local search with Simulated Annealing.
"""
import numpy as np
import time
import random

_N = 256
_MAX_D = _N // 3 + 1

# Precompute all 4-term APs
# APs are stored as tuples of 4 indices (0-based)
_AP_LIST = []
# Map from index to list of (ap_index, position_in_ap)
_INDEX_TO_APS = [[] for _ in range(_N)]

_ap_count = 0
for d in range(1, _MAX_D):
    for a in range(_N - 3 * d):
        # Indices are 0-based: a, a+d, a+2d, a+3d
        i0 = a
        i1 = a + d
        i2 = a + 2 * d
        i3 = a + 3 * d
        
        _AP_LIST.append((i0, i1, i2, i3))
        _ap_count += 1
        
        _INDEX_TO_APS[i0].append((_ap_count - 1, 0))
        _INDEX_TO_APS[i1].append((_ap_count - 1, 1))
        _INDEX_TO_APS[i2].append((_ap_count - 1, 2))
        _INDEX_TO_APS[i3].append((_ap_count - 1, 3))

# Convert to numpy arrays for faster access if needed, but lists are fine for small k
# Let's convert AP list to a 2D numpy array for vectorized mono-checking if we do full recalc, 
# but for delta updates, we access individual elements.
_AP_ARRAY = np.array(_AP_LIST, dtype=np.int32) # Shape (N_APS, 4)
_N_APS = len(_AP_LIST)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (0-indexed)."""
    # t(n) = popcount(n) % 2
    return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

def _initial_score(arr):
    """Calculate initial score and mono status array."""
    # arr is 0/1 numpy array
    # APs are in _AP_ARRAY
    # Extract colors for all APs: shape (N_APS, 4)
    cols = arr[_AP_ARRAY]
    # Check if all 4 are equal
    # c0 == c1 == c2 == c3
    mono = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
    return np.sum(mono), mono

def _is_mono_at(arr, ap_idx):
    """Check if the AP at ap_idx is monochromatic given current arr."""
    ap = _AP_ARRAY[ap_idx]
    c0 = arr[ap[0]]
    c1 = arr[ap[1]]
    c2 = arr[ap[2]]
    c3 = arr[ap[3]]
    return c0 == c1 == c2 == c3

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def time_left():
        return deadline - time.time()

    # 1. Initialization: Thue-Morse is a strong candidate
    best_arr = _thue_morse(_N).copy()
    best_score, best_mono = _initial_score(best_arr)
    
    # Also try a few random starts and block patterns to see if they beat TM
    candidates = [best_arr]
    for _ in range(10):
        candidates.append(np.random.randint(0, 2, _N, dtype=np.int8))
    for bs in [2, 4, 8, 16]:
        candidates.append(np.array([(i // bs) % 2 for i in range(_N)], dtype=np.int8))
        
    for cand in candidates:
        s, m = _initial_score(cand)
        if s < best_score:
            best_score = s
            best_arr = cand.copy()
            best_mono = m.copy()

    # 2. Simulated Annealing with Delta Updates
    # State: current_arr, current_score, current_mono
    cur_arr = best_arr.copy()
    cur_score = best_score
    cur_mono = best_mono.copy()
    
    T = 10.0
    cooling = 0.99995
    min_T = 0.01
    
    # Pre-compute lists for faster access in inner loop
    # _INDEX_TO_APS[i] is list of (ap_idx, pos)
    
    steps = 0
    while time_left() > 10.0:
        # Pick a random index to flip
        i = np.random.randint(0, _N)
        
        # Calculate delta score
        delta = 0
        # We need to check all APs containing i
        for ap_idx, pos in _INDEX_TO_APS[i]:
            # Check if AP was mono before
            was_mono = cur_mono[ap_idx]
            
            # Temporarily flip i in cur_arr? No, we can just check the other 3.
            # If pos is 0, other indices are 1,2,3.
            # Let's just flip in a temporary variable or logic.
            # Actually, flipping in cur_arr and checking is easiest but we need to revert if rejected.
            # Let's compute new mono status without modifying cur_arr first.
            
            ap = _AP_ARRAY[ap_idx]
            c_new = 1 - cur_arr[i]
            
            # Get other 3 colors
            # This is a bit inefficient to loop, but k is small.
            # c0, c1, c2, c3 are cur_arr[ap[0]] etc.
            # One of them is i.
            
            # Optimization: 
            # If was_mono is False:
            #   It becomes mono only if the other 3 are equal to c_new.
            # If was_mono is True:
            #   It becomes non-mono (delta +1) because flipping one in a mono set breaks it.
            
            if was_mono:
                # Flipping one element in a mono AP makes it non-mono
                delta -= 1
            else:
                # Check if flipping i makes it mono
                # The other 3 must be equal to each other AND equal to c_new
                # Get indices of the other 3
                # ap is (i0, i1, i2, i3)
                # pos is which one is i
                
                # Extract other indices
                # Simple unroll for speed
                if pos == 0:
                    c1 = cur_arr[ap[1]]
                    c2 = cur_arr[ap[2]]
                    c3 = cur_arr[ap[3]]
                    if c1 == c2 == c3 == c_new:
                        delta += 1
                elif pos == 1:
                    c0 = cur_arr[ap[0]]
                    c2 = cur_arr[ap[2]]
                    c3 = cur_arr[ap[3]]
                    if c0 == c2 == c3 == c_new:
                        delta += 1
                elif pos == 2:
                    c0 = cur_arr[ap[0]]
                    c1 = cur_arr[ap[1]]
                    c3 = cur_arr[ap[3]]
                    if c0 == c1 == c3 == c_new:
                        delta += 1
                else: # pos == 3
                    c0 = cur_arr[ap[0]]
                    c1 = cur_arr[ap[1]]
                    c2 = cur_arr[ap[2]]
                    if c0 == c1 == c2 == c_new:
                        delta += 1

        new_score = cur_score + delta
        
        # Acceptance criterion
        if delta < 0 or random.random() < np.exp(-delta / max(T, min_T)):
            # Accept move
            # Update state
            cur_arr[i] = 1 - cur_arr[i]
            cur_score = new_score
            
            # Update mono statuses for affected APs
            for ap_idx, pos in _INDEX_TO_APS[i]:
                cur_mono[ap_idx] = _is_mono_at(cur_arr, ap_idx)
                
            # Update global best
            if cur_score < best_score:
                best_score = cur_score
                best_arr = cur_arr.copy()
                best_mono = cur_mono.copy()
            
            steps += 1
        else:
            # Reject move (state unchanged)
            pass
            
        # Cooling
        T *= cooling
        if T < min_T:
            T = min_T

    # 3. Final Hill Climbing (Strict Descent)
    # Use the best found so far
    cur_arr = best_arr.copy()
    cur_score = best_score
    cur_mono = best_mono.copy()
    
    improved = True
    while improved and time_left() > 1.0:
        improved = False
        # Try flipping each index once
        indices = np.random.permutation(_N)
        for i in indices:
            delta = 0
            for ap_idx, pos in _INDEX_TO_APS[i]:
                was_mono = cur_mono[ap_idx]
                if was_mono:
                    delta -= 1
                else:
                    ap = _AP_ARRAY[ap_idx]
                    c_new = 1 - cur_arr[i]
                    if pos == 0:
                        if cur_arr[ap[1]] == cur_arr[ap[2]] == cur_arr[ap[3]] == c_new:
                            delta += 1
                    elif pos == 1:
                        if cur_arr[ap[0]] == cur_arr[ap[2]] == cur_arr[ap[3]] == c_new:
                            delta += 1
                    elif pos == 2:
                        if cur_arr[ap[0]] == cur_arr[ap[1]] == cur_arr[ap[3]] == c_new:
                            delta += 1
                    else:
                        if cur_arr[ap[0]] == cur_arr[ap[1]] == cur_arr[ap[2]] == c_new:
                            delta += 1
                            
            if delta < 0:
                # Apply
                cur_arr[i] = 1 - cur_arr[i]
                cur_score += delta
                for ap_idx, pos in _INDEX_TO_APS[i]:
                    cur_mono[ap_idx] = _is_mono_at(cur_arr, ap_idx)
                improved = True
                if cur_score < best_score:
                    best_score = cur_score
                    best_arr = cur_arr.copy()
                    best_mono = cur_mono.copy()

    return best_arr.tolist()

# EVOLVE-BLOCK-END
