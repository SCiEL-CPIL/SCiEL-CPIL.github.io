"""
Iterated Local Search with Vectorized Multi-Flip Perturbation.
Strategy: 
1. Use vectorized NumPy operations to compute the exact score change when flipping a set of bits.
2. Perform large jumps (multi-flip) to escape local optima.
3. Perform strict descent (single-flip) to converge.
4. Repeat.
"""
import numpy as np
import time
import random

_N = 256
_MAX_D = _N // 3 + 1

# Precompute all 4-term APs
_AP_LIST = []
# Map from index to list of (ap_index, position_in_ap)
_INDEX_TO_APS = [[] for _ in range(_N)]

_ap_count = 0
for d in range(1, _MAX_D):
    for a in range(_N - 3 * d):
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

_AP_ARRAY = np.array(_AP_LIST, dtype=np.int32) # Shape (N_APS, 4)
_N_APS = len(_AP_LIST)

def _thue_morse(n):
    return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

def _calculate_score(arr):
    """Calculate the exact number of monochromatic 4-APs."""
    cols = arr[_AP_ARRAY]
    mono = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
    return np.sum(mono)

def _descend(arr, time_limit):
    """
    Perform strict single-flip descent until no improvement is found or time limit is reached.
    Returns the improved array and its score.
    """
    cur_arr = arr.copy()
    cur_score = _calculate_score(cur_arr)
    
    # Precompute lists for faster access in inner loop
    # _INDEX_TO_APS[i] is list of (ap_idx, pos)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Try flipping each index once in random order
        indices = np.random.permutation(_N)
        for i in indices:
            delta = 0
            # Calculate delta score for flipping i
            for ap_idx, pos in _INDEX_TO_APS[i]:
                # Check if AP was mono before
                ap = _AP_ARRAY[ap_idx]
                c0 = cur_arr[ap[0]]
                c1 = cur_arr[ap[1]]
                c2 = cur_arr[ap[2]]
                c3 = cur_arr[ap[3]]
                was_mono = (c0 == c1 == c2 == c3)
                
                if was_mono:
                    # Flipping one element in a mono AP makes it non-mono
                    delta -= 1
                else:
                    # Check if flipping i makes it mono
                    c_new = 1 - cur_arr[i]
                    if pos == 0:
                        if c1 == c2 == c3 == c_new:
                            delta += 1
                    elif pos == 1:
                        if c0 == c2 == c3 == c_new:
                            delta += 1
                    elif pos == 2:
                        if c0 == c1 == c3 == c_new:
                            delta += 1
                    else:
                        if c0 == c1 == c2 == c_new:
                            delta += 1
                            
            if delta < 0:
                # Apply flip
                cur_arr[i] = 1 - cur_arr[i]
                cur_score += delta
                improved = True
                if time.time() > time_limit:
                    break

    return cur_arr, cur_score

def _perturb(arr, n_flips):
    """
    Flip n_flips random bits in the array.
    """
    perturbed = arr.copy()
    indices = np.random.choice(_N, size=n_flips, replace=False)
    perturbed[indices] = 1 - perturbed[indices]
    return perturbed

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Initialization
    # Start with Thue-Morse, which is known to be good for avoiding short APs
    best_arr = _thue_morse(_N).copy()
    best_score = _calculate_score(best_arr)
    
    # Try a few other initializations to see if they start better
    candidates = [
        _thue_morse(_N),
        np.random.randint(0, 2, _N, dtype=np.int8),
        np.array([(i // 2) % 2 for i in range(_N)], dtype=np.int8),
        np.array([(i // 4) % 2 for i in range(_N)], dtype=np.int8),
        np.array([(i // 8) % 2 for i in range(_N)], dtype=np.int8),
    ]
    
    for cand in candidates:
        s = _calculate_score(cand)
        if s < best_score:
            best_score = s
            best_arr = cand.copy()
    
    # 2. Iterated Local Search
    # We will perform a series of perturbations and descents.
    # The key idea is that multi-flip perturbations can jump over barriers that single-flip descent cannot.
    
    current_arr = best_arr.copy()
    current_score = best_score
    
    # Parameters for perturbation
    # Start with smaller perturbations, increase as we get stuck
    base_n_flips = 5
    max_n_flips = 20
    
    # We'll use a cooling schedule for the perturbation size
    # Or simply increase it if we don't improve for a while
    
    no_improve_count = 0
    n_flips = base_n_flips
    
    while time.time() < deadline:
        # Descend from current state
        descended_arr, descended_score = _descend(current_arr, time.time() + 5.0)
        
        # Update best if improved
        if descended_score < best_score:
            best_score = descended_score
            best_arr = descended_arr.copy()
            no_improve_count = 0
            n_flips = base_n_flips  # Reset perturbation size
        else:
            no_improve_count += 1
            # Increase perturbation size if stuck
            if no_improve_count % 10 == 0:
                n_flips = min(n_flips + 2, max_n_flips)
        
        current_arr = descended_arr
        current_score = descended_score
        
        # If we have time for another iteration, perturb
        if time.time() < deadline - 10.0:
            # Perturb the current best (or current state?)
            # Perturbing the best state is a common ILS strategy (restart from best)
            # But perturbing the current state allows for more exploration
            # Let's perturb the current state, but if we've been stuck, perturb the best
            if no_improve_count > 5:
                source_arr = best_arr
            else:
                source_arr = current_arr
                
            current_arr = _perturb(source_arr, n_flips)
            current_score = _calculate_score(current_arr)
            
            # If the perturbation itself is better than best, update best
            if current_score < best_score:
                best_score = current_score
                best_arr = current_arr.copy()
                no_improve_count = 0
                n_flips = base_n_flips

    return best_arr.tolist()

# EVOLVE-BLOCK-END
