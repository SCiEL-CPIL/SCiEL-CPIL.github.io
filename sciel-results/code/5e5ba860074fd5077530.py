"""Optimized program: SA with FFT score evaluation."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from math import log

MAX_U_LIMIT = 10000
# FFT size must be at least 2 * MAX_U_LIMIT + 1
# 20001 -> 2^15 = 32768
FFT_SIZE = 32768

def _compute_score_fft(U_bool, max_u):
    """
    Compute score using FFT.
    U_bool: boolean array of length FFT_SIZE, where U_bool[i]=1 if i in U.
    max_u: maximum element in U.
    """
    if max_u == 0:
        # U = {0}
        # Sumset {0}, Diffset {0}
        # Ratio 1/1 = 1. Log(1)=0. Score 1.
        return 1.0
        
    if not np.any(U_bool[:max_u+1]):
        return 0.0

    # Sumset
    F = np.fft.fft(U_bool)
    Conv = np.fft.ifft(F * F).real
    # Sumset indices 0..2*max_u
    # We check if count > 0
    # Using > 1e-9 to handle float precision
    sum_count = np.sum(Conv[:2*max_u+1] > 1e-9)
    
    # Diffset
    # Construct V' indicator array.
    # V'[i] = 1 if i in V'.
    # i in V' iff i = max_u - u for some u in U.
    # iff u = max_u - i in U.
    # So V'[i] = U_bool[max_u - i].
    # Valid i are 0..max_u.
    
    V_prime = np.zeros(FFT_SIZE)
    # Vectorized assignment
    V_prime[0:max_u+1] = U_bool[max_u::-1]
    
    F_v = np.fft.fft(V_prime)
    Conv_diff = np.fft.ifft(F * F_v).real
    # Diffset indices 0..2*max_u
    diff_count = np.sum(Conv_diff[:2*max_u+1] > 1e-9)
    
    if sum_count == 0:
        return 0.0
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
    return 1 + log(ratio) / denominator

def _sa_search(U_init, timeout=10.0):
    """Simulated Annealing search."""
    start_time = time.time()
    
    # Initialize U_bool
    U_bool = np.zeros(FFT_SIZE, dtype=np.float64)
    U_set = set(U_init)
    if 0 not in U_set:
        U_set.add(0)
    for x in U_set:
        if x < FFT_SIZE:
            U_bool[x] = 1.0
        else:
            pass
            
    max_u = max(U_set) if U_set else 0
    current_score = _compute_score_fft(U_bool, max_u)
    
    best_U = U_bool.copy()
    best_score = current_score
    best_max_u = max_u
    
    # SA Parameters
    T = 0.1
    cooling = 0.999
    min_T = 1e-4
    
    while time.time() - start_time < timeout:
        if T < min_T:
            T = min_T
            
        # Perturbation
        # 1. Add
        # 2. Remove
        # 3. Move
        op = random.randint(0, 2)
        
        old_max_u = max_u
        changes = []
        
        if op == 0: # Add
            x = random.randint(0, MAX_U_LIMIT)
            if U_bool[x] == 0:
                U_bool[x] = 1.0
                changes.append((x, 0.0))
                if x > max_u:
                    max_u = x
        elif op == 1: # Remove
            # Get list of elements
            elements = np.where(U_bool == 1.0)[0]
            # Ensure we don't remove 0
            elements = elements[elements > 0]
            if len(elements) == 0:
                continue
            x = elements[random.randint(0, len(elements)-1)]
            U_bool[x] = 0.0
            changes.append((x, 1.0))
            if x == max_u:
                # Find new max
                new_elements = np.where(U_bool == 1.0)[0]
                if len(new_elements) > 0:
                    max_u = new_elements[-1]
                else:
                    max_u = 0
        else: # Move
            elements = np.where(U_bool == 1.0)[0]
            elements = elements[elements > 0]
            if len(elements) == 0:
                continue
            x = elements[random.randint(0, len(elements)-1)]
            y = random.randint(0, MAX_U_LIMIT)
            if x == y:
                continue
            U_bool[x] = 0.0
            changes.append((x, 1.0))
            if U_bool[y] == 0:
                U_bool[y] = 1.0
                changes.append((y, 0.0))
                if y > max_u:
                    max_u = y
            
            if x == max_u:
                 # Recalculate max
                 new_elements = np.where(U_bool == 1.0)[0]
                 if len(new_elements) > 0:
                     max_u = new_elements[-1]
                 else:
                     max_u = 0
                     
        new_score = _compute_score_fft(U_bool, max_u)
        
        # Acceptance
        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = U_bool.copy()
                best_max_u = max_u
        else:
            # Revert
            for idx, old_val in changes:
                U_bool[idx] = old_val
            max_u = old_max_u
            
        T *= cooling
        
    # Reconstruct set from best_U
    best_elements = np.where(best_U == 1.0)[0]
    return list(best_elements), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Initial sets
    initial_sets = []
    
    # Geometric
    for base in [2, 3, 5]:
        s = [0]
        v = 1
        while v <= MAX_U_LIMIT:
            s.append(v)
            v *= base
        initial_sets.append(s)
        
    # Random
    for _ in range(5):
        n = random.randint(20, 100)
        rand_set = {0}
        while len(rand_set) < n:
            rand_set.add(random.randint(0, MAX_U_LIMIT))
        initial_sets.append(list(rand_set))
        
    # Dense low
    initial_sets.append(list(range(50)))
    
    best_result = [0]
    best_score = 0.0
    
    for s in initial_sets:
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
        # Run SA
        res, score = _sa_search(s, timeout=min(10.0, remaining))
        if score > best_score:
            best_score = score
            best_result = res
            
    # Final check
    if not best_result or 0 not in best_result:
        best_result = [0]
        
    return best_result
# EVOLVE-BLOCK-END
