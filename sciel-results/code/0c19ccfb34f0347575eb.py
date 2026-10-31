import numpy as np
import time
import random

def compute_score(U):
    if not U or 0 not in U:
        return -np.inf
    max_u = max(U)
    if max_u == 0:
        return 1.0
        
    # Sumset
    # Use FFT
    fft_size = 1
    while fft_size < 2 * max_u + 1:
        fft_size *= 2
        
    arr = np.zeros(fft_size, dtype=np.float64)
    for u in U:
        arr[u] = 1.0
        
    fft_arr = np.fft.fft(arr)
    fft_sum = fft_arr * fft_arr
    sumset_counts = np.fft.ifft(fft_sum).real
    sumset_size = np.sum(sumset_counts > 0.5)
    
    # Difference set
    # O(N^2)
    diffs = set()
    for u1 in U:
        for u2 in U:
            diffs.add(u1 - u2)
    diffset_size = len(diffs)
    
    if sumset_size == 0:
        return -np.inf
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return -np.inf
        
    denom = np.log(2 * max_u + 1)
    score = 1 + np.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Initialize with a good starting point
    # Powers of 3 seem promising
    best_set = [0, 1, 3, 9, 27, 81]
    best_score = compute_score(best_set)
    
    current_set = set(best_set)
    current_score = best_score
    
    temperature = 0.5
    cooling_rate = 0.995
    
    while time.time() < deadline:
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            max_u = max(current_set) if current_set else 0
            limit = max(100, 3 * max_u)
            new_val = random.randint(1, limit)
            if new_val not in current_set:
                new_set = current_set | {new_val}
            else:
                continue
        elif move_type == 'remove':
            if len(current_set) <= 1:
                continue
            candidates = [x for x in current_set if x != 0]
            if not candidates:
                continue
            rem_val = random.choice(candidates)
            new_set = current_set - {rem_val}
        else:
            if len(current_set) <= 1:
                continue
            candidates = [x for x in current_set if x != 0]
            if not candidates:
                continue
            rem_val = random.choice(candidates)
            max_u = max(current_set)
            limit = max(100, 3 * max_u)
            new_val = random.randint(1, limit)
            if new_val not in current_set:
                new_set = (current_set - {rem_val}) | {new_val}
            else:
                continue
        
        new_score = compute_score(list(new_set))
        
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            delta = new_score - current_score
            if delta < 0 and random.random() < np.exp(delta / temperature):
                current_set = new_set
                current_score = new_score
        
        temperature *= cooling_rate
        
    return best_set
