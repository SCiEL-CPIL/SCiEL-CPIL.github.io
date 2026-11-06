import numpy as np
import time
import random

_N = 256
_AP_A = []
_AP_B = []
_AP_C = []
_AP_D = []

for d in range(1, _N // 3 + 1):
    for a in range(_N - 3 * d):
        _AP_A.append(a)
        _AP_B.append(a + d)
        _AP_C.append(a + 2 * d)
        _AP_D.append(a + 3 * d)

_AP_A = np.array(_AP_A, dtype=np.int32)
_AP_B = np.array(_AP_B, dtype=np.int32)
_AP_C = np.array(_AP_C, dtype=np.int32)
_AP_D = np.array(_AP_D, dtype=np.int32)

def _score(arr):
    # arr is numpy array of 0/1
    m = (arr[_AP_A] == arr[_AP_B]) & (arr[_AP_B] == arr[_AP_C]) & (arr[_AP_C] == arr[_AP_D])
    return np.sum(m)

def construct():
    start = time.time()
    deadline = start + 115.0 # 5s safety margin
    
    # Init Thue-Morse
    arr = np.array([bin(i).count('1') % 2 for i in range(_N)], dtype=np.int8)
    current_score = _score(arr)
    best_score = current_score
    best_arr = arr.copy()
    
    # SA Parameters
    T = 5.0
    cooling = 0.99995 # Slow cooling to allow more steps
    # Or fixed number of steps?
    # Let's use time-based.
    
    while time.time() < deadline - 10.0:
        i = random.randint(0, _N - 1)
        arr[i] = 1 - arr[i]
        new_score = _score(arr)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_arr = arr.copy()
        else:
            arr[i] = 1 - arr[i]
            
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    # Final Hill Climbing
    arr = best_arr.copy()
    current_score = best_score
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Random order to avoid bias
        indices = list(range(_N))
        random.shuffle(indices)
        for i in indices:
            arr[i] = 1 - arr[i]
            new_score = _score(arr)
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_arr = arr.copy()
            else:
                arr[i] = 1 - arr[i]
                
    return best_arr.tolist()
