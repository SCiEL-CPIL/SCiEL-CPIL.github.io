import numpy as np
import random

N = 50

def greedy_sidon(n, max_restarts=1000):
    best_set = []
    for _ in range(max_restarts):
        indices = list(range(n))
        random.shuffle(indices)
        current_set = []
        current_sums = set()
        
        for idx in indices:
            # Check if adding idx preserves Sidon property
            valid = True
            new_sums = []
            for j in current_set:
                s = idx + j
                if s in current_sums:
                    valid = False
                    break
                new_sums.append(s)
            
            if not valid:
                continue
                
            s_self = idx + idx
            if s_self in current_sums:
                valid = False
            else:
                current_sums.update(new_sums)
                current_sums.add(s_self)
                current_set.append(idx)
                
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
    return best_set

def construct():
    # Find largest Sidon set using greedy random restarts
    S = greedy_sidon(N, max_restarts=500)
    if not S:
        S = [0]
        
    h = np.zeros(N)
    for i in S:
        h[i] = 1.0 / len(S)
        
    return list(h)

def compute_ratio(h):
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_peak = np.max(conv)
    if max_peak == 0:
        return 0.0
    return l2_sq / (l1 * max_peak)
