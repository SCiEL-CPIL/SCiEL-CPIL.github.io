import numpy as np
import time
import random

N = 18
TOTAL_INPUTS = 1 << N
DEADLINE = time.time() + 85.0

V_full = None
V_subset = None
subset_indices = None
all_candidates = None

def _init_V():
    indices = np.arange(TOTAL_INPUTS, dtype=np.uint32)
    V = (indices[:, None] >> np.arange(N, dtype=np.uint32)) & 1
    return V.astype(np.uint8)

def _setup():
    global V_full, V_subset, subset_indices, all_candidates
    if V_full is None:
        V_full = _init_V()
        subset_indices = np.random.choice(TOTAL_INPUTS, 1024, replace=False)
        V_subset = V_full[subset_indices]
        all_candidates = [(i, j) for i in range(N) for j in range(i+1, N)]

def _is_sorting_subset(net):
    V_curr = V_subset.copy()
    for i, j in net:
        vi = V_curr[:, i]
        vj = V_curr[:, j]
        V_curr[:, i] = np.minimum(vi, vj)
        V_curr[:, j] = np.maximum(vi, vj)
    diffs = np.diff(V_curr, axis=1)
    return np.all(diffs >= 0)

def _is_sorting_full(net):
    V_curr = V_full.copy()
    for i, j in net:
        vi = V_curr[:, i]
        vj = V_curr[:, j]
        V_curr[:, i] = np.minimum(vi, vj)
        V_curr[:, j] = np.maximum(vi, vj)
    diffs = np.diff(V_curr, axis=1)
    return np.all(diffs >= 0)

def construct():
    _setup()
    
    seed_net = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    current_net = seed_net
    best_net = seed_net
    best_count = len(seed_net)
    
    T = 2.0
    cooling = 0.999
    min_T = 0.1
    
    while time.time() < DEADLINE:
        move_type = random.random()
        new_net = current_net.copy()
        
        if move_type < 0.4 and len(current_net) > 0:
            idx = random.randint(0, len(current_net) - 1)
            new_net = current_net[:idx] + current_net[idx+1:]
        elif move_type < 0.7:
            cand = random.choice(all_candidates)
            pos = random.randint(0, len(current_net))
            new_net = current_net[:pos] + [cand] + current_net[pos:]
        else:
            if len(current_net) > 1:
                i, j = random.sample(range(len(current_net)), 2)
                new_net = current_net.copy()
                new_net[i], new_net[j] = new_net[j], new_net[i]
            else:
                cand = random.choice(all_candidates)
                pos = 0
                new_net = current_net[:pos] + [cand] + current_net[pos:]

        new_count = len(new_net)
        delta_E = new_count - len(current_net)
        
        if not _is_sorting_subset(new_net):
            continue
            
        if not _is_sorting_full(new_net):
            continue
            
        accept = False
        if delta_E < 0:
            accept = True
        elif delta_E == 0:
            accept = random.random() < 0.5
        else:
            if T > 0:
                prob = np.exp(-delta_E / T)
                accept = random.random() < prob
        
        if accept:
            current_net = new_net
            if new_count < best_count:
                best_count = new_count
                best_net = new_net
        
        T *= cooling
        if T < min_T:
            T = min_T

    return best_net
