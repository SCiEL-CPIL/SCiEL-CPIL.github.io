import numpy as np
import time
import random

N = 18
NUM_INPUTS = 1 << N

def _precompute_inputs():
    mat = np.zeros((N, NUM_INPUTS), dtype=np.uint8)
    for i in range(N):
        half = 1 << i
        period = 2 * half
        pattern = np.concatenate([np.zeros(half, dtype=np.uint8), np.ones(half, dtype=np.uint8)])
        mat[i] = np.tile(pattern, NUM_INPUTS // period)
    return mat

INPUTS = _precompute_inputs()

def _simulate(comparators):
    mat = INPUTS.copy()
    for (i, j) in comparators:
        mi = mat[i] & mat[j]
        mj = mat[i] | mat[j]
        mat[i] = mi
        mat[j] = mj
    return mat

def _is_valid(comparators):
    mat = _simulate(comparators)
    for i in range(N - 1):
        if np.any(mat[i] > mat[i + 1]):
            return False
    return True

def _greedy_remove(net, deadline):
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                return net
            trial = net[:idx] + net[idx+1:]
            if _is_valid(trial):
                net = trial
                changed = True
                break
    return net

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Initial: odd-even transposition
    net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    if not _is_valid(net):
        return net
    
    # Greedy removal
    net = _greedy_remove(net, deadline)
    
    # Try multiple passes with different removal orders
    for _ in range(3):
        if time.time() > deadline:
            break
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(net)):
                if time.time() > deadline:
                    break
                trial = net[:idx] + net[idx+1:]
                if _is_valid(trial):
                    net = trial
                    changed = True
                    break
    
    return net
