import numpy as np
import time
import random

N = 17
DEADLINE = 50.0

def generate_inputs(N):
    n_inputs = 2**N
    inputs = np.zeros((n_inputs, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(n_inputs) >> i) & 1
    return inputs

INPUTS = generate_inputs(N)

def is_valid(comparators):
    state = INPUTS.copy()
    for i, j in comparators:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    diffs = np.diff(state, axis=1)
    return np.all(diffs >= 0)

def greedy_prune(comps):
    comps = list(comps)
    improved = True
    while improved:
        improved = False
        for i in range(len(comps)):
            temp = comps[:i] + comps[i+1:]
            if is_valid(temp):
                comps = temp
                improved = True
                break
    return comps

def construct():
    start_time = time.time()
    
    # Baseline: Odd-Even Transposition
    baseline = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best = greedy_prune(baseline)
    best_count = len(best)
    
    # Local Search
    while time.time() - start_time < DEADLINE - 5:
        current = best.copy()
        
        # Perturb
        for _ in range(10):
            move = random.choice(['del', 'ins', 'swap', 'mod'])
            if move == 'del' and len(current) > 0:
                idx = random.randint(0, len(current)-1)
                current.pop(idx)
            elif move == 'ins':
                i = random.randint(0, N-2)
                j = random.randint(i+1, N-1)
                current.append((i, j))
            elif move == 'swap' and len(current) > 1:
                i = random.randint(0, len(current)-1)
                j = random.randint(0, len(current)-1)
                current[i], current[j] = current[j], current[i]
            elif move == 'mod' and len(current) > 0:
                idx = random.randint(0, len(current)-1)
                i, j = current[idx]
                if random.random() < 0.5:
                    new_i = random.randint(0, N-1)
                    if new_i != j:
                        if new_i > j: new_i, j = j, new_i
                        current[idx] = (new_i, j)
                else:
                    new_j = random.randint(0, N-1)
                    if new_j != i:
                        if new_j < i: i, new_j = new_j, i
                        current[idx] = (i, new_j)
        
        pruned = greedy_prune(current)
        
        if len(pruned) < best_count:
            best = pruned
            best_count = len(best)
            
    return best
