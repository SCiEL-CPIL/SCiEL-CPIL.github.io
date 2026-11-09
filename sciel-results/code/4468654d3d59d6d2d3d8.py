import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55  # 5s safety margin
    
    # 1. Precompute all arithmetic progressions
    progressions = []
    index_to_progs = [[] for _ in range(n)]
    
    pid = 0
    for d in range(1, n):
        for a in range(d):
            max_k = (n - 1 - a) // d + 1
            for k in range(1, max_k + 1):
                indices = tuple(a + j * d for j in range(k))
                progressions.append(indices)
                for idx in indices:
                    index_to_progs[idx].append(pid)
                pid += 1
                
    num_progs = len(progressions)
    
    # 2. Greedy initialization: start with +1, then greedily fix worst progressions
    rng = random.Random(42)
    signs = np.ones(n, dtype=np.int8)
    
    # Compute initial sums
    sums = np.zeros(num_progs, dtype=np.int16)
    for p_idx, indices in enumerate(progressions):
        sums[p_idx] = np.sum(signs[list(indices)])
    
    # Greedy phase: repeatedly flip the index in the worst progression that most reduces max
    greedy_iters = 200
    for _ in range(greedy_iters):
        if time.time() > deadline - 5:
            break
        worst_pid = np.argmax(np.abs(sums))
        worst_indices = progressions[worst_pid]
        current_worst = abs(sums[worst_pid])
        
        best_flip_idx = -1
        best_new_max = current_worst
        
        for idx in worst_indices:
            old_sign = signs[idx]
            delta = -2 * old_sign
            affected = index_to_progs[idx]
            
            # Simulate flip
            old_vals = sums[affected].copy()
            sums[affected] += delta
            new_max = np.max(np.abs(sums))
            sums[affected] = old_vals
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_flip_idx = idx
        
        if best_flip_idx >= 0:
            old_sign = signs[best_flip_idx]
            delta = -2 * old_sign
            signs[best_flip_idx] = -old_sign
            sums[index_to_progs[best_flip_idx]] += delta
        else:
            break
    
    current_max = np.max(np.abs(sums))
    best_max = current_max
    best_signs = signs.copy()
    
    # 3. Simulated annealing with multi-flip moves
    t_start = time.time()
    temperature = 3.0
    cooling_rate = 0.9995
    min_temperature = 0.1
    
    while time.time() < deadline - 2:
        # Multi-flip move: flip 1-3 random indices
        num_flips = rng.choice([1, 1, 1, 2, 2, 3])
        flip_indices = rng.sample(range(n), num_flips)
        
        old_signs = signs[flip_indices].copy()
        deltas = -2 * old_signs
        
        # Apply all flips
        for i, idx in enumerate(flip_indices):
            signs[idx] = -old_signs[i]
            sums[index_to_progs[idx]] += deltas[i]
        
        new_max = np.max(np.abs(sums))
        
        # Acceptance criterion
        if new_max < best_max:
            best_max = new_max
            best_signs = signs.copy()
            current_max = new_max
        elif new_max <= current_max:
            current_max = new_max
        else:
            # Simulated annealing: accept worse moves with probability
            delta_obj = new_max - current_max
            if temperature > min_temperature:
                accept_prob = np.exp(-delta_obj / temperature)
                if rng.random() < accept_prob:
                    current_max = new_max
                else:
                    # Revert
                    for i, idx in enumerate(flip_indices):
                        signs[idx] = old_signs[i]
                        sums[index_to_progs[idx]] -= deltas[i]
        
        # Cool down
        temperature = max(temperature * cooling_rate, min_temperature)
        
        # Restart if stuck
        if temperature <= min_temperature:
            # Random restart with some perturbation
            for _ in range(rng.randint(5, 15)):
                idx = rng.randint(0, n - 1)
                old_sign = signs[idx]
                delta = -2 * old_sign
                signs[idx] = -old_sign
                sums[index_to_progs[idx]] += delta
            current_max = np.max(np.abs(sums))
            temperature = 1.5
    
    return [int(x) for x in best_signs]
