import numpy as np
import time
import random

def _score(U):
    U = np.array(sorted(set(U)))
    if len(U) == 0:
        return 0.0
    if U[0] != 0:
        # Must contain 0
        return -np.inf
    sums = np.unique(U[:, None] + U[None, :])
    diffs = np.unique(U[:, None] - U[None, :])
    n_sums = len(sums)
    n_diffs = len(diffs)
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    denom = np.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1.0 + np.log(n_diffs / n_sums) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 2, 4, 8],
        [0, 1, 4, 10],
        [0, 1, 3, 8, 12],
        [0, 1, 2, 4, 8, 16],
    ]
    
    best_set = candidates[0]
    best_score = _score(best_set)
    
    # Simulated Annealing
    current_set = list(best_set)
    current_score = best_score
    
    max_val = 60
    max_size = 10
    
    temp = 0.1
    cooling = 0.99
    
    while time.time() < deadline:
        if temp < 0.001:
            temp = 0.01 # Reheat occasionally or just keep going?
            # Better to just restart if stuck
            if random.random() < 0.01:
                temp = 0.1
                current_set = list(best_set)
                current_score = best_score
                continue
        
        # Generate neighbor
        move_type = random.choice(['add', 'remove', 'replace'])
        
        neighbor = list(current_set)
        
        if move_type == 'add':
            if len(neighbor) < max_size:
                x = random.randint(1, max_val)
                if x not in neighbor:
                    neighbor.append(x)
                else:
                    continue
        elif move_type == 'remove':
            if len(neighbor) > 1: # Keep 0
                # Pick index > 0
                idx = random.randint(1, len(neighbor) - 1)
                neighbor.pop(idx)
        else: # replace
            if len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)
                old_x = neighbor[idx]
                new_x = random.randint(1, max_val)
                if new_x not in neighbor:
                    neighbor[idx] = new_x
                else:
                    continue
        
        neighbor_score = _score(neighbor)
        
        if neighbor_score > current_score:
            current_set = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            # Accept with probability exp((new - old) / temp)
            delta = neighbor_score - current_score
            if delta < 0:
                prob = np.exp(delta / temp)
                if random.random() < prob:
                    current_set = neighbor
                    current_score = neighbor_score
        
        temp *= cooling
        
    return best_set
