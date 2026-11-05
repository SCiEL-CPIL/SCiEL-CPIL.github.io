import numpy as np
import time
import random
import itertools

N = 16

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85
    
    # Pre-compute all 2^16 inputs
    # Using uint8
    all_inputs = np.array(list(itertools.product([0, 1], repeat=N))).T
    all_inputs = all_inputs.astype(np.uint8)
    
    def verify(network):
        x = all_inputs.copy()
        for i, j in network:
            # Bitwise min/max for 0/1 values
            min_val = x[i] & x[j]
            max_val = x[i] | x[j]
            x[i] = min_val
            x[j] = max_val
        # Check sorted
        return np.all(np.diff(x, axis=0) >= 0)

    def batcher_oe(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        result = []
        result.extend(batcher_oe(n // 2))
        result.extend([(i + n // 2, j + n // 2) for (i, j) in batcher_oe(n // 2)])
        
        for i in range(n // 2):
            result.append((i, i + n // 2))
            
        for (i, j) in batcher_oe(n // 2):
            result.append((2 * i + 1, 2 * j + 1))
            
        for (i, j) in batcher_oe(n // 2):
            result.append((2 * i, 2 * j))
            
        return result

    current_net = batcher_oe(N)
    
    if not verify(current_net):
        current_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    best_net = list(current_net)
    best_cost = len(best_net)
    
    T = 1.0
    alpha = 0.999
    T_min = 1e-6
    
    while time.time() < deadline:
        candidate = list(current_net)
        move_type = random.random()
        
        if move_type < 0.7:
            # Remove
            if len(candidate) == 0:
                continue
            idx = random.randint(0, len(candidate) - 1)
            candidate.pop(idx)
        elif move_type < 0.9:
            # Swap
            if len(candidate) < 2:
                continue
            i, j = random.sample(range(len(candidate)), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        elif move_type < 0.98:
            # Modify
            if len(candidate) == 0:
                continue
            idx = random.randint(0, len(candidate) - 1)
            i, j = candidate[idx]
            if random.random() < 0.5:
                new_i = random.randint(0, N - 1)
                if new_i == j:
                    continue
                if new_i < j:
                    candidate[idx] = (new_i, j)
                else:
                    candidate[idx] = (j, new_i)
            else:
                new_j = random.randint(0, N - 1)
                if new_j == i:
                    continue
                if i < new_j:
                    candidate[idx] = (i, new_j)
                else:
                    candidate[idx] = (new_j, i)
        else:
            # Add
            if len(candidate) < 200:
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                candidate.append((i, j))
        
        if verify(candidate):
            cand_cost = len(candidate)
            curr_cost = len(current_net)
            
            if cand_cost < curr_cost:
                current_net = candidate
                if cand_cost < best_cost:
                    best_net = list(candidate)
                    best_cost = cand_cost
            elif cand_cost == curr_cost:
                if random.random() < 0.3:
                    current_net = candidate
            else:
                delta = cand_cost - curr_cost
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_net = candidate
        else:
            pass
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_net
# EVOLVE-BLOCK-END
