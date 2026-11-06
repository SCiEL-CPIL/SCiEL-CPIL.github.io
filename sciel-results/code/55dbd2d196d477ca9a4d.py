import numpy as np
import time
import random

N = 16

def construct():
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    inputs = np.arange(2**N)
    inputs_bin = np.zeros((2**N, N), dtype=np.uint8)
    for k in range(N):
        inputs_bin[:, k] = (inputs >> k) & 1
        
    def get_inversions(net):
        state = inputs_bin.copy()
        for i, j in net:
            a = state[:, i]
            b = state[:, j]
            min_val = np.minimum(a, b)
            max_val = np.maximum(a, b)
            state[:, i] = min_val
            state[:, j] = max_val
        return np.sum(state[:, :-1] > state[:, 1:])

    best_network = network.copy()
    best_len = len(network)
    current_inversions = get_inversions(network)
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        if len(network) == 0:
            break
            
        move_type = random.random()
        new_net = network.copy()
        
        if move_type < 0.5:
            # Remove
            if len(new_net) > 0:
                idx = random.randint(0, len(new_net) - 1)
                new_net.pop(idx)
        elif move_type < 0.8:
            # Swap
            if len(new_net) >= 2:
                i1 = random.randint(0, len(new_net) - 1)
                i2 = random.randint(0, len(new_net) - 1)
                if i1 != i2:
                    new_net[i1], new_net[i2] = new_net[i2], new_net[i1]
        else:
            # Move/Change wire
            if len(new_net) > 0:
                idx = random.randint(0, len(new_net) - 1)
                i, j = new_net[idx]
                # Change i or j to a random valid wire
                if random.random() < 0.5:
                    new_i = random.randint(0, N - 1)
                    if new_i != j:
                        new_net[idx] = (new_i, j) if new_i < j else (j, new_i)
                else:
                    new_j = random.randint(0, N - 1)
                    if new_j != i:
                        new_net[idx] = (i, new_j) if i < new_j else (new_j, i)

        new_inv = get_inversions(new_net)
        
        # Energy: Length is primary. Inversions are penalty.
        # We want to minimize Length, but only valid networks (inv=0) are acceptable for final score.
        # During search, we allow inv>0.
        # E = len * 100 + inv
        delta_E = (len(new_net) * 100 + new_inv) - (len(network) * 100 + current_inversions)
        
        if delta_E < 0 or random.random() < np.exp(-delta_E / T):
            network = new_net
            current_inversions = new_inv
            
            if current_inversions == 0 and len(network) < best_len:
                best_network = network.copy()
                best_len = len(network)
                
        T *= cooling
        
    return best_network
