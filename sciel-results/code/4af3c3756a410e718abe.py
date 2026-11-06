import numpy as np
import itertools
import time
import random

N = 17

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Generate Inputs
    # 131,072 inputs
    # Using product is slow, but we only do it once.
    # 131k * 17 = 2.2M elements.
    inputs = np.array(list(itertools.product([0,1], repeat=N)), dtype=np.int8)
    
    # Subset for fast verification
    # Pick 2000 random indices
    idx = np.random.choice(len(inputs), 2000, replace=False)
    subset_inputs = inputs[idx]
    
    # 2. Verification Functions
    def verify_subset(net):
        state = subset_inputs.copy()
        for i, j in net:
            col_i = state[:, i]
            col_j = state[:, j]
            min_val = np.minimum(col_i, col_j)
            max_val = np.maximum(col_i, col_j)
            state[:, i] = min_val
            state[:, j] = max_val
        diffs = np.diff(state, axis=1)
        failing = np.any(diffs < 0, axis=1)
        return np.sum(failing)

    def verify_full(net):
        state = inputs.copy()
        for i, j in net:
            col_i = state[:, i]
            col_j = state[:, j]
            min_val = np.minimum(col_i, col_j)
            max_val = np.maximum(col_i, col_j)
            state[:, i] = min_val
            state[:, j] = max_val
        diffs = np.diff(state, axis=1)
        failing = np.any(diffs < 0, axis=1)
        return np.sum(failing)

    # 3. Initial Network: Odd-Even Transposition
    def generate_oe():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
                
    curr_net = generate_oe()
    
    curr_sub_failing = verify_subset(curr_net)
    curr_cost = len(curr_net) + 10 * curr_sub_failing
    
    best_net = curr_net.copy()
    best_cost = curr_cost
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    def get_neighbor(net):
        if not net:
            return []
        
        op = random.randint(0, 3)
        new_net = net.copy()
        
        if op == 0: # Add
            i = random.randint(0, N-2)
            j = random.randint(i+1, N-1)
            new_net.append((i, j))
            
        elif op == 1: # Remove
            idx = random.randint(0, len(new_net)-1)
            new_net.pop(idx)
            
        elif op == 2: # Move
            idx = random.randint(0, len(new_net)-1)
            swap_idx = random.randint(0, len(new_net)-1)
            new_net[idx], new_net[swap_idx] = new_net[swap_idx], new_net[idx]
            
        elif op == 3: # Change Target
            idx = random.randint(0, len(new_net)-1)
            i, j = new_net[idx]
            new_i = random.randint(0, N-2)
            new_j = random.randint(new_i+1, N-1)
            new_net[idx] = (new_i, new_j)
            
        return new_net

    while time.time() < deadline:
        new_net = get_neighbor(curr_net)
        new_sub_failing = verify_subset(new_net)
        new_cost = len(new_net) + 10 * new_sub_failing
        
        delta = new_cost - curr_cost
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            curr_net = new_net
            curr_sub_failing = new_sub_failing
            curr_cost = new_cost
            
            if curr_sub_failing == 0:
                full_failing = verify_full(curr_net)
                if full_failing == 0:
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_net = curr_net.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = 5.0

    return best_net
