import numpy as np
import time

N = 17
NUM_INPUTS = 1 << N

def _generate_inputs():
    return (np.arange(NUM_INPUTS, dtype=np.uint8).reshape(-1, 1) >> np.arange(N - 1, -1, -1, dtype=np.uint8)).astype(np.uint8)

def _simulate(comparators, inputs):
    state = inputs.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

def _is_sorted(state):
    return np.all(state[:, :-1] <= state[:, 1:])

def construct():
    # Seed
    current_network = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
    
    all_inputs = _generate_inputs()
    
    # Subset for quick check
    # Use a fixed deterministic subset to avoid RNG overhead and ensure reproducibility
    # First 1024 inputs
    subset_inputs = all_inputs[:1024]
    
    deadline = time.time() + 50.0
    
    def is_valid_full(network):
        final_state = _simulate(network, all_inputs)
        return _is_sorted(final_state)
    
    def is_valid_subset(network):
        final_state = _simulate(network, subset_inputs)
        return _is_sorted(final_state)
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try removals
        removed = False
        for k in range(len(current_network)):
            if time.time() > deadline:
                break
            if removed:
                break
            new_network = current_network[:k] + current_network[k+1:]
            if is_valid_subset(new_network):
                if is_valid_full(new_network):
                    current_network = new_network
                    improved = True
                    removed = True
                    break
        
        if removed:
            continue
            
        # 2. Try swaps (adjacent)
        swapped = False
        for k in range(len(current_network) - 1):
            if time.time() > deadline:
                break
            if swapped:
                break
            new_network = current_network.copy()
            new_network[k], new_network[k+1] = current_network[k+1], current_network[k]
            if is_valid_subset(new_network):
                if is_valid_full(new_network):
                    current_network = new_network
                    improved = True
                    swapped = True
                    break
                    
        if swapped:
            continue
            
        # 3. Try longer swaps (distance 2)
        moved = False
        for k in range(len(current_network) - 2):
            if time.time() > deadline:
                break
            if moved:
                break
            new_network = current_network.copy()
            # Move k to k+2
            comp = new_network.pop(k)
            new_network.insert(k+2, comp)
            if is_valid_subset(new_network):
                if is_valid_full(new_network):
                    current_network = new_network
                    improved = True
                    moved = True
                    break

    return current_network
