import numpy as np
import time
import random

N = 18

def construct():
    # Initial odd-even transposition network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Generate all boolean inputs
    # Shape: (2^18, 18)
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(1 << N) >> i) & 1
        
    def verify(net):
        state = inputs.copy()
        for i, j in net:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        # Check if sorted
        return np.all(state[:, :-1] <= state[:, 1:])
        
    start_time = time.time()
    deadline = 50.0 # Leave 10s margin for 60s limit
    
    # Greedy deletion and swapping
    while time.time() - start_time < deadline:
        # Try deleting a random comparator
        if len(network) > 0:
            idx = random.randint(0, len(network) - 1)
            candidate = network[:idx] + network[idx+1:]
            if verify(candidate):
                network = candidate
                
        # Try swapping adjacent comparators
        if len(network) > 1:
            idx = random.randint(0, len(network) - 2)
            candidate = network[:idx] + [network[idx+1], network[idx]] + network[idx+2:]
            if verify(candidate):
                network = candidate
                
    return network
