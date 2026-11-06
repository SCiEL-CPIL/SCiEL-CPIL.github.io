import sys
import numpy as np
import time
import random

N = 18

def construct():
    """
    Constructs a sorting network for N wires using Odd-Even Transposition seed
    and Greedy Removal with NumPy verification.
    """
    
    def verify_fast(comps):
        """Verify if the network sorts all 2^N inputs using NumPy."""
        if not comps:
            return N <= 1
            
        inputs = np.arange(1 << N, dtype=np.uint8)
        # Generate bits matrix
        # Shape: (2^N, N)
        bits = (inputs[:, None] >> np.arange(N, dtype=np.uint8)[None, :]) & 1
        
        for i, j in comps:
            min_val = np.minimum(bits[:, i], bits[:, j])
            max_val = np.maximum(bits[:, i], bits[:, j])
            bits[:, i] = min_val
            bits[:, j] = max_val
        
        # Check if sorted
        sorted_mask = np.all(bits[:, :-1] <= bits[:, 1:], axis=1)
        return np.all(sorted_mask)

    # Seed: Odd-Even Transposition
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    
    best_net = net
    best_count = len(net)
    
    start_time = time.time()
    limit = 55.0
    
    random.seed(42)
    
    # Greedy Removal
    for _ in range(20):
        if time.time() - start_time > limit:
            break
        
        current_net = best_net[:]
        i = 0
        improved = False
        while i < len(current_net):
            if time.time() - start_time > limit:
                break
            
            candidate_net = current_net[:i] + current_net[i+1:]
            
            if verify_fast(candidate_net):
                current_net = candidate_net
                improved = True
                # Do not increment i
            else:
                i += 1
        
        if len(current_net) < best_count:
            best_net = current_net
            best_count = len(current_net)
        else:
            break
            
    return best_net
