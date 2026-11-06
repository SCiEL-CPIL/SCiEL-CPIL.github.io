import numpy as np
import time

N = 16

def generate_batcher(n):
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    left_net = generate_batcher(half)
    right_net = [(i + half, j + half) for i, j in generate_batcher(half)]
    
    merge_net = []
    
    odd_indices = list(range(1, n, 2))
    odd_net = generate_batcher(half)
    for i, j in odd_net:
        merge_net.append((odd_indices[i], odd_indices[j]))
        
    for i in range(half - 1):
        merge_net.append((2*i, 2*i + 1))
        
    even_indices = list(range(0, n, 2))
    even_net = generate_batcher(half)
    for i, j in even_net:
        merge_net.append((even_indices[i], even_indices[j]))
        
    return left_net + right_net + merge_net

# Pre-compute inputs
mask = 1 << np.arange(N-1, -1, -1, dtype=np.uint32)
bits = np.arange(1 << N, dtype=np.uint32)[:, None]
ORIGINAL_INPUTS = (bits & mask).astype(np.uint8)

def is_valid_network(net):
    # Copy inputs
    inputs = ORIGINAL_INPUTS.copy()
    
    for i, j in net:
        min_val = np.minimum(inputs[:, i], inputs[:, j])
        max_val = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_val
        inputs[:, j] = max_val
        
    return np.all(np.diff(inputs, axis=1) >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    net = generate_batcher(N)
    
    # Greedy Deletion
    improved = True
    while improved:
        improved = False
        # Try removing from the end
        for k in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                break
                
            temp_net = net[:k] + net[k+1:]
            
            if is_valid_network(temp_net):
                net = temp_net
                improved = True
                break
        
        if time.time() > deadline:
            break
            
    return net
