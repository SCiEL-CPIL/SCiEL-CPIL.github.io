"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _batcher_oes(N):
    """Generate Batcher's Odd-Even Merge Sort network for N wires."""
    if N <= 1:
        return []
    if N == 2:
        return [(0, 1)]
    
    even_idx = [i for i in range(0, N, 2)]
    odd_idx = [i for i in range(1, N, 2)]
    
    net = []
    
    # Recursively sort even and odd subsequences
    for i, j in _batcher_oes(len(even_idx)):
        net.append((even_idx[i], even_idx[j]))
        
    for i, j in _batcher_oes(len(odd_idx)):
        net.append((odd_idx[i], odd_idx[j]))
        
    n = N // 2
    
    # Cross comparisons
    for k in range(n):
        net.append((even_idx[k], odd_idx[k]))
        
    # Recursive merges
    def _map_net(net, mapping):
        return [(mapping[i], mapping[j]) for i, j in net]
        
    net += _map_net(_batcher_oes(n), even_idx)
    net += _map_net(_batcher_oes(n), odd_idx)
    
    return net

def _is_valid_sorter(net, N=18, timeout=5):
    """Vectorized validation using NumPy."""
    start_time = time.time()
    
    masks = np.arange(1 << N, dtype=np.int64)
    
    for i, j in net:
        if time.time() - start_time > timeout:
            return False
        
        bits_i = (masks >> i) & 1
        bits_j = (masks >> j) & 1
        
        swap_mask = (bits_i > bits_j)
        
        if np.any(swap_mask):
            masks[swap_mask] ^= (1 << i) | (1 << j)
            
    for m in masks:
        k = bin(m).count('1')
        expected = ((1 << k) - 1) << (N - k)
        if m != expected:
            return False
            
    return True

def construct():
    net = _batcher_oes(N)
    
    if not _is_valid_sorter(net):
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    start_time = time.time()
    timeout = 50
    
    indices = list(range(len(net)))
    
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > timeout:
                break
                
            test_net = net[:idx] + net[idx+1:]
            
            if _is_valid_sorter(test_net):
                net = test_net
                improved = True
                break
                
    return net
# EVOLVE-BLOCK-END
