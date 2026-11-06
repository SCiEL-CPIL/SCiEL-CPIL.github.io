import numpy as np
import time

N = 16

# Pre-compute input bits
# Shape (65536, 16)
# We can do this once at module level?
# Yes, standard library is available.
_INPUTS = None

def get_inputs():
    global _INPUTS
    if _INPUTS is None:
        _INPUTS = np.zeros((65536, N), dtype=np.int8)
        for i in range(N):
            _INPUTS[:, i] = (np.arange(65536) >> (N - 1 - i)) & 1
    return _INPUTS

def is_sorting_network(network):
    inputs = get_inputs().copy() # Copy is necessary as we modify it
    
    for i, j in network:
        tmp = np.minimum(inputs[:, i], inputs[:, j])
        inputs[:, j] = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = tmp
        
    return np.all(inputs[:, :-1] <= inputs[:, 1:])

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
            
    if not is_sorting_network(network):
        return network
        
    changed = True
    while changed:
        changed = False
        for k in range(len(network)):
            if time.time() > deadline:
                return network
                
            new_network = network[:k] + network[k+1:]
            if is_sorting_network(new_network):
                network = new_network
                changed = True
                break
                
    return network
