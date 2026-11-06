import numpy as np
import time

N = 18

def _verify(net, inputs_template):
    # inputs_template is a (2**N, N) uint8 array
    # We need a copy because we modify it
    inputs = inputs_template.copy()
    
    for i, j in net:
        col_i = inputs[:, i]
        col_j = inputs[:, j]
        mask = col_i > col_j
        if np.any(mask):
            inputs[mask, i] = col_j[mask]
            inputs[mask, j] = col_i[mask]
            
    diffs = np.diff(inputs, axis=1)
    if not np.all(diffs >= 0):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 50.0 # Leave 10s buffer for safety
    
    # Generate template inputs
    # 2^18 = 262144
    # N = 18
    template = np.zeros((2**N, N), dtype=np.uint8)
    for j in range(N):
        template[:, j] = (np.arange(2**N) >> j) & 1
        
    # 1. Generate initial network: Odd-Even Transposition
    net = []
    for k in range(N):
        if k % 2 == 0:
            for i in range(0, N-1, 2):
                net.append((i, i+1))
        else:
            for i in range(1, N-1, 2):
                net.append((i, i+1))
                
    # Verify initial network (should be True)
    # if not _verify(net, template):
    #     return net # Fallback, though it should be correct
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try removing each comparator
        # Iterate backwards to avoid index shifting issues when popping
        for idx in range(len(net) - 1, -1, -1):
            if (time.time() - start_time) > deadline:
                break
                
            # Try removing net[idx]
            original = net[idx]
            net.pop(idx)
            
            if _verify(net, template):
                improved = True
                # Keep it removed
            else:
                # Restore
                net.insert(idx, original)
                
    return net
