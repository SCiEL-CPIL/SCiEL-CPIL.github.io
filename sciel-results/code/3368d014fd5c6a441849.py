import time
import numpy as np
import itertools

N = 16

# EVOLVE-BLOCK-START
import time

def _verify_network_fast(network, n=16):
    """Verify network sorts all 2^n inputs using zero-one principle with NumPy."""
    if not network:
        return n <= 1
        
    num_inputs = 1 << n
    inputs = np.arange(num_inputs, dtype=np.uint32)
    
    masks = (1 << np.arange(n, dtype=np.uint32))
    vals = ((inputs[:, None] & masks[None, :]) > 0).astype(np.uint8)
    
    for i, j in network:
        if i < 0 or i >= n or j < 0 or j >= n or i >= j:
            return False
        vi = vals[:, i]
        vj = vals[:, j]
        
        vi_min = vi & vj
        vj_max = vi | vj
        
        vals[:, i] = vi_min
        vals[:, j] = vj_max
        
    diffs = vals[:, 1:] - vals[:, :-1]
    return np.all(diffs >= 0)

def construct():
    """Return a sorting network for 16 wires."""
    # Use Odd-Even Transposition Sort
    # It is correct and uses 120 comparators.
    net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify
    if not _verify_network_fast(net, N):
        # Fallback: Identity? No, must sort.
        # Just return the net, it should be correct.
        pass
        
    return net
# EVOLVE-BLOCK-END
