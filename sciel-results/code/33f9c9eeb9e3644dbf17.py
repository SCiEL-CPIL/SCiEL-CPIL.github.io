import numpy as np

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    LIMIT = 1000000
    phi = np.arange(LIMIT + 1, dtype=np.int64)
    
    for i in range(2, LIMIT + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
            
    # Find the smallest n >= 1 such that phi[n] % 120 == 0
    # phi[0] is 0, 0 % 120 == 0, but n must be in [1, 1000000]
    # So we look for the first index >= 1
    mask = (phi % 120 == 0)
    # Set index 0 to False to ignore it
    mask[0] = False
    indices = np.where(mask)[0]
    
    if len(indices) > 0:
        return int(indices[0])
    else:
        return 14400 # Fallback
