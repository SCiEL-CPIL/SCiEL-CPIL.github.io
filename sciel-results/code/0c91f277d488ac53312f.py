import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    # Generate all binary vectors of length d
    # We want a subset with min Hamming distance >= 3
    # Map to {-1, 1}
    
    selected = []
    
    # Optimization: Precompute or just loop
    # Since 2^11 is small, we can just loop
    
    for i in range(2**d):
        # Convert i to binary vector of length d
        # b[j] = (i >> j) & 1
        # Check distance against all selected
        is_valid = True
        for s in selected:
            # s is an integer representation of the binary vector?
            # It's faster to store integers and use popcount
            # dist = popcount(i ^ s)
            # We need dist >= 3
            diff = i ^ s
            # Count bits
            # Python int has no builtin popcount in older versions, 
            # but bin(x).count('1') is fast enough for 11 bits.
            # Or use a lookup table for 16-bit chunks? 
            # 11 bits is tiny.
            if bin(diff).count('1') < 3:
                is_valid = False
                break
        
        if is_valid:
            selected.append(i)
            
    # Convert selected integers to points
    # Point j has coord k = 2 * bit_k - 1
    points = []
    for s in selected:
        p = np.array([(s >> k) & 1 for k in range(d)], dtype=np.int64)
        p = 2 * p - 1
        points.append(p)
        
    points = np.array(points, dtype=np.int64)
    return points
