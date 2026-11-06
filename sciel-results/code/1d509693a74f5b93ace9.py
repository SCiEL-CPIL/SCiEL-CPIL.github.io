import math
import bisect

def construct():
    N = 30
    # Calculate LCM of 1 to 30
    D = 1
    for i in range(1, N + 1):
        D = D * i // math.gcd(D, i)
    
    # Weights w_i = D / i
    weights = [D // i for i in range(1, N + 1)]
    
    half = N // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Generate all signed sums for the first half
    # sum = sum(sigma_i * w_i) where sigma_i in {-1, 1}
    # Using bitmask: 1 -> +1, 0 -> -1
    sums1 = []
    for mask in range(1 << half):
        s = 0
        for i in range(half):
            if mask & (1 << i):
                s += w1[i]
            else:
                s -= w1[i]
        sums1.append((s, mask))
        
    # Generate all signed sums for the second half
    sums2 = []
    len2 = N - half
    for mask in range(1 << len2):
        s = 0
        for i in range(len2):
            if mask & (1 << i):
                s += w2[i]
            else:
                s -= w2[i]
        sums2.append((s, mask))
        
    # Sort sums2 by the sum value
    sums2.sort(key=lambda x: x[0])
    
    # Pre-extract values for binary search
    vals2 = [s[0] for s in sums2]
    
    best_val = float('inf')
    best_mask1 = 0
    best_mask2 = 0
    
    # For each sum in first half, find the best matching sum in second half
    for s1, m1 in sums1:
        target = -s1
        # Find position where target would be inserted
        pos = bisect.bisect_left(vals2, target)
        
        # Check a window of neighbors around pos
        # The optimal non-zero sum will be very close to 0, so s2 close to -s1
        # If the closest is 0, the next closest is adjacent.
        # We check a small window to be safe.
        for p in range(max(0, pos - 5), min(len2, pos + 6)):
            s2, m2 = sums2[p]
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_val:
                best_val = abs_total
                best_mask1 = m1
                best_mask2 = m2
                
    # Reconstruct coefficients
    coeffs = [0] * N
    for i in range(half):
        if best_mask1 & (1 << i):
            coeffs[i] = 1
        else:
            coeffs[i] = -1
            
    for i in range(len2):
        if best_mask2 & (1 << i):
            coeffs[half + i] = 1
        else:
            coeffs[half + i] = -1
            
    return coeffs
