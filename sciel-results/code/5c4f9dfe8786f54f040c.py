import time
from math import gcd
from functools import reduce

def construct():
    deadline = time.time() + 85  # 90s limit, 5s margin
    
    # Calculate LCM of 1..30
    L = reduce(lambda a, b: a * b // gcd(a, b), range(1, 31))
    
    # Weights w[i] = L / (i+1) for i in 0..29
    weights = [L // (i + 1) for i in range(30)]
    
    half = 15
    
    # Generate all sums and masks for the first half (indices 0..14)
    # and the second half (indices 15..29)
    
    def generate_half(start_idx, end_idx):
        """
        Generates list of (sum, mask) for variables in [start_idx, end_idx)
        mask uses 2 bits per variable, relative to start_idx.
        """
        n = end_idx - start_idx
        w = weights[start_idx:end_idx]
        
        sums = [0]
        masks = [0]
        
        for i in range(n):
            wi = w[i]
            new_sums = []
            new_masks = []
            for j in range(len(sums)):
                s = sums[j]
                m = masks[j]
                # val = -1 (mask bit 1)
                new_sums.append(s - wi)
                new_masks.append(m | (1 << (2*i)))
                # val = 0 (mask bit 0)
                new_sums.append(s)
                new_masks.append(m)
                # val = 1 (mask bit 2)
                new_sums.append(s + wi)
                new_masks.append(m | (2 << (2*i)))
            sums = new_sums
            masks = new_masks
            
        return list(zip(sums, masks))
        
    t_start = time.time()
    
    # Generate half 1 (indices 0-14)
    list_A = generate_half(0, half)
    
    # Generate half 2 (indices 15-29)
    list_B = generate_half(half, 30)
    
    # Create dictionary for B: sum -> mask
    # If multiple masks yield the same sum, we only need one.
    dict_B = {}
    for s, m in list_B:
        if s not in dict_B:
            dict_B[s] = m
            
    # Search for smallest non-zero k
    # k = sum_A + sum_B
    # sum_B = k - sum_A
    # We iterate k = 1, -1, 2, -2, ...
    
    # To optimize, we can pre-compute a set of sums from A for faster lookup?
    # No, we need the mask from A too.
    # So we iterate list_A for each k.
    
    # Optimization: If list_A is large, iterating it for each k is O(N_A * K).
    # N_A = 3^15 ~ 14M.
    # If K=1, it's 14M lookups. Fast.
    # If K=10, it's 140M lookups. Might be slow in Python (10-20s).
    # But K is likely 1.
    
    # Let's try iterating k.
    # We can also break early if time is running out.
    
    k = 1
    while True:
        if time.time() > deadline - 5:
            # Fallback: return a random non-zero sum if we haven't found a small one
            # This shouldn't happen, but just in case.
            # Pick first item from A and first from B
            s_a, m_a = list_A[0]
            s_b, m_b = list_B[0]
            total_sum = s_a + s_b
            if total_sum == 0:
                # Force non-zero
                # Flip a coefficient
                # This is a hack, but we expect to find k=1 quickly.
                pass 
            break
            
        found = False
        for s_a, m_a in list_A:
            target_b = k - s_a
            if target_b in dict_B:
                m_b = dict_B[target_b]
                
                # Reconstruct coefficients
                coeffs = [0] * 30
                
                # Decode m_a (for indices 0..14)
                for i in range(half):
                    bits = (m_a >> (2*i)) & 3
                    if bits == 1:
                        coeffs[i] = -1
                    elif bits == 2:
                        coeffs[i] = 1
                    else:
                        coeffs[i] = 0
                        
                # Decode m_b (for indices 15..29)
                for i in range(half):
                    bits = (m_b >> (2*i)) & 3
                    if bits == 1:
                        coeffs[half + i] = -1
                    elif bits == 2:
                        coeffs[half + i] = 1
                    else:
                        coeffs[half + i] = 0
                        
                return coeffs
        
        # If not found, try next k
        if k > 0:
            k = -k
        else:
            k = -k + 1
            # k sequence: 1, -1, 2, -2, 3, -3 ...
            # Wait, my logic above:
            # k=1. Next should be -1.
            # k=-1. Next should be 2.
            # k=2. Next should be -2.
            # k=-2. Next should be 3.
            # Logic:
            # if k > 0: k = -k -> 1 -> -1. Correct.
            # if k <= 0: k = -k + 1 -> -1 -> 2. Correct.
            # 2 -> -2. Correct.
            # -2 -> 3. Correct.

    # Fallback (should never be reached)
    return [0]*29 + [1]

