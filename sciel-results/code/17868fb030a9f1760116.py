import math
import bisect
import numpy as np

def construct():
    N = 30
    # Calculate LCM of 1 to 30
    L = 1
    for i in range(1, 31):
        L = L * i // math.gcd(L, i)
    
    weights = np.array([L // i for i in range(1, 31)], dtype=np.int64)
    
    # Split indices into two halves
    n1 = N // 2
    n2 = N - n1
    
    indices1 = np.arange(n1)
    indices2 = np.arange(n1, N)
    
    def generate_sums(indices):
        # Start with 0
        current_sums = np.array([0], dtype=np.int64)
        current_encs = np.array([0], dtype=np.int64)
        
        for k, idx in enumerate(indices):
            w = weights[idx]
            # Create new sums
            # s - w, s, s + w
            new_sums = np.concatenate([current_sums - w, current_sums, current_sums + w])
            
            # Create new encs
            # Encoding: base 3.
            # Old encs have k digits.
            # New digit is at position k.
            # Digit 0 for -1, 1 for 0, 2 for 1.
            # new_enc = old_enc + digit * (3**k)
            
            pow3 = 3 ** k
            # We need to add 0, 1*pow3, 2*pow3 to the encs
            # The order in new_sums is: [s-w, s, s+w] corresponding to digits 0, 1, 2
            # So we need to add 0, pow3, 2*pow3 to the corresponding encs
            # But current_encs is repeated 3 times.
            # We can create a pattern array
            # pattern = [0, pow3, 2*pow3] repeated len(current_encs) times
            
            # Efficient way:
            # Repeat current_encs 3 times
            repeated_encs = np.tile(current_encs, 3)
            # Create the addition pattern
            # The order in concatenate is:
            # Block 1: current_sums - w (digit 0)
            # Block 2: current_sums (digit 1)
            # Block 3: current_sums + w (digit 2)
            
            # So for the first block, add 0
            # For the second block, add pow3
            # For the third block, add 2*pow3
            
            n_curr = len(current_encs)
            add_pattern = np.zeros(3 * n_curr, dtype=np.int64)
            add_pattern[0:n_curr] = 0
            add_pattern[n_curr:2*n_curr] = pow3
            add_pattern[2*n_curr:3*n_curr] = 2 * pow3
            
            new_encs = repeated_encs + add_pattern
            
            current_sums = new_sums
            current_encs = new_encs
            
        return current_sums, current_encs

    # Generate for half 1
    sums1, encs1 = generate_sums(indices1)
    
    # Sort by sum
    # argsort
    sorted_indices = np.argsort(sums1)
    sorted_sums = sums1[sorted_indices]
    sorted_encs = encs1[sorted_indices]
    
    best_val = float('inf')
    best_enc1 = 0
    best_enc2 = 0
    
    # Generate for half 2
    sums2, encs2 = generate_sums(indices2)
    
    # To speed up, we can iterate over sums2
    # For each s2, find closest s1 to -s2
    # Using bisect on numpy array?
    # np.searchsorted is equivalent to bisect
    
    for s2, e2 in zip(sums2, encs2):
        target = -s2
        idx = np.searchsorted(sorted_sums, target)
        
        candidates = []
        if idx < len(sorted_sums):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
            
        for i in candidates:
            s1 = sorted_sums[i]
            total_sum = s1 + s2
            if total_sum == 0:
                continue
            abs_sum = abs(total_sum)
            if abs_sum < best_val:
                best_val = abs_sum
                best_enc1 = sorted_encs[i]
                best_enc2 = e2
                if best_val == 1:
                    # Can't get better than 1 (in integer units)
                    # But we need to break out of both loops
                    break
        if best_val == 1:
            break

    # Decode
    c = [0] * N
    enc = best_enc1
    for k, idx in enumerate(indices1):
        digit = enc % 3
        enc //= 3
        if digit == 0:
            c[idx] = -1
        elif digit == 1:
            c[idx] = 0
        else:
            c[idx] = 1
            
    enc = best_enc2
    for k, idx in enumerate(indices2):
        digit = enc % 3
        enc //= 3
        if digit == 0:
            c[idx] = -1
        elif digit == 1:
            c[idx] = 0
        else:
            c[idx] = 1
            
    return c
