import math
import time

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    # 1. Calculate LCM of 1..30
    L = 1
    for i in range(1, 31):
        L = L * i // math.gcd(L, i)
        
    # 2. Calculate weights w_i = L / i
    weights = [L // i for i in range(1, 31)]
    
    # Split weights into two halves
    w1 = weights[:15]
    w2 = weights[15:]
    
    # 3. Generate all sums for the first half
    # Dictionary: sum_value -> encoded_coefficients
    # Encoding: c in {-1, 0, 1} -> v in {2, 0, 1}
    # code = sum(v_i * 3^i)
    
    # Base 3 powers for decoding later
    pow3 = [1] * 15
    for i in range(1, 15):
        pow3[i] = pow3[i-1] * 3
        
    # Map value to code digit
    # -1 -> 2, 0 -> 0, 1 -> 1
    val_to_digit = {-1: 2, 0: 0, 1: 1}
    
    # Iterative generation for first half
    # Start with sum 0, code 0
    sums1 = {0: 0}
    
    for i, w in enumerate(w1):
        new_sums = {}
        p3 = pow3[i]
        for s, code in sums1.items():
            # c = 0
            new_sums[s] = code
            # c = 1
            new_sums[s + w] = code + p3
            # c = -1
            new_sums[s - w] = code + 2 * p3
        sums1 = new_sums
        
        # Check time
        if time.time() > deadline:
            break
            
    # If we didn't finish generation, we can't do MITM properly.
    # But 3^15 is small, should be fine.
    
    # 4. Iterate through second half sums and check for N=1 or N=-1
    # We do this iteratively to avoid storing all S2 sums.
    
    # We need to be able to check if a target sum exists in sums1.
    # sums1 is a dict, so O(1) lookup.
    
    # We will generate S2 sums recursively or iteratively.
    # Since we just need to find ONE solution, we can stop early.
    
    # Let's use a recursive generator for the second half to avoid memory overhead of storing all S2.
    # But recursion depth 15 is fine.
    
    # However, Python recursion overhead might be significant for 3^15 calls.
    # Let's use an iterative approach with a stack or just nested loops?
    # Nested loops of depth 15 is not easy to write.
    # Let's use a list-based iterative expansion for S2 as well, but we can stop early if we find a match.
    # Actually, we can just iterate through the expansion.
    
    # Let's try a different way: 
    # Generate all S2 sums in a list? 14M ints. ~100MB.
    # Then loop through them?
    # This is faster than recursive generation in Python.
    
    sums2 = {0: 0}
    for i, w in enumerate(w2):
        new_sums = {}
        p3 = pow3[i]
        for s, code in sums2.items():
            new_sums[s] = code
            new_sums[s + w] = code + p3
            new_sums[s - w] = code + 2 * p3
        sums2 = new_sums
        if time.time() > deadline:
            break

    # Now we have sums1 and sums2 dictionaries.
    # We want to find s1 in sums1 and s2 in sums2 such that s1 + s2 = 1 or -1.
    
    # To optimize, iterate over the smaller dict? They are same size.
    # Iterate over sums2 keys.
    
    found_solution = None
    
    # Check for N = 1
    for s2, code2 in sums2.items():
        target = 1 - s2
        if target in sums1:
            code1 = sums1[target]
            # Reconstruct coefficients
            coeffs = [0] * 30
            # Decode code1
            c1 = code1
            for i in range(15):
                v = c1 % 3
                c1 //= 3
                if v == 0:
                    coeffs[i] = 0
                elif v == 1:
                    coeffs[i] = 1
                else:
                    coeffs[i] = -1
            # Decode code2
            c2 = code2
            for i in range(15):
                v = c2 % 3
                c2 //= 3
                if v == 0:
                    coeffs[15 + i] = 0
                elif v == 1:
                    coeffs[15 + i] = 1
                else:
                    coeffs[15 + i] = -1
            found_solution = coeffs
            break
    
    if found_solution is None:
        # Check for N = -1
        for s2, code2 in sums2.items():
            target = -1 - s2
            if target in sums1:
                code1 = sums1[target]
                coeffs = [0] * 30
                c1 = code1
                for i in range(15):
                    v = c1 % 3
                    c1 //= 3
                    if v == 0:
                        coeffs[i] = 0
                    elif v == 1:
                        coeffs[i] = 1
                    else:
                        coeffs[i] = -1
                c2 = code2
                for i in range(15):
                    v = c2 % 3
                    c2 //= 3
                    if v == 0:
                        coeffs[15 + i] = 0
                    elif v == 1:
                        coeffs[15 + i] = 1
                    else:
                        coeffs[15 + i] = -1
                found_solution = coeffs
                break

    if found_solution is not None:
        return found_solution

    # Fallback: Find minimum |N|
    # Sort keys of sums1
    keys1 = sorted(sums1.keys())
    
    min_abs_N = float('inf')
    best_coeffs = None
    
    for s2, code2 in sums2.items():
        # Binary search for -s2 in keys1
        target = -s2
        # Use bisect
        import bisect
        idx = bisect.bisect_left(keys1, target)
        
        # Check neighbors
        candidates = []
        if idx < len(keys1):
            candidates.append(keys1[idx])
        if idx > 0:
            candidates.append(keys1[idx-1])
            
        for s1 in candidates:
            N = s1 + s2
            if N != 0:
                if abs(N) < min_abs_N:
                    min_abs_N = abs(N)
                    code1 = sums1[s1]
                    coeffs = [0] * 30
                    c1 = code1
                    for i in range(15):
                        v = c1 % 3
                        c1 //= 3
                        if v == 0:
                            coeffs[i] = 0
                        elif v == 1:
                            coeffs[i] = 1
                        else:
                            coeffs[i] = -1
                    c2 = code2
                    for i in range(15):
                        v = c2 % 3
                        c2 //= 3
                        if v == 0:
                            coeffs[15 + i] = 0
                        elif v == 1:
                            coeffs[15 + i] = 1
                        else:
                            coeffs[15 + i] = -1
                    best_coeffs = coeffs
    
    if best_coeffs is not None:
        return best_coeffs
    
    # If all else fails, return seed
    return [0] * 28 + [1, -1]
