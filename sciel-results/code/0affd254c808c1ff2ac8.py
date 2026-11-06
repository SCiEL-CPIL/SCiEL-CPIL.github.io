import time
import sys
from math import gcd
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    weights = [L // (i + 1) for i in range(n)]
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    # Helper functions for encoding
    def encode_coeffs(coeffs):
        code = 0
        for c in coeffs:
            code = code * 3 + (c + 1)
        return code

    def decode_coeffs(code, n):
        coeffs = [0] * n
        for i in range(n-1, -1, -1):
            coeffs[i] = (code % 3) - 1
            code //= 3
        return coeffs

    # Generate all sums for the second half
    # sums2_dict: sum -> code
    sums2_dict = {0: 0} # code 0 corresponds to all -1? No.
    # encode([0]*15)
    # 0 -> 1, 0 -> 1... code is 111...111 (base 3)
    # Let's just compute it.
    init_code = encode_coeffs([0]*len(w2))
    sums2_dict = {0: init_code}
    
    current_layer = [(0, init_code)]
    
    for i, w in enumerate(w2):
        next_layer = []
        for s, code in current_layer:
            # We need to update the code for the i-th digit
            # code = c0 * 3^14 + ... + ci * 3^(14-i) + ...
            # It's easier to just keep the coeffs list in the layer?
            # No, that defeats the purpose.
            # We can compute the new code from the old code.
            # Let's say we process digits from left to right (i=0 to 14).
            # code = sum(c_j * 3^(14-j))
            # When we set c_i, we need to know its place value.
            # Place value for i is 3^(14-i).
            
            # Actually, it's easier to generate the codes from scratch or use a list of coeffs in the layer.
            # Given the speed requirements, let's stick to list of coeffs in the layer for simplicity,
            # but convert to code when storing in dict.
            # The layer size is 3^i.
            # Total operations is sum(3^i) ~ 3^15.
            # List slicing is the bottleneck.
            
            # Let's try to optimize the layer representation.
            # We can store (sum, coeffs_list) in the layer.
            # And only encode when adding to dict.
            
            pass # Placeholder

        # Let's revert to list of coeffs in layer, but be careful with memory.
        # 3^15 items, each with a list of 15 ints.
        # This is too much memory.
        
        # Alternative: Use a recursive generator for the second half too, and build the dict.
        # This avoids storing the layer.
        
        pass

    # Let's use the recursive generator for both halves.
    # It's cleaner and memory efficient.
    
    def gen_sums(weights):
        n_w = len(weights)
        coeffs = [0] * n_w
        
        def dfs(idx, current_sum):
            if idx == n_w:
                yield current_sum, encode_coeffs(coeffs)
                return
            
            w = weights[idx]
            
            coeffs[idx] = -1
            yield from dfs(idx + 1, current_sum - w)
            
            coeffs[idx] = 0
            yield from dfs(idx + 1, current_sum)
            
            coeffs[idx] = 1
            yield from dfs(idx + 1, current_sum + w)
            
            coeffs[idx] = 0

        yield from dfs(0, 0)

    # Build dict for second half
    for s, code in gen_sums(w2):
        if s not in sums2_dict:
            sums2_dict[s] = code
            
        if time.time() > deadline - 30.0:
            break

    sorted_keys = sorted(sums2_dict.keys())
    
    best_abs = None
    best_code1 = None
    best_code2 = None
    
    for s1, code1 in gen_sums(w1):
        target = -s1
        pos = bisect.bisect_left(sorted_keys, target)
        
        candidates = []
        if pos < len(sorted_keys):
            candidates.append(sorted_keys[pos])
        if pos > 0:
            candidates.append(sorted_keys[pos-1])
        if pos + 1 < len(sorted_keys):
            candidates.append(sorted_keys[pos+1])
            
        for s2 in candidates:
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if best_abs is None or abs_total < best_abs:
                best_abs = abs_total
                best_code1 = code1
                best_code2 = sums2_dict[s2]
                
                if best_abs == 1:
                    break
        
        if best_abs == 1:
            break
            
        if time.time() > deadline - 5.0:
            break

    if best_abs is None:
        return [1] + [0]*29

    c1 = decode_coeffs(best_code1, len(w1))
    c2 = decode_coeffs(best_code2, len(w2))
    
    return c1 + c2
