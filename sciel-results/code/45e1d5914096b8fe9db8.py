import math
from functools import reduce

def construct():
    # Calculate LCM of 1 to 30
    L = reduce(math.lcm, range(1, 31))
    
    # Weights
    weights = [L // i for i in range(1, 31)]
    
    # Split weights
    w1 = weights[:15]
    w2 = weights[15:]
    
    def generate_sums(w):
        sums = [(0, 0)]
        for i, wt in enumerate(w):
            new_sums = []
            for s, c in sums:
                # c=0, c=1, c=-1
                # encode: bit 2*i and 2*i+1
                # 0 -> 00
                # 1 -> 01
                # -1 -> 10
                # We just need to store the integer representation
                # Let's use base 3 representation for simplicity
                # digit 0: coeff 0
                # digit 1: coeff 1
                # digit 2: coeff -1
                pass
            # This iterative doubling is O(3^N) memory if not careful.
            # Let's do it properly.
            return sums
        
    # Better generation
    def gen(w):
        res = [(0, 0)]
        for i, wt in enumerate(w):
            new_res = []
            for s, code in res:
                # coeff 0
                new_res.append((s, code))
                # coeff 1
                new_res.append((s + wt, code + (1 << (2*i))))
                # coeff -1
                new_res.append((s - wt, code + (2 << (2*i))))
            res = new_res
        return res

    s1_list = gen(w1)
    s2_list = gen(w2)
    
    # Sort by sum
    s1_list.sort(key=lambda x: x[0])
    s2_list.sort(key=lambda x: x[0])
    
    best_abs = float('inf')
    best_i = 0
    best_j = 0
    best_sum = 0
    
    i = 0
    j = len(s2_list) - 1
    
    while i < len(s1_list) and j >= 0:
        sum_val = s1_list[i][0] + s2_list[j][0]
        if sum_val == 0:
            # We need non-zero, so we can't use this.
            # Move away from 0?
            # If we move i++ or j--, we might find a non-zero.
            # Let's just move both? Or standard logic:
            # If sum is 0, it's the best possible absolute value.
            # But we need non-zero.
            # So we should look for the next best.
            # We can increment i and decrement j to explore neighbors.
            i += 1
            j -= 1
        elif sum_val > 0:
            if abs(sum_val) < best_abs:
                best_abs = abs(sum_val)
                best_i = i
                best_j = j
                best_sum = sum_val
            j -= 1
        else: # sum_val < 0
            if abs(sum_val) < best_abs:
                best_abs = abs(sum_val)
                best_i = i
                best_j = j
                best_sum = sum_val
            i += 1
            
    # If best_abs is still inf, something went wrong (e.g. all sums were 0? impossible)
    # Reconstruct coefficients
    code1 = s1_list[best_i][1]
    code2 = s2_list[best_j][1]
    
    coeffs = [0] * 30
    for k in range(15):
        d1 = (code1 >> (2*k)) & 3
        d2 = (code2 >> (2*k)) & 3
        # 0->0, 1->1, 2->-1
        if d1 == 1: coeffs[k] = 1
        elif d1 == 2: coeffs[k] = -1
        else: coeffs[k] = 0
        
        if d2 == 1: coeffs[15+k] = 1
        elif d2 == 2: coeffs[15+k] = -1
        else: coeffs[15+k] = 0
        
    return coeffs
