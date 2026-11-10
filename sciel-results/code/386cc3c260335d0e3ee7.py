import math
from itertools import product

def construct():
    # Calculate LCM of 1 to 30
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    D = 1
    for i in range(1, 31):
        D = lcm(D, i)
        
    # Weights w_i = D / i
    weights = [D // i for i in range(1, 31)]
    
    # Split into two halves
    w1 = weights[:15]
    w2 = weights[15:]
    
    # Generate all possible sums for the second half (w2)
    # Store in a set for O(1) lookup
    S2 = {0}
    for w in w2:
        new_S2 = set()
        for s in S2:
            new_S2.add(s)
            new_S2.add(s + w)
            new_S2.add(s - w)
        S2 = new_S2
        
    # Precompute suffix sums for w2 for pruning in DFS
    suffix_sums = [0] * 16
    for i in range(14, -1, -1):
        suffix_sums[i] = suffix_sums[i+1] + w2[i]
        
    # DFS to find coefficients for a target sum in the second half
    def solve_i2(idx, current_sum, target):
        if idx == 15:
            if current_sum == target:
                return []
            return None
        
        rem = suffix_sums[idx]
        diff = target - current_sum
        if abs(diff) > rem:
            return None
            
        for c in [0, 1, -1]:
            res = solve_i2(idx + 1, current_sum + c * w2[idx], target)
            if res is not None:
                return [c] + res
        return None

    # Unpack weights for faster access
    w1_0, w1_1, w1_2, w1_3, w1_4, w1_5, w1_6, w1_7, w1_8, w1_9, w1_10, w1_11, w1_12, w1_13, w1_14 = w1
    
    # Iterate through all combinations for the first half (w1)
    for c1 in product([-1, 0, 1], repeat=15):
        s1 = (c1[0]*w1_0 + c1[1]*w1_1 + c1[2]*w1_2 + c1[3]*w1_3 + 
              c1[4]*w1_4 + c1[5]*w1_5 + c1[6]*w1_6 + c1[7]*w1_7 + 
              c1[8]*w1_8 + c1[9]*w1_9 + c1[10]*w1_10 + c1[11]*w1_11 + 
              c1[12]*w1_12 + c1[13]*w1_13 + c1[14]*w1_14)
        
        # Check for target 1
        t1 = 1 - s1
        if t1 in S2:
            c2 = solve_i2(0, 0, t1)
            if c2 is not None:
                return list(c1) + c2
                
        # Check for target -1
        t2 = -1 - s1
        if t2 in S2:
            c2 = solve_i2(0, 0, t2)
            if c2 is not None:
                return list(c1) + c2
                
    # Fallback
    return [0] * 28 + [1, -1]
