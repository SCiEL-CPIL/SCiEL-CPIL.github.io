import math

def construct():
    # Use a superincreasing set to maximize the ratio |U-U|/|U+U|
    # U = {0} U {2^0, 2^1, ..., 2^(k-1)}
    # This ensures |U-U| is close to 2*max(U) and |U+U| is polynomial in |U|
    
    k = 500
    U = [0] + [2**i for i in range(k)]
    
    # Calculate |U+U| and |U-U|
    # For performance, we can use sets
    sum_set = set()
    diff_set = set()
    
    # Optimization: Since U is superincreasing, we can compute sizes analytically
    # |U+U| = 1 + k + k*(k-1)/2 = (k^2+k+2)/2 ?
    # Let's just compute it to be safe, k=500 is fast enough (250k ops)
    
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    p = len(sum_set)
    d = len(diff_set)
    max_u = max(U)
    
    # Score calculation
    if p == 0:
        return [0, 1, 3]
        
    val = 1 + math.log(d/p) / math.log(2*max_u + 1)
    
    return U
