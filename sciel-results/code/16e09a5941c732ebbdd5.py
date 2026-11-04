import math

def score_set(U):
    U = sorted(U)
    if not U or U[0] != 0:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute |U+U|
    sum_set = set()
    for i in range(len(U)):
        for j in range(i, len(U)):
            sum_set.add(U[i] + U[j])
    
    # Compute |U-U|
    diff_set = set()
    for i in range(len(U)):
        for j in range(len(U)):
            diff_set.add(U[i] - U[j])
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def construct():
    best_set = [0]
    best_score = 0
    
    # Generate Mersenne-like sets: {2^k - 1}
    # Test small N
    for N in range(3, 15):
        U = [2**k - 1 for k in range(N)]
        # U[0] is 0
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
            
    return best_set
