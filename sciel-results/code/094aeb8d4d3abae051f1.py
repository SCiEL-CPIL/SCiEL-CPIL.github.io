import time
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocal floats for fast pruning
    recip_floats = [1.0 / d for d in denominators]
    
    # Precompute suffix sums of reciprocals
    # suffix[i] = sum of recip_floats[j] for j in range(i, n)
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + recip_floats[i]
    
    best_sum_float = 0.0
    best_sum_exact = Fraction(0)
    best_indices = []
    
    # Initial greedy solution to set a good baseline
    # Greedy: pick smallest denominators that fit
    curr_sum_exact = Fraction(0)
    curr_sum_float = 0.0
    curr_indices = []
    for i, d in enumerate(denominators):
        if curr_sum_float + recip_floats[i] < 1.0:
            curr_sum_float += recip_floats[i]
            curr_sum_exact += Fraction(1, d)
            curr_indices.append(i)
    
    # Verify exact sum is < 1
    if curr_sum_exact < 1:
        best_sum_float = curr_sum_float
        best_sum_exact = curr_sum_exact
        best_indices = curr_indices[:]
    
    start_time = time.time()
    deadline = 85.0  # 5 sec margin
    
    path = []
    
    def dfs(i, curr_sum_f, curr_sum_e):
        nonlocal best_sum_float, best_sum_exact, best_indices
        
        if time.time() > deadline:
            return
            
        # Pruning: if current sum + max possible remaining sum <= best, prune
        if curr_sum_f + suffix[i] <= best_sum_float:
            return
            
        if i == n:
            # Leaf node
            if curr_sum_f > best_sum_float and curr_sum_e < 1:
                best_sum_float = curr_sum_f
                best_sum_exact = curr_sum_e
                best_indices = path[:]
            return
        
        # Try including current item
        next_f = curr_sum_f + recip_floats[i]
        if next_f < 1.0:
            # Check exact sum to ensure validity
            next_e = curr_sum_e + Fraction(1, denominators[i])
            if next_e < 1:
                path.append(i)
                dfs(i + 1, next_f, next_e)
                path.pop()
        
        # Try excluding current item
        dfs(i + 1, curr_sum_f, curr_sum_e)

    dfs(0, 0.0, Fraction(0))
    
    return best_indices
