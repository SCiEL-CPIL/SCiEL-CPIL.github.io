import zlib

def priority(vector, n):
    # Calculate Hamming weight (number of non-zero coordinates)
    weight = 0
    first_non_zero = 0
    val = 0
    
    # Weights for the heuristic
    # 1. Minimize Hamming weight (sparsity)
    # 2. Prefer first non-zero to be 1 (symmetry breaking)
    # 3. Minimize lexicographic value (deterministic tie-breaker)
    
    for i in range(n):
        x = vector[i]
        if x != 0:
            weight += 1
            if first_non_zero == 0:
                first_non_zero = x
        val += x * (3 ** i)
        
    # Score components
    # Higher score is better.
    # We want to minimize weight, so -weight.
    # We want first_non_zero to be 1. If it is 1, penalty 0. If 2, penalty 1. If 0 (all zero), penalty 0.
    first_penalty = 0 if first_non_zero in [0, 1] else 1
    
    # Combine
    # Scale weight heavily
    score = -1000 * weight - 10 * first_penalty - val
    
    return score
