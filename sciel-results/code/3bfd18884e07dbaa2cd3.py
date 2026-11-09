def priority(vector, n):
    # Count zeros
    zeros = sum(1 for x in vector if x == 0)
    # Count non-zeros (treat 1 and 2 equally due to symmetry in F3 cap sets)
    non_zeros = n - zeros
    
    # Primary: more zeros is better
    # Secondary: fewer non-zeros is better
    # We combine them into a single float.
    # To ensure zeros dominate, we use a large multiplier.
    # Max non_zeros is n.
    # So multiplier should be > n.
    # Let's use 10*n + 1 to be safe.
    
    weight = 10 * n + 1
    score = zeros * weight - non_zeros
    return float(score)
