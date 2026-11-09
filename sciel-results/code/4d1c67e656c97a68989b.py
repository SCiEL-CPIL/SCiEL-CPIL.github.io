def priority(vector, n):
    # Count zeros
    zeros = sum(1 for x in vector if x == 0)
    # Count non-zero elements (symmetric treatment of 1 and 2)
    non_zero_count = n - zeros
    
    # Primary: more zeros is better
    # Secondary: fewer non-zeros is better
    # We use a large multiplier to ensure zeros dominate.
    # Max non_zero_count is n.
    # Multiplier 10*n + 1 ensures that a difference of 1 in zeros
    # outweighs any difference in non_zero_count.
    
    weight = 10 * n + 1
    score = zeros * weight - non_zero_count
    return float(score)
