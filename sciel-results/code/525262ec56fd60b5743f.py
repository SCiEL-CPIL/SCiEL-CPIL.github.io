def priority(vector, n):
    zeros = sum(1 for x in vector if x == 0)
    ones = sum(1 for x in vector if x == 1)
    twos = sum(1 for x in vector if x == 2)
    
    # Primary: more zeros is better
    # Secondary: prefer vectors with a mix of 1s and 2s (higher interaction count)
    # This is based on the idea that 1s and 2s do not form 3-APs with each other,
    # so vectors containing both are "safer" in a mixed set.
    
    weight = 10 * n + 1
    # The product ones*twos is 0 if pure, max if balanced.
    # Max product is (n/2)^2 ~ n^2/4.
    # We add it to the score.
    score = zeros * weight + ones * twos
    return float(score)
