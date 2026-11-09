"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n.
    
    Strategy: Probabilistic collision-based priority.
    We estimate the likelihood that a vector forms a 3-term arithmetic progression (3-AP)
    with other vectors in the set. Vectors with coordinates close to the center (1) are
    generally safer than those at the extremes (0 or 2).
    
    We model the 'collision risk' by analyzing the distribution of coordinates:
    1. Extremity Score: Penalize vectors with many 0s or 2s.
    2. Entropy: Prefer vectors with high entropy (mixed values).
    3. Local Density Proxy: Use the variance of the coordinates as a proxy for
       how 'spread out' the vector is in the Hamming cube, which correlates with
       lower local AP density.
    """
    c0 = 0
    c1 = 0
    c2 = 0
    sum_val = 0
    sum_sq = 0
    
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
        sum_val += v
        sum_sq += v * v
        
    # 1. Balance / Entropy
    # Ideal distribution is n/3 for each.
    # We want to maximize the entropy or minimize the deviation from uniform.
    # Using Shannon entropy approximation or just deviation.
    # Deviation from uniform:
    target = n / 3.0
    dev = abs(c0 - target) + abs(c1 - target) + abs(c2 - target)
    
    # 2. Extremity Penalty
    # Vectors with many 0s or 2s are more likely to be part of short APs
    # with other extreme vectors.
    # Let's penalize the count of 0s and 2s.
    extreme_count = c0 + c2
    extremity_penalty = extreme_count / n
    
    # 3. Variance
    # Mean = sum_val / n
    # Variance = sum_sq/n - mean^2
    # High variance means values are spread out (more 0s and 2s), low variance means clustered around 1.
    # We prefer low variance (clustered around 1) because 1 is the 'safe' middle value.
    mean = sum_val / n
    variance = sum_sq / n - mean * mean
    
    # Combine scores
    # We want high priority (large value) for:
    # - Low deviation from uniform (balanced counts)
    # - Low extremity (few 0s/2s)
    # - Low variance (clustered around 1)
    
    # Normalize terms to [0, 1] range roughly
    # dev is in [0, n]
    norm_dev = dev / n
    # extremity_penalty is in [0, 1]
    # variance is in [0, 2/3] (max variance for {0,1,2} is when half 0 and half 2, mean 1, var = 1)
    # Actually max var for {0,1,2} is 1 (0 and 2 only).
    norm_var = variance / 1.0
    
    # Weighting:
    # Balance is important to avoid corners.
    # Extremity is important to avoid APs.
    # Variance is a proxy for extremity but more nuanced.
    
    score = - (2.0 * norm_dev + 1.5 * extremity_penalty + 1.0 * norm_var)
    
    # Deterministic tie-breaker
    # Use a hash-like function to break ties.
    # Simple weighted sum with prime weights to scatter ties.
    tie_break = 0.0
    # Use a simple hash: sum(i * v[i] * prime_i)
    # Primes: 2, 3, 5, 7, 11...
    # Or just a simple polynomial hash mod a large number?
    # Let's use a simple integer hash converted to float.
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v + i) % 1000003
    score += h * 1e-10
    
    return score

# EVOLVE-BLOCK-END
