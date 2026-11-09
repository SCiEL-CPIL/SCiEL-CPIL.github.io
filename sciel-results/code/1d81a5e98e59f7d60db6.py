"""Cap set greedy construction with refined 2-gram entropy and structural penalties."""
import time
import math
import random

_START_TIME = time.time()
TIME_LIMIT = 280  # seconds

# Pre-compute a random seed for deterministic but varied tie-breaking
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def priority(vector, n):
    """
    Static priority function combining 2-gram entropy, balance, adjacent-pair penalty,
    zero-position spread, and a strong hash tie-breaker.
    
    Strategy:
    1. Compute the histogram of all pairwise coordinate values (i, j) where i < j.
       There are 9 possible pairs: (0,0), (0,1), ..., (2,2).
    2. Calculate the Shannon entropy of this histogram.
    3. Prefer vectors with higher entropy (more 'random' structure).
    4. Reward zero-position spread: zeros distributed across distant coordinates
       are less likely to form 3-term APs since 0 is the additive identity in F_3.
    5. Penalize adjacent identical digits (local structure) to break symmetries.
    6. Reward balanced 1-gram counts (0, 1, 2).
    7. Use a strong hash for tie-breaking.
    """
    if n < 2:
        # Fallback for small n
        h = 0
        for i in range(n):
            h = (h * 31 + vector[i]) % (10**9 + 7)
        return h * 1e-6

    # 1. Count 2-grams
    # pairs[0] = count of (0,0), pairs[1] = count of (0,1), etc.
    # Index: v_i * 3 + v_j
    pairs = [0] * 9
    
    # Optimization: Iterate through all pairs
    # For n=10, 45 pairs. For n=12, 66 pairs. Very fast.
    for i in range(n):
        vi = vector[i]
        for j in range(i + 1, n):
            vj = vector[j]
            idx = vi * 3 + vj
            pairs[idx] += 1
            
    # 2. Calculate Entropy
    total_pairs = n * (n - 1) // 2
    entropy = 0.0
    for count in pairs:
        if count > 0:
            p = count / total_pairs
            entropy -= p * math.log(p)
            
    # We want to MAXIMIZE entropy.
    # Normalize entropy to be in [0, 1] range roughly?
    # Max entropy is log(9) ~ 2.19.
    score = entropy * 100.0
    
    # 3. Add a small bias towards balanced 1-grams (0,1,2 counts)
    #    This helps avoid degenerate cases like all 0s (entropy 0).
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    t = n / 3.0
    balance_penalty = (c0 - t) ** 2 + (c1 - t) ** 2 + (c2 - t) ** 2
    score -= balance_penalty * 1.0  # Mild penalty
    
    # 4. Zero-position spread: reward zeros that are far apart
    #    In F_3, 0 is the additive identity. If zeros are clustered, they're more
    #    likely to participate in symmetric triples. Spread them out.
    zero_positions = [i for i in range(n) if vector[i] == 0]
    if len(zero_positions) >= 2:
        # Sum of pairwise distances between zero positions
        # Normalize by max possible sum for this number of zeros
        zero_spread = 0.0
        k = len(zero_positions)
        for a in range(k):
            for b in range(a + 1, k):
                zero_spread += abs(zero_positions[a] - zero_positions[b])
        # Approximate max: for k zeros, max sum of pairwise distances
        # is achieved by placing them at extremes. For simplicity, use a rough norm.
        # The max sum of pairwise distances for k points in [0, n-1] is roughly
        # k*(k-1)/2 * (n-1) / 2 when evenly spread, but let's just normalize by n^2
        zero_spread_score = zero_spread / (n * n)
        score += zero_spread_score * 15.0  # Weight for zero spread
    
    # 5. Adjacent-pair penalty: penalize local structure
    #    Adjacent identical digits are more likely to form APs with nearby vectors.
    adj_penalty = 0
    for i in range(n - 1):
        if vector[i] == vector[i+1]:
            adj_penalty += 1
    score -= adj_penalty * 5.0  # Moderate penalty
    
    # 6. Strong Hash Tie-Breaker
    #    Use a polynomial hash to ensure unique ordering for identical structures.
    h = 0
    for i in range(n):
        h = (h * 131 + vector[i]) % (10**9 + 7)
        
    # Scale hash to be a small perturbation
    score += h * 1e-6
    
    return score
