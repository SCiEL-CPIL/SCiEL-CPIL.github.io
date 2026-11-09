"""Cap set greedy construction with 2-gram entropy and Fourier-based refinement."""
import time
import math
import random

_START_TIME = time.time()
TIME_LIMIT = 280  # seconds

# Pre-compute a random seed for deterministic but varied tie-breaking
# We use a fixed seed to ensure reproducibility, but the hash is strong.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def priority(vector, n):
    """
    Static priority function using 2-gram entropy, balance, Fourier magnitude,
    zero-position spread, and a strong hash tie-breaker.
    
    Strategy:
    1. Compute the histogram of all pairwise coordinate values (i, j) where i < j.
    2. Calculate the Shannon entropy of this histogram.
    3. Prefer vectors with higher entropy (more 'random' structure).
    4. Penalize vectors with large magnitude in the lowest non-trivial Fourier frequency.
       This correlates with the vector having a strong linear/structural component
       which may increase the risk of forming 3-term APs.
    5. Reward zero-position spread.
    6. Use a strong hash for tie-breaking.
    """
    if n < 2:
        # Fallback for small n
        h = 0
        for i in range(n):
            h = (h * 31 + vector[i]) % (10**9 + 7)
        return h * 1e-6

    # 1. Count 2-grams
    pairs = [0] * 9
    
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
            
    # Max entropy is log(9) ~ 2.19.
    score = entropy * 100.0
    
    # 3. Balance penalty
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    t = n / 3.0
    balance_penalty = (c0 - t) ** 2 + (c1 - t) ** 2 + (c2 - t) ** 2
    score -= balance_penalty * 1.0
    
    # 4. Fourier-based penalty
    # Compute the magnitude of the DFT at frequency k=1.
    # X_1 = sum_{i=0}^{n-1} v_i * exp(-2*pi*i*i/n)
    # We use the values v_i directly as the signal.
    # This measures the "linear" component of the vector.
    # We want to minimize this magnitude to prefer "random" vectors.
    real_part = 0.0
    imag_part = 0.0
    angle_step = 2.0 * math.pi / n
    for i in range(n):
        angle = -angle_step * i
        real_part += vector[i] * math.cos(angle)
        imag_part += vector[i] * math.sin(angle)
    
    fourier_mag = math.sqrt(real_part * real_part + imag_part * imag_part)
    
    # Normalize the magnitude. Max possible magnitude is sum(|v_i|) <= 2n.
    # So magnitude is in [0, 2n].
    # We subtract a scaled version to penalize high magnitude.
    score -= (fourier_mag / n) * 5.0
    
    # 5. Zero-position spread
    zero_positions = [i for i in range(n) if vector[i] == 0]
    if len(zero_positions) >= 2:
        zero_spread = 0.0
        k = len(zero_positions)
        for a in range(k):
            for b in range(a + 1, k):
                zero_spread += abs(zero_positions[a] - zero_positions[b])
        zero_spread_score = zero_spread / (n * n)
        score += zero_spread_score * 15.0
    
    # 6. Strong Hash Tie-Breaker
    h = 0
    for i in range(n):
        h = (h * 131 + vector[i]) % (10**9 + 7)
        
    score += h * 1e-6
    
    return score
