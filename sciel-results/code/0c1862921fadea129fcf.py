# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

# Deadline guard: the evaluator kills after 330s. We keep a safety margin.
_DEADLINE = 330.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - 30.0 - (time.time() - _START_TIME)

# Cache for priority computation
_priority_cache = {}

def priority(vector, n):
    """
    Prioritize vectors to build a large cap set greedily.
    
    Key insight: A vector v is "dangerous" if it forms many 3-term APs with
    other vectors. We want to prioritize vectors that are LESS likely to form
    APs. 
    
    Strategy:
    1. Spectral smoothness: Vectors with low-frequency Fourier components over
       Z_3^n are more "structured" and less likely to form accidental APs.
    2. Centrality: Vectors with balanced symbol distribution (close to n/3 of each)
       are more central and have more potential partners, so we penalize them.
    3. Hamming weight symmetry breaking: Break ties using a deterministic hash.
    
    The priority is computed as a combination of these signals.
    """
    key = (tuple(vector), n)
    if key in _priority_cache:
        return _priority_cache[key]
    
    # Check deadline - if we're close, return a simple fallback
    if _time_left() < 5.0:
        # Fallback: simple heuristic
        n1 = sum(1 for x in vector if x == 1)
        n2 = sum(1 for x in vector if x == 2)
        n0 = n - n1 - n2
        target = n / 3.0
        centrality = (n0 - target)**2 + (n1 - target)**2 + (n2 - target)**2
        result = -centrality
        _priority_cache[key] = result
        return result
    
    n0 = 0
    n1 = 0
    n2 = 0
    for x in vector:
        if x == 0:
            n0 += 1
        elif x == 1:
            n1 += 1
        else:
            n2 += 1
    
    # 1. Centrality penalty: balanced vectors are more dangerous
    target = n / 3.0
    centrality = (n0 - target)**2 + (n1 - target)**2 + (n2 - target)**2
    # Higher centrality deviation = less balanced = safer
    # We want to MAXIMIZE this, so we add it positively
    
    # 2. Spectral smoothness approximation:
    #    For a vector v in Z_3^n, the DFT magnitude at frequency 0 is constant.
    #    Low-frequency components correspond to slowly varying functions.
    #    We approximate spectral energy by looking at the "roughness" of the vector.
    #    A vector that alternates between values has high frequency content.
    #    We measure this by the number of adjacent coordinate changes.
    
    # Count transitions (adjacent differences)
    transitions = 0
    for i in range(1, n):
        if vector[i] != vector[i-1]:
            transitions += 1
    
    # Normalize transitions: max is n-1
    roughness = transitions / max(n - 1, 1)
    # Higher roughness = more high-frequency = less smooth = potentially safer
    # But we also want some structure, so use a smooth surrogate
    
    # 3. Midpoint risk: vectors with many 1s are more likely to be midpoints
    midpoint_risk = n1 * 1.0
    
    # 4. Variance-based signal
    mean_val = (n0 * 0 + n1 * 1 + n2 * 2) / n if n > 0 else 0
    variance = (n0 * 0 + n1 * 1 + n2 * 4) / n - mean_val**2 if n > 0 else 0
    
    # 5. Deterministic tie-breaking using a hash of the vector
    #    This helps break symmetries and can improve greedy selection
    vec_str = ''.join(str(x) for x in vector)
    hash_val = abs(hash(vec_str)) % 1000000
    tie_break = hash_val / 1000000.0  # [0, 1)
    
    # Combine signals:
    # - centrality: higher is better (less balanced)
    # - roughness: moderate is good (too smooth or too rough can be bad)
    # - midpoint_risk: lower is better
    # - variance: moderate is good
    # - tie_break: small random perturbation for symmetry breaking
    
    # Use a smooth surrogate for the combination
    # We want to maximize: centrality + smoothness_bonus - midpoint_risk
    
    # Smoothness bonus: peaks at moderate roughness
    # Use a Gaussian-like function centered at roughness = 0.3
    smoothness_bonus = math.exp(-((roughness - 0.3)**2) / 0.1)
    
    # Combined score
    score = (centrality * 0.3 + 
             smoothness_bonus * 0.5 - 
             midpoint_risk * 0.2 + 
             variance * 0.1 + 
             tie_break * 0.01)
    
    _priority_cache[key] = score
    return score
# EVOLVE-BLOCK-END
