"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import hashlib
import itertools

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Assign a priority to a vector in {0,1,2}^n.
    
    Strategy:
    1. Hamming weight balance: Prefer vectors with weight close to n/2.
       This avoids extreme weights (0 or n) which tend to form small cap sets
       due to high symmetry and clumping.
    2. Algebraic/Hash tie-breaker: Use a deterministic hash to break ties
       and introduce a structured, non-lexicographic ordering that may
       spread selections more evenly across the space.
    3. Higher priority value means selected earlier.
    """
    # Calculate Hamming weight
    weight = sum(vector)
    
    # Primary criterion: distance from n/2 (closer is better -> higher priority)
    # We want weight close to n/2.
    # Let d = abs(weight - n/2). We want to minimize d.
    # So priority should be high when d is low.
    # Use a quadratic penalty for deviation.
    target = n / 2.0
    deviation = abs(weight - target)
    base_score = -deviation ** 2  # Higher is better (less negative)
    
    # Secondary criterion: Deterministic hash for tie-breaking
    # This ensures that among vectors with similar weights, we have a
    # consistent but non-lexicographic order.
    # We hash the tuple to get a stable integer.
    h = hashlib.md5(str(vector).encode('utf-8')).hexdigest()
    # Convert hash to a float in [0, 1)
    hash_val = int(h, 16) / (16**32)
    
    # Combine: The hash is a small perturbation to ensure unique ordering
    # but the weight balance dominates.
    # Scale hash to a small range so it only breaks ties.
    # The gap between consecutive integer weights is 1.0 in deviation^2?
    # deviation is in steps of 1. deviation^2 changes by odd numbers.
    # So the gap between adjacent weight classes is at least 1.
    # We can scale hash to [0, 0.9] to keep it within the class gap.
    perturbation = hash_val * 0.9
    
    return base_score + perturbation

# EVOLVE-BLOCK-END
