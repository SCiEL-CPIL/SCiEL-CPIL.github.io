"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    
    Strategy:
    1. Compute a base score based on a linear form over F_3 (treating 0,1,2 as field elements).
       We use multiple random-looking linear forms (fixed seeds for determinism) to break ties
       and create a diverse ordering.
    2. Add a small perturbation based on the vector's "structure" to avoid degeneracies.
    
    The key insight for cap sets is that we want to avoid 3-term arithmetic progressions.
    In F_3^n, a 3-term AP is x, y, z where x+z = 2y (component-wise mod 3).
    Equivalently, for any two distinct vectors a, b in the set, a+b must not be in the set
    (where addition is component-wise mod 3, and we check if a+b is a valid vector).
    
    Actually, the condition is: no three distinct elements x, y, z in the set satisfy x + y + z = 0 mod 3 (component-wise).
    Equivalently, no three elements form a 3-term AP: x, y, z with x + z = 2y.
    
    A better greedy heuristic: prioritize vectors that are "far" from existing pairs.
    But since priority is called before we know the set, we need a static ordering.
    
    Alternative approach: Use a randomized linear priority that correlates with being part of fewer 3-term APs.
    Vectors with more "balanced" coordinate distributions might be better.
    
    Let's use: sum of coordinates mod 3, weighted by position, plus a hash-like term.
    """
    if n == 0:
        return 0.0
    
    # Convert to numpy for vectorized operations
    v = np.array(vector, dtype=np.int64)
    
    # Multiple linear forms to create diverse ordering
    # Use fixed pseudo-random weights for determinism
    # Weight 1: alternating signs
    w1 = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.float64)
    score1 = np.dot(v, w1)
    
    # Weight 2: quadratic-like pattern
    w2 = np.array([(i * i) % 3 for i in range(n)], dtype=np.float64)
    score2 = np.dot(v, w2)
    
    # Weight 3: prime-based
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    w3 = np.array([primes[i % len(primes)] for i in range(n)], dtype=np.float64)
    score3 = np.dot(v, w3)
    
    # Combine with different weights
    combined = score1 * 1.0 + score2 * 0.7 + score3 * 0.3
    
    # Add a small term based on number of non-zero coordinates
    non_zero = np.count_nonzero(v)
    combined += non_zero * 0.01
    
    # Add a hash-like term for tie-breaking
    # Use a simple hash based on the vector
    h = 0
    for i, val in enumerate(v):
        h = (h * 31 + val * (i + 1)) % 1000003
    combined += h * 1e-6
    
    return float(combined)

# EVOLVE-BLOCK-END
