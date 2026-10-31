import math
import time

# Global deadline tracking
_start_time = None
_DEADLINE_FRACTION = 0.95  # Use 95% of time budget

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    A cap set has no three distinct vectors a, b, c with a + b + c ≡ 0 (mod 3).
    
    Strategy:
    1. Primary: Prioritize vectors with all 3 distinct symbols (most "generic").
       Vectors using only 1 or 2 symbols are more likely to form APs.
    2. Secondary: Among vectors with the same distinct count, prioritize those
       with symbol counts close to uniform (n/3 each), measured by chi-squared
       deviation. This makes vectors "generic" so their AP partners are also
       generic, reducing collisions with already-added vectors.
    3. Tertiary: Structural hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Count symbols
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Distinct count (3 is best, 2 is middle, 1 is worst)
    counts = [c0, c1, c2]
    distinct_count = sum(1 for c in counts if c > 0)
    
    # Chi-squared deviation from uniform distribution
    ideal = n / 3.0
    if ideal > 0:
        chi2 = ((c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2) / ideal
    else:
        chi2 = 0.0
    
    # Primary score: distinct count (3 > 2 > 1)
    # Scale to make it dominant: 3 distinct gets +30000, 2 gets +20000, 1 gets +10000
    distinct_score = distinct_count * 10000.0
    
    # Secondary score: lower chi2 = higher priority
    # Increased weight from 10 to 25 for better differentiation among
    # vectors with the same distinct count.
    balance_score = -chi2 * 25.0
    
    # Tertiary: Penalize longest run of identical symbols.
    # Structured vectors (long runs) tend to have AP partners that are
    # also structured, creating clusters of potential conflicts.
    max_run = 1
    current_run = 1
    for i in range(1, n):
        if vector[i] == vector[i-1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
    run_penalty = -max_run * 2.0
    
    # Quaternary: structural hash for deterministic tie-breaking
    h = 0
    base = 31
    for i, val in enumerate(vector):
        h = (h * base + val + i * 7) % (10**9 + 7)
    
    hash_score = h / (10**9 + 7)
    
    return float(distinct_score + balance_score + run_penalty + hash_score)
