"""Cap set greedy builder with bitwise parity interleaving and geometric conflict minimization."""
import time
import numpy as np

# Deadline tracking
_start_time = None
_DEADLINE = 330.0 * 0.95  # 313.5s safety margin

# Precompute some constants for speed
# A fixed mask for the XOR operation. We want a mask that has good bit distribution.
# Let's use a mask that is roughly half 1s and half 0s in the relevant bits.
# Since n is small (typically <= 10-12 for these tasks), 3^n is at most ~531441 (for n=12).
# 2^20 is 1048576, so we need about 20 bits.
# Let's pick a mask with a specific pattern to ensure good interleaving.
_MASK = 0x55555555  # 0101 0101 ... pattern

def priority(vector, n):
    """
    Static priority function for cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Prioritize vectors with all 3 distinct symbols (most "generic").
       Vectors using only 1 or 2 symbols are more likely to form APs.
    2. Secondary: Bitwise parity-based interleaving.
       Treat the vector as a base-3 integer. XOR with a fixed mask.
       Count the number of 1-bits in the result.
       This creates a chaotic but deterministic interleaving that breaks
       statistical correlations between local symbol balance and global line formation.
    3. Tertiary: L1 balance deviation from uniform (faster than chi-squared).
    4. Quaternary: Fast structural hash for deterministic tie-breaking.
    
    This combines the distinct-count dominance from the current program
    with the bitwise parity interleaving from the inspirations.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0
    
    # Fast symbol counting
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
    # Scale to make it dominant: 3 distinct gets +30000, 2 gets +20000, 1 gets +10000
    if c0 > 0 and c1 > 0 and c2 > 0:
        distinct_score = 30000.0
    elif c0 > 0 and c1 > 0 or c0 > 0 and c2 > 0 or c1 > 0 and c2 > 0:
        distinct_score = 20000.0
    else:
        distinct_score = 10000.0
    
    # Bitwise parity-based interleaving
    # Convert vector to base-3 integer
    base3_val = 0
    for i, val in enumerate(vector):
        base3_val = base3_val * 3 + val
    
    # XOR with fixed mask
    xored = base3_val ^ _MASK
    
    # Count number of 1-bits (popcount)
    # Python 3.10+ has int.bit_count(), but let's use a compatible method
    # bin(x).count('1') is slow, so use a lookup or bit manipulation
    # For small numbers, we can use a simple loop or bin().count()
    # Since base3_val is at most 3^12 = 531441, xored is also small
    # bin(xored).count('1') is fast enough for small numbers
    parity_score = bin(xored).count('1') * 10.0
    
    # Balance: L1 deviation from uniform
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    balance_score = -dev * 10.0
    
    # Fast hash: simple polynomial rolling hash with bit masking for speed
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    
    hash_score = h / 0x7FFFFFFF
    
    return float(distinct_score + parity_score + balance_score + hash_score)
