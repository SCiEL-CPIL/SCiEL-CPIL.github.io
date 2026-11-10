"""Few-term Egyptian fractions with denominators greater than four."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    # Hardcoded valid solution with 10 terms, all denominators in [5, 10000000]
    # Sum: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/15 + 1/230 + 1/57960 = 1
    denoms = [5, 6, 7, 8, 9, 10, 12, 15, 230, 57960]
    
    # Verification
    if not denoms or len(denoms) > 256:
        return []
    if any(d < 5 or d > 10000000 for d in denoms):
        return []
    if len(set(denoms)) != len(denoms):
        return []
    total = sum(Fraction(1, d) for d in denoms)
    if total != 1:
        return []
        
    return denoms

# EVOLVE-BLOCK-END
