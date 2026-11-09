"""Search for the minimal non-zero |sum c[i-1]/i| over 30 coefficients in {-1,0,1}."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from collections import deque

_DEADLINE = 75.0  # safety margin under 90s wall-clock kill

def construct():
    t0 = time.time()
    start = Fraction(0, 1)
    # State: (fraction_value, tuple_of_choices_sofar)
    # We do a BFS by depth (number of terms assigned).
    # Use a dict to keep the best (shortest path) to each fraction, but we
    # really just need to explore all reachable fractions at depth 30.
    # To keep memory sane, we store for each depth a dict: fraction -> choices.
    # But number of distinct fractions can be large. Let's estimate:
    # At depth d, the sum is over a subset of {1/1..1/d} with signs.
    # The denominator divides lcm(1..d). For d=30, lcm is ~ 2.3e12.
    # Number of distinct sums is bounded by 3^30 ~ 2e14, too many.
    # But many collide. In practice, the number of distinct fractions at
    # depth 30 is much smaller. Let's try.
    
    # Actually, a smarter approach: BFS where we track the set of reachable
    # fractions at each depth. Start with {0}. At each step, from each fraction
    # we can add +1/i, -1/i, or 0.
    
    # But we also need to reconstruct the coefficients. Let's store parent pointers.
    # Given time constraints, let's do a BFS that stores for each fraction at
    # depth d the best (any) path to get there. We'll use a dict mapping
    # fraction to the tuple of choices that produced it (first one found).
    
    # State at depth d: dict from Fraction to tuple of choices (length d)
    current = {start: ()}
    
    best_frac = None
    best_abs = None
    best_choices = None
    
    for d in range(1, 31):
        if time.time() > _DEADLINE:
            break
        next_layer = {}
        inv = Fraction(1, d)
        for frac, choices in current.items():
            # Three options: +1/d, -1/d, 0
            for delta, bit in [(inv, 1), (-inv, -1), (Fraction(0,1), 0)]:
                new_frac = frac + delta
                new_choices = choices + (bit,)
                if new_frac not in next_layer:
                    next_layer[new_frac] = new_choices
        
        current = next_layer
        
        # Check for smallest non-zero absolute value
        for frac in current:
            if frac == 0:
                continue
            abs_frac = abs(frac)
            if best_abs is None or abs_frac < best_abs:
                best_abs = abs_frac
                best_frac = frac
                best_choices = current[frac]
        
        # If we reached depth 30, we can stop
        if d == 30:
            break
    
    if best_choices is None:
        # Fallback: the seed program's answer
        return [0] * 28 + [1, -1]
    
    return list(best_choices)
# EVOLVE-BLOCK-END
