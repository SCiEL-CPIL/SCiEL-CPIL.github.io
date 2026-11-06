"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _verify(denoms):
    """Check distinct, in [5, 10^7], sum of reciprocals == 1."""
    if len(denoms) < 1 or len(denoms) > 256:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    for d in denoms:
        if not (5 <= d <= 10000000):
            return False
    s = Fraction(0, 1)
    for d in denoms:
        s += Fraction(1, d)
    return s == 1

def _dp_solve(target, deadline):
    """Solve using subset DP over divisors of target."""
    reached = {0: []}
    for d in range(5, target + 1):
        if time.time() > deadline:
            return None
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                new_values = values + [d]
                if len(new_values) <= 256:
                    reached[total + w] = new_values
        if target in reached:
            return reached[target]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline, 10s safety margin
    
    # Try different targets to find a shorter solution
    # Smaller targets might give shorter solutions
    targets = [2520, 840, 420, 210, 120, 60, 30, 12]
    
    best = None
    for target in targets:
        if time.time() > deadline:
            break
        result = _dp_solve(target, deadline)
        if result is not None and _verify(result):
            if best is None or len(result) < len(best):
                best = result
                if len(best) <= 4:  # Can't do better than 4 terms
                    break
    
    if best is not None:
        return best
    
    # Fallback: use a known valid solution
    # This should never happen, but just in case
    raise RuntimeError("no solution found")
# EVOLVE-BLOCK-END
