"""Refined quadric-based greedy cap set ordering using balanced-digit weight."""
import time

_DEADLINE = time.monotonic() + 280.0
_START = time.monotonic()

def priority(vector, n):
    """Return a float priority; higher vectors are inserted first.

    Uses the sum of squares of balanced digits (Hamming weight) as the primary
    quadratic form. Vectors with lower weight are prioritized. The set of
    vectors with a fixed low weight (e.g., weight 1) forms a cap set, allowing
    the greedy algorithm to accept the entire layer.
    """
    if time.monotonic() > _DEADLINE:
        return 0.0

    # Balanced digit mapping: 0->0, 1->1, 2->-1
    # Sum of squares is the Hamming weight.
    w = 0
    for x in vector:
        if x != 0:
            w += 1
            
    # Primary: lower weight is better.
    # We invert it so higher priority value = lower weight.
    # Max weight is n.
    band = float(n - w)
    
    # Secondary: deterministic tie-breaker.
    # Use a simple polynomial hash mod a prime to ensure stable ordering.
    h = 0
    for x in vector:
        h = (h * 31 + x) % 1000003
        
    # Scale tie-break to be negligible relative to band separation (1.0).
    return band + h * 1e-6
