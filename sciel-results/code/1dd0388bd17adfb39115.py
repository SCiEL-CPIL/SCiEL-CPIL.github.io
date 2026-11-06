import time
import math

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Use count for speed
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    target = n / 3.0
    
    # Sum of squared deviations
    dev = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
    
    # Primary score: -dev
    # Secondary score: hash to break ties
    # We want a float return.
    # We can combine them.
    # Let's use a small epsilon for the hash to not disturb the main score too much.
    # Or just return -dev. The evaluator might use a stable sort, so ties are broken by input order.
    # If input order is lexicographic, ties are broken lexicographically.
    # To decorrelate, we can add a small hash component.
    
    # A simple hash: sum of (v * 3**i) mod something?
    # Or just use the tuple itself in a hash?
    # Python's hash is deterministic within a run, but might vary between runs?
    # The task requires deterministic behavior for the score?
    # "Score: cap set size". The size is an integer.
    # The priority just guides the order.
    # If the order changes, the final cap set size might change.
    # We want a robust order.
    
    # Let's stick to -dev for now, as it was the measured improvement.
    # If we want to add tie-breaking, we can use a pseudo-random number based on the vector.
    # But let's keep it simple first.
    
    return -dev

# EVOLVE-BLOCK-END
