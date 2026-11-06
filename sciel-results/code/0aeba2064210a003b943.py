import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    # Greedy selection
    chosen = []
    for a in range(511, 0, -1):
        if all(__import__('math').gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            
    # Local Search: Try to improve by swapping or adding/removing
    # We can try to replace a chosen number with a smaller one if it allows adding a larger one?
    # Or simply try to add any missing number that is coprime to all.
    # The greedy already does this in order.
    
    # Let's do a few random restarts or perturbations if time permits.
    # But greedy is deterministic and fast.
    
    # Calculate score to verify
    score = sum(1.0 / (512 - a) for a in chosen)
    
    return chosen
