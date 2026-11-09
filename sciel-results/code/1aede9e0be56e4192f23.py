import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0  # Safety margin

def priority(vector, n):
    # Check deadline? 
    # If we are past the deadline, what should we return?
    # The evaluator is sorting. If we return garbage, it might break.
    # But if the evaluator is stuck in the sorting/insertion phase, it's not calling priority anymore.
    # Priority is called during the generation/sorting phase.
    # So the deadline guard is less relevant here unless n is huge and generation takes long.
    
    # Fast balance calculation
    n0 = 0
    n1 = 0
    n2 = 0
    for v in vector:
        if v == 0:
            n0 += 1
        elif v == 1:
            n1 += 1
        else:
            n2 += 1
            
    # Balance score: minimize variance
    # Ideal count is n/3
    target = n / 3.0
    var = (n0 - target)**2 + (n1 - target)**2 + (n2 - target)**2
    
    # We want higher priority for lower variance
    balance_score = -var
    
    # Tie breaker: Deterministic hash
    # Use a simple hash
    h = 0
    for i, v in enumerate(vector):
        # Simple mixing
        h = (h * 31 + v + i) & 0xFFFFFFFF
        
    # Normalize hash to [0, 1)
    hash_score = (h % 10000) / 10000.0
    
    # Scale balance score to dominate
    # var is at most n^2. 
    # Let's just add the hash as a small fraction
    return balance_score * 100.0 + hash_score
