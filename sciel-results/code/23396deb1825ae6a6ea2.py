import time
import random

def _greedy_build_randomized(order):
    """Build a sequence by trying values in the given order."""
    values = []
    sums_set = set()
    for x in order:
        # Check if adding x creates any duplicate consecutive block sum
        # New consecutive blocks ending at x:
        # [x], [prev, x], [prev2, prev, x], ...
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        # Check for conflicts with existing sums
        conflict = False
        for t in tails:
            if t in sums_set:
                conflict = True
                break
        if not conflict:
            values.append(x)
            sums_set.update(tails)
    return values

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    best_values = []
    best_len = 0
    
    # First, try the deterministic greedy (original approach)
    values, sums = [], set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    if len(values) > best_len:
        best_len = len(values)
        best_values = values[:]
    
    if time.time() > deadline:
        return best_values
    
    # Now try randomized restarts with different shuffles
    all_values = list(range(1, 257))
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        random.shuffle(all_values)
        values = _greedy_build_randomized(all_values)
        
        if len(values) > best_len:
            best_len = len(values)
            best_values = values[:]
    
    return best_values
# EVOLVE-BLOCK-END
