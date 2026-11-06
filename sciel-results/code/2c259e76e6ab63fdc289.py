"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    selected = []
    reachable = {0}
    
    for x in range(100, 0, -1):
        if time.time() > deadline:
            break
        # Check if adding x would create sum 200
        # 200 is reachable iff (200 - x) is in reachable
        if (200 - x) in reachable:
            continue
        # Safe to add x
        selected.append(x)
        # Update reachable sums (only those <= 200 matter)
        new_sums = {s + x for s in reachable if s + x <= 200}
        reachable |= new_sums
    
    # Fallback: if somehow empty, return a trivially valid set
    if not selected:
        selected = [1]
    
    return selected
# EVOLVE-BLOCK-END
